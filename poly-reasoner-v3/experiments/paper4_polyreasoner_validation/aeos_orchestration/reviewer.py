import litellm
import os
import contextlib
import re

litellm.drop_params = True

REVIEWER_SYSTEM_PROMPT = """You are the Lead ML Strategist (ReviewerAgent).
You oversee a CoderAgent that builds classification models. 

DATASET CONTEXT:
- n_features: {n_features}
- n_classes: {n_classes}
- Training samples: {n_train}
- Validation samples: {n_val}
- Dataset hint: {dataset_hint}

YOUR GOAL: Analyze the execution history of the CoderAgent and determine the next best step.
- Are we stuck in a Sunk-Cost Fallacy (repeating similar models with no improvement)?
- Have we hit a mathematical plateau?
- If the current path is failing, suggest a completely different model family (e.g. from Trees to SVM, or from Sklearn to PyTorch).

STOPPING RULES (you MUST follow these strictly):
1. You CANNOT issue STOP before iteration 5 (minimum exploration floor).
2. You CANNOT issue STOP unless at least 3 DIFFERENT model families have been tried.
3. You SHOULD issue STOP when: accuracy has not improved for 8+ consecutive successful iterations AND you have tried 3+ families.
4. You SHOULD issue STOP when: the last 5 iterations all produced errors or very low accuracy with no recovery path.

If you believe no further improvement is likely AND the stopping rules are satisfied, output exactly: DIRECTIVE: STOP

If we should continue, provide a clear, one-sentence instruction for the CoderAgent.
Start your instruction with exactly: DIRECTIVE: <your instruction here>

Example 1 (too early — must continue):
History shows 2 iterations with RandomForest at 92% accuracy. Only 1 family tried.
DIRECTIVE: Try a Support Vector Machine (SVC with RBF kernel) after StandardScaler preprocessing.

Example 2 (valid stop):
History shows 10 iterations. Tried RandomForest, GradientBoosting, SVM. Accuracy stuck at 92.9% for 7 iters.
DIRECTIVE: STOP
"""

MATH_REVIEWER_SYSTEM_PROMPT = """You are the Lead ML Strategist (ReviewerAgent) running a mathematical self-reflection gate.
You oversee a CoderAgent that builds classification models.

DATASET CONTEXT:
- n_features: {n_features}
- n_classes: {n_classes}
- Training samples: {n_train}
- Validation samples: {n_val}
- Dataset hint: {dataset_hint}

MANDATORY YIELD CALCULATION (you MUST do this exactly):
1) Compute V = number of valid (non-error) iterations in the last 5 runs (max 5).
2) Compute accuracy improvement delta from the latest successful step:
   delta = current_best_accuracy - previous_best_accuracy
3) Set multiplier M by rule:
   - M = 1.5 if delta > 0.001
   - M = 0.1 if delta < -0.005 OR V = 0
   - M = 0.5 otherwise
4) Compute Yield: Y = (V / 5) * M
5) If Y < 0.45, you MUST output exactly: DIRECTIVE: STOP

OUTPUT FORMAT REQUIREMENTS:
- Show step-by-step math for V, delta, M, and Y.
- End with a final line beginning with exactly: DIRECTIVE:
- If Y < 0.45, final line MUST be exactly: DIRECTIVE: STOP
- If Y >= 0.45, provide exactly one concise actionable instruction for the coder:
  DIRECTIVE: <instruction>
"""


class ReviewerAgent:
    def __init__(self, model="ollama/ministral:14b", api_key=None, api_base=None, dataset_hint="", use_math_prompt=False):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.dataset_hint = dataset_hint
        self.use_math_prompt = use_math_prompt
        self.last_raw_response = ""
        self.last_math_inputs = None
        
    def get_directive(self, n_features, n_classes, n_train, n_val, history):
        system_template = MATH_REVIEWER_SYSTEM_PROMPT if self.use_math_prompt else REVIEWER_SYSTEM_PROMPT
        system_str = system_template.format(
            n_features=n_features, n_classes=n_classes,
            n_train=n_train, n_val=n_val, dataset_hint=self.dataset_hint
        )

        # Compute context stats for the prompt
        current_iteration = len(history) + 1
        families_tried = set()
        for h in history:
            if h.get('family') and h['family'] not in ('REVIEWER_STOP', 'CODER_ERROR', 'Unknown'):
                families_tried.add(h['family'])
        n_families = len(families_tried)

        prompt = "Here is the execution history:\n\n"
        if not history:
            prompt += "No history yet. This is Iteration 1. Start with a robust baseline like Random Forest or Gradient Boosting.\n"
        else:
            for h in history:
                prompt += f"Iter {h['iteration']}: "
                if h.get('error'):
                    prompt += f"FAILED -> {h['error'][:150]}\n"
                else:
                    marker = " *** BEST ***" if h.get('is_best') else ""
                    val_acc = h.get("val_accuracy")
                    val_loss = h.get("val_loss")
                    if isinstance(val_acc, (int, float)) and isinstance(val_loss, (int, float)):
                        prompt += f"Acc: {val_acc:.4f} | Loss: {val_loss:.4f} | Model: {h.get('family', '?')}{marker}\n"
                    else:
                        prompt += f"NO_METRIC -> Model: {h.get('family', '?')}{marker}\n"
                    if 'code' in h and h['code']:
                        code_snippet = h['code'].splitlines()[:3]
                        prompt += f"    Code preview: {' '.join(code_snippet)}...\n"

        prompt += f"\n[CONTEXT] Current iteration: {current_iteration} | Families tried: {n_families} ({', '.join(sorted(families_tried)) or 'none yet'})"
        prompt += f"\n[REMINDER] You CANNOT issue STOP before iteration 5 or before 3 families are tried."

        if self.use_math_prompt:
            last_five = history[-5:]
            # Count only successful trials (non-error + real numeric metric). This excludes timeouts,
            # runtime failures, and sentinel rows like REVIEWER_STOP that have no metrics.
            valid_last_five = [
                h
                for h in last_five
                if (not h.get("error")) and isinstance(h.get("val_accuracy"), (int, float))
            ]

            successful = [h for h in history if not h.get('error') and isinstance(h.get('val_accuracy'), (int, float))]
            if successful:
                best_values = []
                running = float("-inf")
                for h in successful:
                    running = max(running, float(h['val_accuracy']))
                    best_values.append(running)
                current_best = best_values[-1]
                previous_best = best_values[-2] if len(best_values) >= 2 else best_values[-1]
            else:
                current_best = 0.0
                previous_best = 0.0

            delta = current_best - previous_best
            self.last_math_inputs = {
                "v_last5_valid": len(valid_last_five),
                "previous_best_accuracy": float(previous_best),
                "current_best_accuracy": float(current_best),
                "delta": float(delta),
                "m_rule": "1.5 if delta>0.001 else 0.1 if delta<-0.005 or V=0 else 0.5",
            }
            prompt += (
                f"\n[MATH INPUTS] Last-5 valid count V={len(valid_last_five)}; "
                f"previous_best_accuracy={previous_best:.6f}; "
                f"current_best_accuracy={current_best:.6f}; delta={delta:.6f}."
            )
            prompt += (
                "\n[MATH RULE] Use these exact rules for M: "
                "1.5 if delta > 0.001; 0.1 if delta < -0.005 or V=0; else 0.5. "
                "Then compute Y=(V/5)*M."
            )
            prompt += "\n[MATH GATE] If Y < 0.45, final line must be exactly: DIRECTIVE: STOP"
        else:
            self.last_math_inputs = None

        prompt += "\nWhat is your next directive? Respond with your reasoning, ending with DIRECTIVE: <action or STOP>."

        return self._call_llm(system_str, prompt, current_iteration, n_families)

    def _call_llm(self, system_str, prompt, current_iteration=99, n_families=99):
        try:
            messages = [
                {"role": "system", "content": system_str},
                {"role": "user", "content": prompt}
            ]
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3, # low temp for consistent reasoning
                "max_tokens": 512,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base

            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                response = litellm.completion(**kwargs)
            
            raw_output = response.choices[0].message.content
            self.last_raw_response = raw_output or ""
            print(f"\n  [REVIEWER] Response:")
            for line in raw_output.splitlines():
                print(f"    {line}")
                
            return self._extract_directive(raw_output, current_iteration, n_families)

        except Exception as e:
            self.last_raw_response = f"[REVIEWER ERROR] {type(e).__name__}: {str(e)[:200]}"
            print(f"\n  [REVIEWER ERROR] {type(e).__name__}: {str(e)[:200]}")
            return "Try a different approach to improve accuracy."

    def _extract_directive(self, text, current_iteration=99, n_families=99):
        # Prioritize explicit STOP
        if "STOP" in text.upper().split():
            directive = "STOP"
        else:
            match = re.search(r"DIRECTIVE:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
            if match:
                directive = match.group(1).strip()
            else:
                # If the LLM failed to format properly, just pass its entire reasoning to the coder
                directive = text.strip()

        # Hard guard: override premature STOPs
        if directive.upper().startswith("STOP"):
            if current_iteration < 5:
                print(f"  [REVIEWER GUARD] STOP overridden — only {current_iteration} iterations done (min 5).")
                return "We are still in early exploration. Try a completely different algorithm family (e.g. SVM, GradientBoosting, or a neural network)."
            if n_families < 3:
                print(f"  [REVIEWER GUARD] STOP overridden — only {n_families} family/families tried (min 3).")
                return "You haven't explored enough model families. Try a completely different approach (e.g. SVM or neural network if you've only used tree models)."

        return directive
