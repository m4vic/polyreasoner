import litellm
import os
import contextlib
import re

litellm.drop_params = True

CODER_SYSTEM_PROMPT = """You are a CoderAgent (ML Engineer).
You have a classification dataset. Here is everything you know:
- n_features: {n_features}
- n_classes: {n_classes}
- Training samples: {n_train}
- Validation samples: {n_val}

DATASET TYPE: {dataset_hint}

YOUR TASK: Write a Python function `solve(X_train, y_train, X_val, y_val)` that:
1. Builds and trains the model specified in the DIRECTIVE.
2. Returns predictions as a numpy array of shape (n_val,) with integer class labels.

Available in your namespace: numpy (as np), sklearn submodules, torch, nn, optim.

RULES:
1. You MUST define: def solve(X_train, y_train, X_val, y_val): ... return predictions
2. predictions must be a numpy array of integers in range [0, {max_class}]
3. Your code has a {timeout}-second time limit. Be efficient.
4. Output ONLY the code inside ```python ... ```. No explanations.
"""

class CoderAgent:
    def __init__(self, model="ollama/deepseek-coder:6.7b", api_key=None, api_base=None, dataset_hint=""):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.dataset_hint = dataset_hint

    def _detect_model_family(self, code):
        code_lower = code.lower()
        if 'randomforest' in code_lower: return 'RandomForest'
        if 'gradientboosting' in code_lower: return 'GradientBoosting'
        if 'extratrees' in code_lower: return 'ExtraTrees'
        if 'svc' in code_lower or 'svm' in code_lower: return 'SVM'
        if 'logisticregression' in code_lower: return 'LogisticRegression'
        if 'mlpclassifier' in code_lower: return 'sklearn_MLP'
        if 'nn.module' in code_lower or 'nn.linear' in code_lower: return 'PyTorch_NN'
        if 'votingclassifier' in code_lower or 'stackingclassifier' in code_lower: return 'Ensemble'
        return 'Unknown'

    def generate_code(self, n_features, n_classes, n_train, n_val, directive, history, best_code=None, timeout=120):
        system_str = CODER_SYSTEM_PROMPT.format(
            n_features=n_features, n_classes=n_classes,
            n_train=n_train, n_val=n_val,
            max_class=n_classes - 1, timeout=timeout,
            dataset_hint=self.dataset_hint
        )
        
        prompt = f"DIRECTIVE FROM REVIEWER: {directive}\n\n"
        
        if history and history[-1].get('error'):
            prompt += f"WARNING: The previous attempt resulted in an error:\n{history[-1]['error']}\nPlease fix this if applicable.\n\n"
        
        if best_code:
            prompt += "For context, here is the current best working code:\n```python\n"
            prompt += best_code + "\n```\n\n"
            
        prompt += "Now, output the new code inside ```python ... ```."

        return self._call_llm(system_str, prompt)

    def _call_llm(self, system_str, prompt):
        try:
            messages = [
                {"role": "system", "content": system_str},
                {"role": "user", "content": prompt}
            ]
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
            }
            if self.api_key: kwargs["api_key"] = self.api_key
            if self.api_base: kwargs["api_base"] = self.api_base

            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                response = litellm.completion(**kwargs)
            
            raw_output = response.choices[0].message.content
            return self._extract_code(raw_output)
            
        except Exception as e:
            print(f"\n  [CODER ERROR] {type(e).__name__}: {str(e)[:200]}")
            raise e

    def _extract_code(self, text):
        stripped = text.strip()
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
        
        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match: return match.group(1).strip()
        
        code = stripped
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"```\s*$", "", code)
        return code.strip()
