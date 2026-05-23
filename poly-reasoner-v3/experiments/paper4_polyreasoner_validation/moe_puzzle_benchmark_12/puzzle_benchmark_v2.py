"""
Paper 3 — Thread B: Puzzle Benchmark v2 (Fixed)
=================================================
Fixes from v1:
  - Handle qwen3.5 extended-thinking (<think> tags) properly
  - Replace missing mistral:7b with available models
  - Test multiple MoE panels for diversity evidence
  - Add deepseek-r1:8b as a reasoning baseline

Configurations per panel:
  A) Single (no CoT)   - basic single model
  B) Single + CoT      - same model with chain-of-thought
  C) MoE-Vote          - 3 diverse models, majority vote
  D) MoE-Synth         - 3 diverse models + synthesizer reconciles
"""

import os
import json
import time
import re
from datetime import datetime
from collections import Counter

import ollama

# ─── Configuration ────────────────────────────────────────────────────────
OUTPUT_DIR = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\aeos_behave\paper3_thread_b"

# ─── Panel Definitions ────────────────────────────────────────────────────
# Each panel tests a different combination
PANELS = {
    "panel_A": {
        "name": "Mixed-Family Panel",
        "single_model": "qwen3.5:9b",
        "moe_models": ["qwen2.5-coder:7b", "llama3.1:8b", "deepseek-coder:6.7b"],
        "synth_model": "qwen3.5:9b",
    },
    "panel_B": {
        "name": "Reasoning-Heavy Panel",
        "single_model": "deepseek-r1:8b",
        "moe_models": ["deepseek-r1:8b", "qwen3.5:4b", "gemma4:latest"],
        "synth_model": "deepseek-r1:8b",
    },
    "panel_C": {
        "name": "Small-Models Panel",
        "single_model": "qwen3.5:4b",
        "moe_models": ["phi3:mini", "qwen2.5-coder:3b", "qwen3.5:4b"],
        "synth_model": "qwen3.5:4b",
    },
}

# ─── Puzzle Set ───────────────────────────────────────────────────────────
PUZZLES = [
    # --- Logic Puzzles ---
    {
        "id": "logic_01", "category": "logic", "difficulty": "easy",
        "question": "If all Bloops are Razzles, and all Razzles are Lazzles, are all Bloops definitely Lazzles?",
        "answer": "yes",
        "answer_check": lambda r: "yes" in r.lower()
    },
    {
        "id": "logic_02", "category": "logic", "difficulty": "medium",
        "question": (
            "Three friends — Alice, Bob, Carol — each pick a different number from {1, 2, 3}. "
            "Alice does not pick 1. Bob picks a higher number than Alice. "
            "What number does Carol pick?"
        ),
        "answer": "1",
        "answer_check": lambda r: bool(re.search(r'\b1\b', r))
    },
    {
        "id": "logic_03", "category": "logic", "difficulty": "hard",
        "question": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        "answer": "9",
        "answer_check": lambda r: bool(re.search(r'\b9\b', r))
    },
    {
        "id": "logic_04", "category": "logic", "difficulty": "hard",
        "question": (
            "You have a 3-gallon jug and a 5-gallon jug, no markings. "
            "How do you measure exactly 4 gallons? Give the minimum number of steps."
        ),
        "answer": "6",
        "answer_check": lambda r: bool(re.search(r'\b6\b', r)) or "six" in r.lower()
    },
    # --- Math Word Problems ---
    {
        "id": "math_01", "category": "math", "difficulty": "easy",
        "question": (
            "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost in cents?"
        ),
        "answer": "5",
        "answer_check": lambda r: bool(re.search(r'\b5\b', r)) and not bool(re.search(r'\b10\b', r[:50]))
    },
    {
        "id": "math_02", "category": "math", "difficulty": "medium",
        "question": (
            "If it takes 5 machines 5 minutes to make 5 widgets, "
            "how many minutes does it take 100 machines to make 100 widgets?"
        ),
        "answer": "5",
        "answer_check": lambda r: bool(re.search(r'\b5\b', r))
    },
    {
        "id": "math_03", "category": "math", "difficulty": "hard",
        "question": (
            "A lily pad doubles in size every day. It takes 48 days to cover half the pond. "
            "How many days to cover the whole pond?"
        ),
        "answer": "49",
        "answer_check": lambda r: bool(re.search(r'\b49\b', r))
    },
    # --- Lateral Thinking ---
    {
        "id": "lateral_01", "category": "lateral", "difficulty": "medium",
        "question": (
            "A man walks into a restaurant and orders albatross soup. "
            "He takes one sip and immediately goes home and kills himself. Why?"
        ),
        "answer": "He realized the soup was not real albatross — it meant his wife had died on the island.",
        "answer_check": lambda r: (
            "wife" in r.lower() or "island" in r.lower() or "shipwreck" in r.lower()
        )
    },
    {
        "id": "lateral_02", "category": "lateral", "difficulty": "easy",
        "question": "A woman shoots her husband, then dines with him that evening. How?",
        "answer": "She is a photographer. She shot him with a camera.",
        "answer_check": lambda r: "photo" in r.lower() or "camera" in r.lower()
    },
    # --- Constraint Satisfaction ---
    {
        "id": "constraint_01", "category": "constraint", "difficulty": "medium",
        "question": (
            "Four people need to cross a bridge at night with one torch. "
            "The bridge holds max 2 people. Crossing times: A=1min, B=2min, C=5min, D=10min. "
            "A pair crosses at the slower person's speed. The torch must be walked back. "
            "What is the minimum time for all 4 to cross?"
        ),
        "answer": "17",
        "answer_check": lambda r: bool(re.search(r'\b17\b', r))
    },
    # --- Trick Questions ---
    {
        "id": "trick_01", "category": "trick", "difficulty": "easy",
        "question": "How many months have 28 days?",
        "answer": "12",
        "answer_check": lambda r: bool(re.search(r'\b12\b', r)) or "all" in r.lower() or "twelve" in r.lower()
    },
    {
        "id": "trick_02", "category": "trick", "difficulty": "medium",
        "question": "A rooster lays an egg on top of a barn roof. Which way does it roll?",
        "answer": "Roosters don't lay eggs.",
        "answer_check": lambda r: "rooster" in r.lower() and ("don't" in r.lower() or "cannot" in r.lower() or "not" in r.lower() or "can't" in r.lower())
    },
]


# ─── LLM Call (with thinking-model fix) ──────────────────────────────────
def strip_thinking(text):
    """Remove <think>...</think> blocks from extended-thinking models."""
    # Remove complete think blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also handle unclosed think blocks (model cut off mid-thinking)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def ask_model(model, question, system_prompt=None, timeout=120):
    """Synchronous Ollama call. Handles thinking models."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    try:
        start = time.time()
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.1, "num_predict": 1024}
        )
        elapsed = time.time() - start

        content = response["message"].get("content", "").strip()

        # Handle thinking models (qwen3.5, deepseek-r1)
        if "<think>" in content:
            visible = strip_thinking(content)
            if visible:
                content = visible
            # else keep original (thinking was ALL of it)

        return content, elapsed
    except Exception as e:
        return f"ERROR: {e}", 0.0


# ─── Answer Extraction ───────────────────────────────────────────────────
def extract_answer(text):
    """Extract final answer from model output."""
    if not text or text.startswith("ERROR:"):
        return text or ""

    # Look for ANSWER: or FINAL ANSWER: patterns
    for pattern in [r'FINAL ANSWER:\s*(.+)', r'ANSWER:\s*(.+)', r'Answer:\s*(.+)']:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fall back to last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text[:200]


# ─── System Prompts ──────────────────────────────────────────────────────
SINGLE_SYSTEM = (
    "You are a precise reasoning assistant. Answer the question directly and show your work. "
    "End with your final answer clearly labeled as: FINAL ANSWER: <your answer>"
)

COT_SYSTEM = """You are a precise reasoning assistant using chain-of-thought.
STEP 1: Restate the problem in your own words.
STEP 2: Identify any traps or trick elements.
STEP 3: Reason through it step-by-step.
STEP 4: State your final answer on the last line as: FINAL ANSWER: <your answer>"""

MOE_EXPERT_SYSTEM = "You are a precise reasoning assistant. Answer concisely. End with: FINAL ANSWER: <answer>"

SYNTH_SYSTEM = """You are a synthesis judge. You receive answers from multiple models on the same puzzle.
Your job:
1. Identify where they agree and where they disagree.
2. If they disagree, reason about which answer is correct and why.
3. State the final consensus answer clearly as: FINAL ANSWER: <answer>"""


# ─── Run Configurations ─────────────────────────────────────────────────
def run_single(puzzle, model):
    answer, latency = ask_model(model, puzzle["question"], SINGLE_SYSTEM)
    extracted = extract_answer(answer)
    correct = puzzle["answer_check"](answer)
    return {"raw": answer, "extracted": extracted, "correct": correct, "latency": round(latency, 2)}


def run_single_cot(puzzle, model):
    answer, latency = ask_model(model, puzzle["question"], COT_SYSTEM)
    extracted = extract_answer(answer)
    correct = puzzle["answer_check"](answer)
    return {"raw": answer, "extracted": extracted, "correct": correct, "latency": round(latency, 2)}


def run_moe_vote(puzzle, moe_models):
    """Each expert votes independently, majority wins."""
    expert_results = []
    total_latency = 0.0

    for model in moe_models:
        answer, latency = ask_model(model, puzzle["question"], MOE_EXPERT_SYSTEM)
        extracted = extract_answer(answer)
        correct = puzzle["answer_check"](answer)
        expert_results.append({
            "model": model, "raw": answer, "extracted": extracted,
            "correct": correct, "latency": round(latency, 2)
        })
        total_latency += latency

    n_correct = sum(1 for r in expert_results if r["correct"])
    n_active = sum(1 for r in expert_results if not r["raw"].startswith("ERROR:"))
    majority_correct = n_correct > n_active / 2 if n_active > 0 else False

    return {
        "experts": expert_results,
        "correct": majority_correct,
        "n_experts_correct": n_correct,
        "n_active_experts": n_active,
        "latency": round(total_latency, 2)
    }


def run_moe_synth(puzzle, moe_models, synth_model):
    """Each expert answers, then synthesizer reconciles."""
    expert_results = []
    total_latency = 0.0

    for model in moe_models:
        answer, latency = ask_model(model, puzzle["question"], MOE_EXPERT_SYSTEM)
        extracted = extract_answer(answer)
        expert_results.append({
            "model": model, "raw": answer, "extracted": extracted,
            "latency": round(latency, 2)
        })
        total_latency += latency

    # Build synthesis prompt
    synth_prompt = f"PUZZLE: {puzzle['question']}\n\n"
    for i, r in enumerate(expert_results):
        synth_prompt += f"Model {i+1} ({r['model']}) answer:\n{r['raw']}\n\n"

    synth_answer, synth_latency = ask_model(synth_model, synth_prompt, SYNTH_SYSTEM)
    total_latency += synth_latency

    correct = puzzle["answer_check"](synth_answer)
    extracted = extract_answer(synth_answer)

    return {
        "experts": expert_results,
        "synthesizer_raw": synth_answer,
        "extracted": extracted,
        "correct": correct,
        "latency": round(total_latency, 2)
    }


# ─── Main ─────────────────────────────────────────────────────────────────
def run_panel(panel_id, panel_cfg):
    """Run full benchmark for one panel configuration."""
    name = panel_cfg["name"]
    single_model = panel_cfg["single_model"]
    moe_models = panel_cfg["moe_models"]
    synth_model = panel_cfg["synth_model"]

    print(f"\n{'='*65}")
    print(f"  PANEL: {name} ({panel_id})")
    print(f"  Single: {single_model}")
    print(f"  MoE:    {' + '.join(moe_models)}")
    print(f"  Synth:  {synth_model}")
    print(f"{'='*65}")

    all_results = []

    for i, puzzle in enumerate(PUZZLES):
        print(f"\n[{i+1}/{len(PUZZLES)}] {puzzle['id']} ({puzzle['category']}, {puzzle['difficulty']})")
        print(f"  Q: {puzzle['question'][:80]}...")

        puzzle_result = {
            "id": puzzle["id"],
            "category": puzzle["category"],
            "difficulty": puzzle["difficulty"],
            "question": puzzle["question"],
            "expected_answer": puzzle["answer"],
        }

        # Config A: Single
        print(f"  A) Single ({single_model})...", end=" ", flush=True)
        r_single = run_single(puzzle, single_model)
        puzzle_result["single"] = r_single
        print(f"{'✅' if r_single['correct'] else '❌'} [{r_single['extracted'][:40]}] ({r_single['latency']}s)")

        # Config B: Single + CoT
        print(f"  B) Single+CoT...", end=" ", flush=True)
        r_cot = run_single_cot(puzzle, single_model)
        puzzle_result["single_cot"] = r_cot
        print(f"{'✅' if r_cot['correct'] else '❌'} [{r_cot['extracted'][:40]}] ({r_cot['latency']}s)")

        # Config C: MoE Vote
        print(f"  C) MoE-Vote...", end=" ", flush=True)
        r_vote = run_moe_vote(puzzle, moe_models)
        puzzle_result["moe_vote"] = r_vote
        print(f"{'✅' if r_vote['correct'] else '❌'} ({r_vote['latency']}s, {r_vote['n_experts_correct']}/{r_vote['n_active_experts']} correct)")

        # Config D: MoE + Synthesizer
        print(f"  D) MoE-Synth...", end=" ", flush=True)
        r_synth = run_moe_synth(puzzle, moe_models, synth_model)
        puzzle_result["moe_synth"] = r_synth
        print(f"{'✅' if r_synth['correct'] else '❌'} [{r_synth['extracted'][:40]}] ({r_synth['latency']}s)")

        all_results.append(puzzle_result)

    # ─── Panel Summary ──────────────────────────────────────────────────
    configs = ["single", "single_cot", "moe_vote", "moe_synth"]
    config_labels = {
        "single": f"Single ({single_model})",
        "single_cot": "Single + CoT",
        "moe_vote": f"MoE-Vote ({len(moe_models)} experts)",
        "moe_synth": "MoE-Synth",
    }

    print(f"\n{'─'*65}")
    print(f"  {name} — RESULTS")
    print(f"{'─'*65}")
    print(f"  {'Config':<35} {'Correct':>8} {'Accuracy':>10} {'Avg Lat':>10}")
    print(f"  {'-'*65}")

    summary = {}
    for cfg in configs:
        n_correct = sum(1 for r in all_results if r.get(cfg, {}).get("correct", False))
        n_total = len(all_results)
        acc = n_correct / n_total if n_total > 0 else 0
        latencies = [r[cfg]["latency"] for r in all_results if cfg in r and "latency" in r[cfg]]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        print(f"  {config_labels[cfg]:<35} {n_correct:>5}/{n_total:<3} {acc:>10.1%} {avg_lat:>8.1f}s")
        summary[cfg] = {"n_correct": n_correct, "n_total": n_total, "accuracy": round(acc, 4), "avg_latency": round(avg_lat, 2)}

    return {
        "panel_id": panel_id,
        "panel_name": name,
        "models": {"single": single_model, "moe": moe_models, "synth": synth_model},
        "summary": summary,
        "puzzles": all_results,
    }


def main():
    print("=" * 65)
    print("  Paper 3 — Thread B: Multi-Panel Puzzle Benchmark v2")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_panels = {}

    for panel_id, panel_cfg in PANELS.items():
        result = run_panel(panel_id, panel_cfg)
        all_panels[panel_id] = result

    # ─── Cross-Panel Comparison ──────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  CROSS-PANEL SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Panel':<25} {'Single':>8} {'CoT':>8} {'MoE-Vote':>10} {'MoE-Synth':>10}")
    print(f"  {'-'*65}")

    for pid, pdata in all_panels.items():
        s = pdata["summary"]
        sgl = f"{s['single']['n_correct']}/12"
        cot = f"{s['single_cot']['n_correct']}/12"
        vote = f"{s['moe_vote']['n_correct']}/12"
        syn = f"{s['moe_synth']['n_correct']}/12"
        print(f"  {pdata['panel_name']:<25} {sgl:>8} {cot:>8} {vote:>10} {syn:>10}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "version": "v2",
        "panels": all_panels,
    }

    out_path = os.path.join(OUTPUT_DIR, "puzzle_benchmark_v2_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    print(f"\n  💾 Results saved to {out_path}")
    print("\n🎉 MULTI-PANEL PUZZLE BENCHMARK COMPLETE!")


if __name__ == "__main__":
    main()
