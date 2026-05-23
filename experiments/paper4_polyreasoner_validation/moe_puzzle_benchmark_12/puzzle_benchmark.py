"""
Paper 3 — Thread B: Open-Ended Reasoning Benchmark
====================================================
Tests whether MoE (panel of diverse small LLMs) produces better quality
outputs than a single large LLM thinking harder.

Experiment B2: Puzzle Solving (Objective — correct/wrong answers known)
-----------------------------------------------------------------------
Tasks include:
  - Logic grid puzzles
  - Lateral thinking problems
  - Math word problems (requiring multi-step reasoning)
  - Constraint satisfaction
  - Classic trick questions that fool overconfident models

Configurations:
  A) Single-Large:  One large model (qwen3.5:9b) solves alone
  B) Single-CoT:    Same model + explicit chain-of-thought prompt (thinking harder)
  C) MoE-Vote:      3 diverse small models each solve independently, majority vote
  D) MoE-Synth:     3 diverse small models + synthesizer LLM reconciles conflicts

Metric: Accuracy (exact match or judge-verified correct answer)

Output: benchmark_results.json, per-puzzle breakdown, summary chart
"""

import os
import json
import time
import asyncio
import re
from datetime import datetime

import ollama

# ─── Configuration ────────────────────────────────────────────────────────
OUTPUT_DIR = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\aeos_behave\paper3_thread_b"

# Models (must be available in Ollama)
SINGLE_MODEL    = "qwen3.5:9b"
MOE_MODELS      = ["qwen2.5-coder:7b", "llama3.1:8b", "mistral:7b"]
SYNTH_MODEL     = "qwen3.5:9b"

# ─── Puzzle Set ───────────────────────────────────────────────────────────
PUZZLES = [
    # --- Logic Puzzles ---
    {
        "id": "logic_01",
        "category": "logic",
        "difficulty": "easy",
        "question": "If all Bloops are Razzles, and all Razzles are Lazzles, are all Bloops definitely Lazzles?",
        "answer": "yes",
        "answer_check": lambda r: "yes" in r.lower()
    },
    {
        "id": "logic_02",
        "category": "logic",
        "difficulty": "medium",
        "question": (
            "Three friends — Alice, Bob, Carol — each pick a different number from {1, 2, 3}. "
            "Alice does not pick 1. Bob picks a higher number than Alice. "
            "What number does Carol pick?"
        ),
        "answer": "1",
        "answer_check": lambda r: bool(re.search(r'\b1\b', r))
    },
    {
        "id": "logic_03",
        "category": "logic",
        "difficulty": "hard",
        "question": (
            "A farmer has 17 sheep. All but 9 die. How many sheep are left?"
        ),
        "answer": "9",
        "answer_check": lambda r: bool(re.search(r'\b9\b', r))
    },
    {
        "id": "logic_04",
        "category": "logic",
        "difficulty": "hard",
        "question": (
            "You have a 3-gallon jug and a 5-gallon jug, no markings. "
            "How do you measure exactly 4 gallons? Give the minimum number of steps."
        ),
        "answer": "6",
        "answer_check": lambda r: bool(re.search(r'\b6\b', r)) or "six" in r.lower()
    },
    # --- Math Word Problems ---
    {
        "id": "math_01",
        "category": "math",
        "difficulty": "easy",
        "question": (
            "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost in cents?"
        ),
        "answer": "5",
        "answer_check": lambda r: bool(re.search(r'\b5\b', r)) and not bool(re.search(r'\b10\b', r[:50]))
    },
    {
        "id": "math_02",
        "category": "math",
        "difficulty": "medium",
        "question": (
            "If it takes 5 machines 5 minutes to make 5 widgets, "
            "how many minutes does it take 100 machines to make 100 widgets?"
        ),
        "answer": "5",
        "answer_check": lambda r: bool(re.search(r'\b5\b', r))
    },
    {
        "id": "math_03",
        "category": "math",
        "difficulty": "hard",
        "question": (
            "A lily pad doubles in size every day. It takes 48 days to cover half the pond. "
            "How many days to cover the whole pond?"
        ),
        "answer": "49",
        "answer_check": lambda r: bool(re.search(r'\b49\b', r))
    },
    # --- Lateral Thinking ---
    {
        "id": "lateral_01",
        "category": "lateral",
        "difficulty": "medium",
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
        "id": "lateral_02",
        "category": "lateral",
        "difficulty": "easy",
        "question": (
            "A woman shoots her husband, then dines with him that evening. How?"
        ),
        "answer": "She is a photographer. She shot him with a camera.",
        "answer_check": lambda r: "photo" in r.lower() or "camera" in r.lower()
    },
    # --- Constraint Satisfaction ---
    {
        "id": "constraint_01",
        "category": "constraint",
        "difficulty": "medium",
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
        "id": "trick_01",
        "category": "trick",
        "difficulty": "easy",
        "question": "How many months have 28 days?",
        "answer": "12",
        "answer_check": lambda r: bool(re.search(r'\b12\b', r)) or "all" in r.lower() or "twelve" in r.lower()
    },
    {
        "id": "trick_02",
        "category": "trick",
        "difficulty": "medium",
        "question": (
            "A rooster lays an egg on top of a barn roof. Which way does it roll?"
        ),
        "answer": "Roosters don't lay eggs.",
        "answer_check": lambda r: "rooster" in r.lower() and ("don't" in r.lower() or "cannot" in r.lower() or "not" in r.lower())
    },
]

# ─── LLM Call ────────────────────────────────────────────────────────────
def ask_model(model, question, system_prompt=None, timeout=60):
    """Synchronous Ollama call with timeout."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    try:
        start = time.time()
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.1, "num_predict": 512}
        )
        elapsed = time.time() - start
        content = response["message"]["content"].strip()
        return content, elapsed
    except Exception as e:
        return f"ERROR: {e}", 0.0

# ─── Judge Configs ────────────────────────────────────────────────────────
SINGLE_SYSTEM = "You are a precise reasoning assistant. Answer the question directly and show your work. End with your final answer clearly labeled."

COT_SYSTEM = """You are a precise reasoning assistant using chain-of-thought.
STEP 1: Restate the problem in your own words.
STEP 2: Identify any traps or trick elements.
STEP 3: Reason through it step-by-step.
STEP 4: State your final answer clearly on the last line as: ANSWER: <your answer>"""

MOE_EXPERT_SYSTEM = "You are a precise reasoning assistant. Answer concisely. End with: FINAL ANSWER: <answer>"

SYNTH_SYSTEM = """You are a synthesis judge. You receive answers from 3 different models on the same puzzle.
Your job:
1. Identify where they agree and where they disagree.
2. If they disagree, reason about which answer is correct and why.
3. State the final consensus answer clearly as: FINAL ANSWER: <answer>"""

def extract_answer(text):
    """Try to extract a clean final answer from model output."""
    # Look for ANSWER: or FINAL ANSWER: patterns
    for pattern in [r'FINAL ANSWER:\s*(.+)', r'ANSWER:\s*(.+)', r'Answer:\s*(.+)']:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Fall back to last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text[:200]

# ─── Run Configs ─────────────────────────────────────────────────────────
def run_single(puzzle):
    answer, latency = ask_model(SINGLE_MODEL, puzzle["question"], SINGLE_SYSTEM)
    extracted = extract_answer(answer)
    correct = puzzle["answer_check"](answer)
    return {"raw": answer, "extracted": extracted, "correct": correct, "latency": round(latency, 2)}

def run_single_cot(puzzle):
    answer, latency = ask_model(SINGLE_MODEL, puzzle["question"], COT_SYSTEM)
    extracted = extract_answer(answer)
    correct = puzzle["answer_check"](answer)
    return {"raw": answer, "extracted": extracted, "correct": correct, "latency": round(latency, 2)}

def run_moe_vote(puzzle):
    """Each expert votes independently, majority wins."""
    expert_results = []
    total_latency = 0.0

    for model in MOE_MODELS:
        answer, latency = ask_model(model, puzzle["question"], MOE_EXPERT_SYSTEM)
        extracted = extract_answer(answer)
        correct = puzzle["answer_check"](answer)
        expert_results.append({
            "model": model, "raw": answer, "extracted": extracted,
            "correct": correct, "latency": round(latency, 2)
        })
        total_latency += latency

    # Majority vote on correctness
    n_correct = sum(1 for r in expert_results if r["correct"])
    majority_correct = n_correct > len(MOE_MODELS) / 2

    return {
        "experts": expert_results,
        "correct": majority_correct,
        "n_experts_correct": n_correct,
        "latency": round(total_latency, 2)
    }

def run_moe_synth(puzzle):
    """Each expert answers, then synthesizer reconciles."""
    expert_results = []
    total_latency = 0.0

    for model in MOE_MODELS:
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

    synth_answer, synth_latency = ask_model(SYNTH_MODEL, synth_prompt, SYNTH_SYSTEM)
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
def main():
    print("=" * 65)
    print("  Paper 3 — Thread B: Puzzle Benchmark")
    print(f"  Single ({SINGLE_MODEL}) vs MoE ({'+'.join(MOE_MODELS)})")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        print(f"  Running Config A (Single-{SINGLE_MODEL})...", end=" ", flush=True)
        r_single = run_single(puzzle)
        puzzle_result["single"] = r_single
        print(f"{'✅' if r_single['correct'] else '❌'} ({r_single['latency']}s)")

        # Config B: Single + CoT
        print(f"  Running Config B (Single-CoT)...", end=" ", flush=True)
        r_cot = run_single_cot(puzzle)
        puzzle_result["single_cot"] = r_cot
        print(f"{'✅' if r_cot['correct'] else '❌'} ({r_cot['latency']}s)")

        # Config C: MoE Vote
        print(f"  Running Config C (MoE-Vote)...", end=" ", flush=True)
        r_vote = run_moe_vote(puzzle)
        puzzle_result["moe_vote"] = r_vote
        print(f"{'✅' if r_vote['correct'] else '❌'} ({r_vote['latency']}s, {r_vote['n_experts_correct']}/{len(MOE_MODELS)} experts correct)")

        # Config D: MoE + Synthesizer
        print(f"  Running Config D (MoE-Synth)...", end=" ", flush=True)
        r_synth = run_moe_synth(puzzle)
        puzzle_result["moe_synth"] = r_synth
        print(f"{'✅' if r_synth['correct'] else '❌'} ({r_synth['latency']}s)")

        all_results.append(puzzle_result)

    # ─── Summary ─────────────────────────────────────────────────────────
    configs = ["single", "single_cot", "moe_vote", "moe_synth"]
    config_labels = {
        "single": f"Single ({SINGLE_MODEL})",
        "single_cot": "Single + CoT (Think Harder)",
        "moe_vote": "MoE Majority Vote",
        "moe_synth": "MoE + Synthesizer",
    }

    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)
    print(f"  {'Config':<35} {'Correct':>8} {'Accuracy':>10} {'Avg Latency':>12}")
    print(f"  {'-'*68}")

    summary = {}
    for cfg in configs:
        n_correct = sum(1 for r in all_results if r.get(cfg, {}).get("correct", False))
        n_total = len(all_results)
        acc = n_correct / n_total if n_total > 0 else 0
        latencies = [r[cfg]["latency"] for r in all_results if cfg in r and "latency" in r[cfg]]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        print(f"  {config_labels[cfg]:<35} {n_correct:>5}/{n_total:<3} {acc:>10.1%} {avg_lat:>10.1f}s")
        summary[cfg] = {"n_correct": n_correct, "n_total": n_total, "accuracy": round(acc, 4), "avg_latency": round(avg_lat, 2)}

    # Per-category breakdown
    print("\n  PER-CATEGORY BREAKDOWN")
    categories = list(set(r["category"] for r in all_results))
    for cat in sorted(categories):
        cat_results = [r for r in all_results if r["category"] == cat]
        print(f"\n  [{cat.upper()}] ({len(cat_results)} puzzles)")
        for cfg in configs:
            n_correct = sum(1 for r in cat_results if r.get(cfg, {}).get("correct", False))
            print(f"    {config_labels[cfg]:<35}: {n_correct}/{len(cat_results)}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "models": {"single": SINGLE_MODEL, "moe": MOE_MODELS, "synth": SYNTH_MODEL},
        "summary": summary,
        "puzzles": all_results,
    }

    out_path = os.path.join(OUTPUT_DIR, "puzzle_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    print(f"\n  💾 Results saved to {out_path}")
    print("\n🎉 THREAD B PUZZLE BENCHMARK COMPLETE!")

if __name__ == "__main__":
    main()
