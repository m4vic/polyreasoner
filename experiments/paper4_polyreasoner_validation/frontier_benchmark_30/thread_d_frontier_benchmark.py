"""
Paper 3 — Thread D: Frontier Baseline Benchmark (v3 + Resume)
=================================================
Features:
- Loads benchmark_data_v3.json
- Supports 8 MoE Panel configurations
- Supports Groq, OpenAI, Anthropic, Local Ollama
- **RESUME FEATURE**: Automatically picks up where it left off if interrupted.
"""

import os
import json
import time
import re
import sys
import argparse
from datetime import datetime
from openai import OpenAI
import anthropic
import ollama

# ─── Output ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\aeos_behave\paper3_thread_d"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_PATH = os.path.join(OUTPUT_DIR, "benchmark_data_v3.json")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "frontier_benchmark_results_v3.json")

# ─── Provider Configs ──────────────────────────────────────────────────────
PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    "openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
    "anthropic": {"api_key_env": "ANTHROPIC_API_KEY"}
}

# ─── MoE Panel Roster ─────────────────────────────────────────────────────
MOE_PANELS = {
    "Panel_A": ["qwen2.5-coder:7b", "llama3.1:8b", "deepseek-coder:6.7b"],
    "Panel_B": ["deepseek-r1:8b", "qwen3.5:9b", "llama3.1:8b"],
    "Panel_C": ["qwen2.5-coder:3b", "phi3:mini", "qwen2.5-coder:1.5b"],
    "Panel_D": ["qwen2.5-coder:14b", "deepseek-coder-v2:16b", "gemma4:latest"],
    "Panel_E": ["llama3.1:8b", "gemma4:latest", "ministral-3:14b", "deepseek-r1:8b", "phi3:mini"],
    "Panel_F": ["qwen2.5-coder:7b", "qwen2.5-coder:7b", "qwen2.5-coder:7b"],
    "Panel_G": ["qwen2.5-coder:7b", "qwen2.5-coder:14b", "qwen3.5:9b"],
    "Panel_H": ["qwen2.5-coder:1.5b", "llama3.1:8b", "deepseek-coder-v2:16b"],
}

FRONTIER_MODELS = [
    # Groq — production open-weight models (verified May 2026)
    ("groq",      "llama-3.3-70b-versatile",   "Llama-3.3-70B",    70,  "open_large"),
    ("groq",      "meta-llama/llama-4-scout-17b-16e-instruct", "Llama-4-Scout", 17, "open_large"),
    ("groq",      "qwen/qwen3-32b",             "Qwen3-32B",        32,  "open_large"),
    # OpenAI
    ("openai",    "gpt-4o-mini",               "GPT-4o-mini",      "?", "frontier_cheap"),
    ("openai",    "gpt-4o",                    "GPT-4o",           "?", "frontier"),
    # Anthropic — Claude 4.x generation (verified May 2026)
    ("anthropic", "claude-haiku-4-5-20251001", "Claude-Haiku-4.5", "?", "frontier_cheap"),
    ("anthropic", "claude-sonnet-4-6",         "Claude-Sonnet-4.6","?", "frontier"),
]

PUZZLE_SYSTEM = "You are a precise reasoning assistant. Answer the puzzle directly. End with: FINAL ANSWER: <short answer>"
STRATEGIC_SYSTEM = "You are a high-level strategic advisor. Provide a detailed, structured analysis with actionable insights."

# ─── Helpers ─────────────────────────────────────────────────────────────
def strip_thinking(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return re.sub(r'<think>.*$', '', text, flags=re.DOTALL).strip()

STOPWORDS = {
    "a", "an", "the",
    "his", "her", "their", "our", "my", "your", "its",
    "this", "that", "these", "those",
}

def check_puzzle(response, expected):
    if not isinstance(response, str):
        return False

    resp_raw = strip_thinking(response).strip()
    resp = resp_raw.lower()
    exp = str(expected).strip().lower()

    # Explicitly mark provider failures as incorrect.
    if resp.startswith("error:") or resp.startswith("skip:"):
        return False

    # 1) Direct match (e.g., "5 cents")
    if exp and exp in resp:
        return True

    # 2) Keyword match (ignores stopwords like "his" in "His son")
    keywords = [
        k for k in re.split(r"[^a-z0-9]+", exp)
        if len(k) > 2 and k not in STOPWORDS
    ]
    if keywords and all(k in resp for k in keywords):
        return True

    # 3) Numeric match (handles 0.05, 5, etc.)
    nums_in_resp = re.findall(r"\d+\.?\d*", resp)
    nums_in_exp = re.findall(r"\d+\.?\d*", exp)
    if nums_in_exp:
        for n in nums_in_exp:
            if n in nums_in_resp:
                return True
        # Special case for "0.05" vs "5" (cents)
        if "0.05" in nums_in_exp and ("0.05" in nums_in_resp or "5" in nums_in_resp):
            return True

    # 4) Domain-specific fallbacks (kept conservative and explicit)
    if "son" in exp and "son" in resp:
        return True
    if "monopoly" in exp and "monopoly" in resp:
        return True
    if "second" in exp and "second" in resp:
        return True
    if "piano" in exp and ("piano" in resp or "keyboard" in resp):
        return True
    if "keyboard" in exp and ("piano" in resp or "keyboard" in resp):
        return True
    if "all" in exp and "12" in exp and ("all" in resp or "every" in resp):
        return True
    if "none" in exp and ("none" in resp or "zero" in resp or "no dirt" in resp):
        return True
    if "stamp" in exp and "stamp" in resp:
        return True
    if exp.strip() == "fire" and "fire" in resp:
        return True
    if "breath" in exp and "breath" in resp:
        return True
    if "footstep" in exp and "foot" in resp:
        return True
    if "silence" in exp and ("silence" in resp or "nothing" in resp):
        return True
    if "heroin" in exp and "heroin" in resp:
        return True

    return False


# ─── API Clients ─────────────────────────────────────────────────────────
def ask_cloud(provider_key, model_id, question, system_prompt):
    cfg = PROVIDERS[provider_key]
    key = os.environ.get(cfg["api_key_env"])
    if not key: return f"SKIP: Missing {cfg['api_key_env']}", 0.0
    try:
        start = time.time()
        if provider_key == "anthropic":
            client = anthropic.Anthropic(api_key=key)
            resp = client.messages.create(model=model_id, max_tokens=2048, system=system_prompt, messages=[{"role": "user", "content": question}])
            content = resp.content[0].text
        else:
            client = OpenAI(api_key=key, base_url=cfg["base_url"])
            resp = client.chat.completions.create(model=model_id, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}], temperature=0.1, max_tokens=2048)
            content = resp.choices[0].message.content
        return strip_thinking(content), time.time() - start
    except Exception as e: return f"ERROR: {str(e)}", 0.0

def ask_local(model, question, system_prompt):
    try:
        start = time.time()
        resp = ollama.chat(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}], options={"temperature": 0.1, "num_predict": 2048})
        return strip_thinking(resp["message"]["content"]), time.time() - start
    except Exception as e: return f"ERROR: {str(e)}", 0.0

def run_moe_panel(panel_name, models, task, is_strategic=False):
    expert_results = []; total_lat = 0.0; sys_p = STRATEGIC_SYSTEM if is_strategic else PUZZLE_SYSTEM
    for m in models:
        raw, lat = ask_local(m, task["question"], sys_p)
        expert_results.append({"model": m, "raw": raw, "latency": lat})
        total_lat += lat
    correct = False if is_strategic else (sum([check_puzzle(r["raw"], task["answer"]) for r in expert_results]) > len(expert_results)/2)
    return {"panel": panel_name, "correct": correct if not is_strategic else None, "latency": total_lat, "experts": expert_results}

# ─── Resume/Checkpoint Logic ─────────────────────────────────────────────
def load_results():
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r") as f: return json.load(f)
        except: return {"puzzles": [], "strategic": []}
    return {"puzzles": [], "strategic": []}

def save_results(results):
    with open(RESULTS_PATH, "w") as f: json.dump(results, f, indent=2)

# ─── Main ────────────────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(description="Paper 3 Thread D frontier benchmark (resume + force rerun flags).")
    parser.add_argument("--force-moe", action="store_true", help="Rerun local MoE panels even if cached.")
    parser.add_argument("--force-frontier", action="store_true", help="Rerun frontier (API) models even if cached.")
    parser.add_argument("--skip-moe", action="store_true", help="Skip local MoE panels entirely.")
    parser.add_argument("--skip-frontier", action="store_true", help="Skip frontier (API) models entirely.")
    parser.add_argument("--only-puzzles", action="store_true", help="Run only puzzle tasks (skip strategic).")
    parser.add_argument("--only-strategic", action="store_true", help="Run only strategic tasks (skip puzzles).")
    parser.add_argument("--panels", nargs="*", default=None, help="Optional subset of panels (e.g., Panel_A Panel_E).")
    parser.add_argument("--frontier", nargs="*", default=None, help="Optional subset of frontier display names (e.g., GPT-4o).")
    return parser.parse_args()


def main(args=None):
    args = args or _parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    if not os.path.exists(DATA_PATH):
        return print(f"Error: {DATA_PATH} not found.")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    results = load_results()
    completed_puzzles = {r["id"] for r in results.get("puzzles", [])}
    completed_strategic = {r["id"] for r in results.get("strategic", [])}

    print("\n[INFO] Frontier Baseline v3 (resume enabled)")
    print(f"  Puzzles: {len(completed_puzzles)}/{len(data['puzzles'])} done")
    print(f"  Strategic: {len(completed_strategic)}/{len(data['strategic'])} done\n")

    panel_filter = set(args.panels) if args.panels else None
    frontier_filter = set(args.frontier) if args.frontier else None

    run_puzzles = not args.only_strategic
    run_strategic = not args.only_puzzles

    # 1. Puzzles
    if run_puzzles:
        for i, p in enumerate(data["puzzles"]):
            print(f"[{i+1}/30] Puzzle: {p['id']}...", end=" ", flush=True)

            res = next((r for r in results["puzzles"] if r["id"] == p["id"]), None)
            if not res:
                res = {"id": p["id"], "moe_panels": {}, "frontier_models": {}}
                results["puzzles"].append(res)

            if not args.skip_moe:
                for name, models in MOE_PANELS.items():
                    if panel_filter and name not in panel_filter:
                        continue
                    if args.force_moe or name not in res["moe_panels"]:
                        res["moe_panels"][name] = run_moe_panel(name, models, p)
                    else:
                        expert_res = res["moe_panels"][name]["experts"]
                        correct_count = sum([check_puzzle(r["raw"], p["answer"]) for r in expert_res])
                        res["moe_panels"][name]["correct"] = (correct_count > len(expert_res)/2)

            if not args.skip_frontier:
                for prov, mid, disp, _, _ in FRONTIER_MODELS:
                    if frontier_filter and disp not in frontier_filter:
                        continue
                    needs_run = args.force_frontier or (disp not in res["frontier_models"])
                    if not needs_run:
                        raw_text = res["frontier_models"][disp].get("raw", "")
                        if "ERROR:" in raw_text or "SKIP:" in raw_text:
                            needs_run = True

                    if needs_run:
                        raw, lat = ask_cloud(prov, mid, p["question"], PUZZLE_SYSTEM)
                        res["frontier_models"][disp] = {"raw": raw, "latency": lat, "correct": check_puzzle(raw, p["answer"])}
                    else:
                        res["frontier_models"][disp]["correct"] = check_puzzle(res["frontier_models"][disp]["raw"], p["answer"])

            save_results(results)
            print("Done (Updated/Retried)")

    # 2. Strategic
    if run_strategic:
        for i, s in enumerate(data["strategic"]):
            print(f"[{i+1}/5] Strategic: {s['id']}...", end=" ", flush=True)

            res = next((r for r in results["strategic"] if r["id"] == s["id"]), None)
            if not res:
                res = {"id": s["id"], "moe_panels": {}, "frontier_models": {}}
                results["strategic"].append(res)

            if not args.skip_moe:
                for name in ["Panel_A", "Panel_E"]:
                    if panel_filter and name not in panel_filter:
                        continue
                    if args.force_moe or name not in res["moe_panels"]:
                        res["moe_panels"][name] = run_moe_panel(name, MOE_PANELS[name], s, is_strategic=True)

            if not args.skip_frontier:
                for disp in ["GPT-4o", "Claude-Sonnet-4.6", "Llama-3.3-70B"]:
                    if frontier_filter and disp not in frontier_filter:
                        continue
                    needs_run = args.force_frontier or (disp not in res["frontier_models"])
                    if not needs_run:
                        raw_text = res["frontier_models"][disp].get("raw", "")
                        needs_run = "ERROR:" in raw_text or "SKIP:" in raw_text
                    if needs_run:
                        try:
                            m_data = next(m for m in FRONTIER_MODELS if m[2] == disp)
                            raw, lat = ask_cloud(m_data[0], m_data[1], s["question"], STRATEGIC_SYSTEM)
                            res["frontier_models"][disp] = {"raw": raw, "latency": lat}
                        except StopIteration:
                            print(f"Warning: {disp} not in FRONTIER_MODELS")

            save_results(results)
            print("Done (Updated/Retried)")

    print(f"\n[DONE] Benchmark complete. Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main(_parse_args())
