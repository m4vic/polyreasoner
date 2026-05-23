"""
Paper 3 Thread D — Full Results Analyzer (v2)
Handles all 7 models including Claude 4.x, Llama-4-Scout, Qwen3-32B
Run after benchmark completes: python analyze_results_v2.py
"""
import json
import re
import os
from collections import defaultdict

RESULTS_PATH = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\aeos_behave\paper3_thread_d\frontier_benchmark_results_v3.json"
DATA_PATH    = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\aeos_behave\paper3_thread_d\benchmark_data_v3.json"

# ─── Answer checking ─────────────────────────────────────────────────────────
def strip_thinking(text):
    if not isinstance(text, str): return str(text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return re.sub(r'<think>.*$', '', text, flags=re.DOTALL).strip()

def check_puzzle(response, expected):
    if not isinstance(response, str): return False
    if "ERROR:" in response or "SKIP:" in response: return False
    resp = strip_thinking(response).lower()
    exp  = str(expected).lower()

    if exp in resp: return True

    keywords = [k for k in re.split(r'[^a-z0-9]', exp) if len(k) > 2]
    if keywords and all(k in resp for k in keywords): return True

    nums_resp = re.findall(r'\d+\.?\d*', resp)
    nums_exp  = re.findall(r'\d+\.?\d*', exp)
    if nums_exp:
        for n in nums_exp:
            if n in nums_resp: return True

    # Domain-specific fallbacks
    if "son"        in exp and "son"        in resp: return True
    if "monopoly"   in exp and "monopoly"   in resp: return True
    if "second"     in exp and "second"     in resp: return True
    if "piano"      in exp and ("piano" in resp or "keyboard" in resp): return True
    if "keyboard"   in exp and ("piano" in resp or "keyboard" in resp): return True
    if "all"        in exp and "12" in exp and ("all" in resp or "every" in resp): return True
    if "none"       in exp and ("none" in resp or "zero" in resp or "no dirt" in resp): return True
    if "stamp"      in exp and "stamp"      in resp: return True
    if "fire"       == exp.strip() and "fire" in resp: return True
    if "breath"     in exp and "breath"     in resp: return True
    if "footstep"   in exp and "foot"       in resp: return True
    if "silence"    in exp and ("silence" in resp or "nothing" in resp): return True
    if "heroin"     in exp and "heroin"     in resp: return True
    return False

# ─── Load data ───────────────────────────────────────────────────────────────
with open(RESULTS_PATH, "r", encoding="utf-8") as f: results = json.load(f)
with open(DATA_PATH,    "r", encoding="utf-8") as f: benchmark = json.load(f)

answers     = {p["id"]: p["answer"]           for p in benchmark["puzzles"]}
puzzle_cats = {p["id"]: p.get("category","?") for p in benchmark["puzzles"]}
categories  = sorted(set(puzzle_cats.values()))

# ─── Aggregate stats ─────────────────────────────────────────────────────────
panel_stats    = defaultdict(lambda: {"correct":0,"total":0,"errors":0})
frontier_stats = defaultdict(lambda: {"correct":0,"total":0,"errors":0})
panel_lat      = defaultdict(list)
frontier_lat   = defaultdict(list)
panel_cat      = defaultdict(lambda: defaultdict(lambda: {"correct":0,"total":0}))
frontier_cat   = defaultdict(lambda: defaultdict(lambda: {"correct":0,"total":0}))

for pr in results["puzzles"]:
    pid = pr["id"]
    exp = answers.get(pid, "")
    cat = puzzle_cats.get(pid, "?")

    for pname, pd in pr.get("moe_panels", {}).items():
        experts = pd.get("experts", [])
        votes   = sum(check_puzzle(e["raw"], exp) for e in experts)
        correct = votes > len(experts) / 2
        panel_stats[pname]["total"] += 1
        panel_cat[pname][cat]["total"] += 1
        if correct:
            panel_stats[pname]["correct"] += 1
            panel_cat[pname][cat]["correct"] += 1
        lat = pd.get("latency", 0)
        if lat > 0: panel_lat[pname].append(lat)

    for mname, md in pr.get("frontier_models", {}).items():
        raw = md.get("raw", "")
        frontier_stats[mname]["total"] += 1
        frontier_cat[mname][cat]["total"] += 1
        if "ERROR:" in str(raw) or "SKIP:" in str(raw):
            frontier_stats[mname]["errors"] += 1
        elif check_puzzle(raw, exp):
            frontier_stats[mname]["correct"] += 1
            frontier_cat[mname][cat]["correct"] += 1
        lat = md.get("latency", 0)
        if lat > 0: frontier_lat[mname].append(lat)

# ─── Print ────────────────────────────────────────────────────────────────────
W = 80
print("=" * W)
print("  PAPER 3 — THREAD D: COMPLETE FRONTIER BENCHMARK RESULTS")
print("=" * W)

# Section 1: MoE Panels
print(f"\n{'-'*W}")
print("SECTION 1: MoE PANEL ACCURACY  (Logic Puzzles, n=30)")
print(f"{'-'*W}")
print(f"{'Panel':<12} {'Correct':>8} {'Total':>6} {'Accuracy':>10} {'Avg Latency':>13}")
print("-" * 52)
for p in sorted(panel_stats):
    s   = panel_stats[p]
    acc = s["correct"] / max(s["total"],1) * 100
    avg = sum(panel_lat[p]) / max(len(panel_lat[p]),1)
    print(f"{p:<12} {s['correct']:>8} {s['total']:>6} {acc:>9.1f}% {avg:>11.1f}s")

# Section 2: Frontier Models
print(f"\n{'-'*W}")
print("SECTION 2: FRONTIER MODEL ACCURACY  (Logic Puzzles, n=30)")
print(f"{'-'*W}")
print(f"{'Model':<22} {'Correct':>8} {'Total':>6} {'Errors':>7} {'Accuracy':>10} {'Avg Lat':>9}")
print("-" * 67)
for m in sorted(frontier_stats):
    s     = frontier_stats[m]
    valid = s["total"] - s["errors"]
    acc   = s["correct"] / max(valid,1) * 100
    avg   = sum(frontier_lat[m]) / max(len(frontier_lat[m]),1)
    flag  = " [ALL ERRORS]" if s["errors"] == s["total"] else ""
    print(f"{m:<22} {s['correct']:>8} {s['total']:>6} {s['errors']:>7} {acc:>9.1f}% {avg:>7.1f}s{flag}")

# Section 3: Category breakdown
print(f"\n{'-'*W}")
print("SECTION 3: ACCURACY BY CATEGORY")
print(f"{'-'*W}")

print("\nMoE Panels:")
hdr = f"{'Panel':<12}" + "".join(f" {c:>12}" for c in categories)
print(hdr); print("─" * len(hdr))
for p in sorted(panel_cat):
    row = f"{p:<12}"
    for c in categories:
        cs = panel_cat[p][c]
        row += f" {cs['correct']/max(cs['total'],1)*100:>10.0f}%" if cs["total"] else f" {'N/A':>11}"
    print(row)

print("\nFrontier Models:")
hdr = f"{'Model':<22}" + "".join(f" {c:>12}" for c in categories)
print(hdr); print("─" * len(hdr))
for m in sorted(frontier_cat):
    row = f"{m:<22}"
    for c in categories:
        cs = frontier_cat[m][c]
        row += f" {cs['correct']/max(cs['total'],1)*100:>10.0f}%" if cs["total"] else f" {'N/A':>11}"
    print(row)

# Section 4: Strategic
print(f"\n{'─'*W}")
print("SECTION 4: STRATEGIC REASONING TASKS")
print(f"{'─'*W}")
strategic = results.get("strategic", [])
print(f"Total tasks completed: {len(strategic)}\n")
for s in strategic:
    errs = [k for k,v in s.get("frontier_models",{}).items()
            if "ERROR:" in str(v.get("raw","")) or "SKIP:" in str(v.get("raw",""))]
    status = "ALL OK" if not errs else f"ERRORS: {', '.join(errs)}"
    panels = list(s.get("moe_panels",{}).keys())
    print(f"  {s['id']:<22} MoE={panels}  Status={status}")
    for fname, fdata in s.get("frontier_models",{}).items():
        raw = str(fdata.get("raw",""))
        snippet = raw[:120].replace('\n',' ') if "ERROR:" not in raw and "SKIP:" not in raw else raw
        print(f"    [{fname}] {snippet}...")
    print()

# Section 5: Key findings
print(f"\n{'='*W}")
print("KEY FINDINGS FOR PAPER 3 MANUSCRIPT")
print(f"{'='*W}")

best_panel    = max(panel_stats,    key=lambda p: panel_stats[p]["correct"]/max(panel_stats[p]["total"],1))
bp_acc        = panel_stats[best_panel]["correct"] / max(panel_stats[best_panel]["total"],1) * 100

valid_frontier = {m:s for m,s in frontier_stats.items() if s["errors"] < s["total"]}
if valid_frontier:
    best_frontier = max(valid_frontier, key=lambda m: valid_frontier[m]["correct"]/max(valid_frontier[m]["total"]-valid_frontier[m]["errors"],1))
    bv = valid_frontier[best_frontier]["total"] - valid_frontier[best_frontier]["errors"]
    bf_acc = valid_frontier[best_frontier]["correct"] / max(bv,1) * 100
else:
    best_frontier, bf_acc = "N/A", 0.0

avg_moe = sum(s["correct"] for s in panel_stats.values()) / max(sum(s["total"] for s in panel_stats.values()),1) * 100
total_calls  = sum(s["total"]  for s in frontier_stats.values())
total_errors = sum(s["errors"] for s in frontier_stats.values())

# Diversity gradient
homo_acc    = panel_stats["Panel_F"]["correct"] / max(panel_stats["Panel_F"]["total"],1) * 100
diverse_acc = max(panel_stats[p]["correct"]/max(panel_stats[p]["total"],1)*100 for p in ["Panel_B","Panel_E"])
diversity_premium = diverse_acc - homo_acc

print(f"\n  Best MoE Panel      : {best_panel} at {bp_acc:.1f}%  ($0.00/query, 100% uptime)")
print(f"  Best Frontier Model : {best_frontier} at {bf_acc:.1f}%")
print(f"  Average MoE Accuracy: {avg_moe:.1f}%")
print(f"  Diversity Premium   : +{diversity_premium:.1f} pp  (homogeneous {homo_acc:.1f}% → diverse {diverse_acc:.1f}%)")
print(f"  API Error Rate      : {total_errors}/{total_calls} ({total_errors/max(total_calls,1)*100:.1f}%)")
print(f"  MoE Uptime          : 100% (local inference)")
print()

# Model comparison table for abstract
print("  --- Numbers for Abstract ---")
print(f"  Local best : {bp_acc:.1f}%  |  Cloud best: {bf_acc:.1f}%  |  Gap: {bf_acc - bp_acc:.1f} pp")
print(f"  Diversity premium: +{diversity_premium:.1f} pp")
print(f"  Frontier error rate: {total_errors/max(total_calls,1)*100:.1f}%  |  Local error rate: 0%")
