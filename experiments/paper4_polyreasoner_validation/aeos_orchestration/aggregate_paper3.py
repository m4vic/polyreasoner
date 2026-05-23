"""
Paper 3 — Thread A: Cross-Modality Aggregation
================================================
Reads all Exp1 (single-agent) and Exp2 (dual-agent) result JSONs across
Tabular, Vision, and Text modalities. Produces:
  - Summary tables (single vs dual per modality)
  - Cross-modality comparison figure
  - paper3_thread_a_results.json for the paper
"""

import os
import json
import glob
import numpy as np
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ─── Configuration ───────────────────────────────────────────────────────
RESULTS_BASE = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\aeos_behave\results"
OUTPUT_DIR   = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\aeos_behave\paper3_thread_a"

MODALITIES = {
    "tabular": os.path.join(RESULTS_BASE, "tabular2"),
    "vision":  os.path.join(RESULTS_BASE, "vision"),
    "text":    os.path.join(RESULTS_BASE, "text"),
}

# ─── JSON Parsing ────────────────────────────────────────────────────────
def clean_model_name(name):
    """Remove ollama/ prefix and normalize."""
    if not name:
        return "unknown"
    return name.replace("ollama/", "").strip()

def parse_result_file(path):
    """Parse a single AEOS result JSON. Returns dict with key fields."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ⚠️ Could not parse {os.path.basename(path)}: {e}")
        return None

    # Only handle dict format (not list)
    if not isinstance(data, dict):
        return None

    fname = os.path.basename(path)

    # Determine exp type from filename or JSON
    exp_field = data.get("exp", "")
    if fname.startswith("exp2_") or "critic" in exp_field.lower() or "agent" in exp_field.lower():
        exp_type = "dual"
    elif fname.startswith("exp1_"):
        exp_type = "single"
    elif fname.startswith("exp3_"):
        exp_type = "tri"  # tri-agent, skip for now or treat as dual
        return None  # Skip tri-agent for this aggregation
    else:
        return None  # Unknown format

    result = {
        "file": fname,
        "run_id": data.get("run_id", ""),
        "exp_type": exp_type,
        "accuracy": data.get("best_accuracy"),
        "iterations": data.get("total_iterations"),
        "sunk_cost": data.get("sunk_cost_episodes"),
        "waste_count": data.get("waste_count"),
        "stop_reason": data.get("stop_reason"),
        "time_seconds": data.get("total_time_seconds"),
        "model": None,
        "reviewer": None,
        "coder": None,
    }

    if exp_type == "single":
        result["model"] = clean_model_name(data.get("model"))
    else:
        result["reviewer"] = clean_model_name(data.get("reviewer_model"))
        result["coder"] = clean_model_name(data.get("coder_model"))

    return result

def load_modality(modality_name, results_dir):
    """Load all result JSONs for a modality, deduplicating by run_id."""
    if not os.path.exists(results_dir):
        print(f"  ⚠️ Directory not found: {results_dir}")
        return []

    pattern = os.path.join(results_dir, "exp*.json")
    files = sorted(glob.glob(pattern))
    print(f"  Found {len(files)} JSON files for {modality_name}")

    # Parse all
    raw_results = []
    for fpath in files:
        r = parse_result_file(fpath)
        if r and r["accuracy"] is not None:
            r["modality"] = modality_name
            raw_results.append(r)

    # Deduplicate: for each (exp_type, model_key, run_id), keep the latest file
    # This handles tabular2 having duplicate run files from different dates
    seen = {}
    for r in raw_results:
        if r["exp_type"] == "single":
            key = (r["exp_type"], r["model"], r["run_id"])
        else:
            key = (r["exp_type"], r["reviewer"], r["coder"], r["run_id"])
        # Last file in sorted order wins (latest timestamp in filename)
        seen[key] = r

    results = list(seen.values())
    skipped = len(raw_results) - len(results)
    if skipped > 0:
        print(f"  Deduplicated: {skipped} duplicate run_ids removed")
    print(f"  Final: {len(results)} unique results ({sum(1 for r in results if r['exp_type']=='single')} single, {sum(1 for r in results if r['exp_type']=='dual')} dual)")

    return results

# ─── Aggregation ─────────────────────────────────────────────────────────
def aggregate_by_model(results, exp_type):
    """Group results by model/pairing and compute avg/max accuracy."""
    grouped = defaultdict(list)

    for r in results:
        if r["exp_type"] != exp_type:
            continue
        if exp_type == "single":
            key = r["model"] or "unknown"
        else:
            reviewer = r["reviewer"] or "?"
            coder = r["coder"] or "?"
            key = f"{reviewer} → {coder}"
        grouped[key].append(r)

    summary = []
    for key, runs in grouped.items():
        accs = [r["accuracy"] for r in runs if r["accuracy"] is not None]
        iters = [r["iterations"] for r in runs if r["iterations"] is not None]
        costs = [r["sunk_cost"] for r in runs if r["sunk_cost"] is not None]
        times = [r["time_seconds"] for r in runs if r["time_seconds"] is not None]
        stop_reasons = [r["stop_reason"] for r in runs if r["stop_reason"]]

        if not accs:
            continue

        summary.append({
            "key": key,
            "exp_type": exp_type,
            "n_runs": len(runs),
            "avg_accuracy": round(float(np.mean(accs)), 4),
            "max_accuracy": round(float(np.max(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4) if len(accs) > 1 else 0.0,
            "avg_iterations": round(float(np.mean(iters)), 1) if iters else None,
            "avg_sunk_cost": round(float(np.mean(costs)), 1) if costs else None,
            "avg_time_s": round(float(np.mean(times)), 0) if times else None,
            "stop_reasons": list(set(stop_reasons)),
        })

    return sorted(summary, key=lambda x: x["avg_accuracy"], reverse=True)

# ─── Reporting ────────────────────────────────────────────────────────────
def print_table(summary, modality, exp_type):
    label = "SINGLE-AGENT" if exp_type == "single" else "DUAL-AGENT"
    print(f"\n  [{modality.upper()}] {label} RESULTS")
    print(f"  {'Model/Pairing':<45} {'Avg Acc':>8} {'Max Acc':>8} {'±Std':>7} {'Runs':>5} {'AvgIter':>8} {'SunkCost':>9}")
    print(f"  {'-'*95}")
    for r in summary:
        std_str = f"±{r['std_accuracy']:.3f}"
        iter_str = f"{r['avg_iterations']:.0f}" if r['avg_iterations'] else "—"
        cost_str = f"{r['avg_sunk_cost']:.0f}" if r['avg_sunk_cost'] is not None else "—"
        print(f"  {r['key']:<45} {r['avg_accuracy']:>8.4f} {r['max_accuracy']:>8.4f} {std_str:>7} {r['n_runs']:>5} {iter_str:>8} {cost_str:>9}")

# ─── Plotting ─────────────────────────────────────────────────────────────
def plot_cross_modality(all_modality_data, output_path):
    """Bar chart: best single vs best dual per modality."""
    if not HAS_PLOT:
        print("  ⚠️ matplotlib not available, skipping plot")
        return

    modalities = list(all_modality_data.keys())
    best_single = []
    best_dual = []

    for mod in modalities:
        singles = all_modality_data[mod]["single"]
        duals = all_modality_data[mod]["dual"]
        best_single.append(max((r["avg_accuracy"] for r in singles), default=0))
        best_dual.append(max((r["avg_accuracy"] for r in duals), default=0))

    x = np.arange(len(modalities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, best_single, width, label="Best Single-Agent",
                   color="#3498db", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, best_dual, width, label="Best Dual-Agent",
                   color="#e74c3c", alpha=0.85, edgecolor="white")

    # Value labels
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Modality", fontsize=13)
    ax.set_ylabel("Avg Accuracy (Best Pairing)", fontsize=13)
    ax.set_title("Paper 3 — Thread A: Diversity of Weights vs Depth of Reasoning\n"
                 "Best Single-Agent vs Best Dual-Agent Across Modalities", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modalities], fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0.5, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n  📊 Cross-modality chart saved to {output_path}")

def plot_full_comparison(all_modality_data, output_path):
    """3-panel figure: one panel per modality, all pairings ranked."""
    if not HAS_PLOT:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    colors = {"single": "#3498db", "dual": "#e74c3c"}

    for ax, (mod, data) in zip(axes, all_modality_data.items()):
        singles = sorted(data["single"], key=lambda x: x["avg_accuracy"])
        duals = sorted(data["dual"], key=lambda x: x["avg_accuracy"])

        all_entries = (
            [(r["key"][:30], r["avg_accuracy"], "single") for r in singles] +
            [(r["key"][:30], r["avg_accuracy"], "dual") for r in duals]
        )

        if not all_entries:
            ax.set_title(f"{mod.capitalize()} (no data)")
            continue

        labels = [e[0] for e in all_entries]
        accs = [e[1] for e in all_entries]
        clrs = [colors[e[2]] for e in all_entries]

        bars = ax.barh(range(len(labels)), accs, color=clrs, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Avg Accuracy", fontsize=10)
        ax.set_title(f"{mod.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xlim(0.4, 1.02)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    single_patch = mpatches.Patch(color=colors["single"], label="Single-Agent")
    dual_patch = mpatches.Patch(color=colors["dual"], label="Dual-Agent")
    fig.legend(handles=[single_patch, dual_patch], loc="upper center",
               ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle("Paper 3 — Thread A: All Results by Modality", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Full comparison chart saved to {output_path}")

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Paper 3 — Thread A: Cross-Modality Aggregation")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_modality_data = {}
    final_report = {}

    for modality, results_dir in MODALITIES.items():
        print(f"\n{'─'*65}")
        print(f"[*] Loading {modality.upper()} from {results_dir}")
        results = load_modality(modality, results_dir)

        single_summary = aggregate_by_model(results, "single")
        dual_summary   = aggregate_by_model(results, "dual")

        print_table(single_summary, modality, "single")
        print_table(dual_summary, modality, "dual")

        best_single = single_summary[0] if single_summary else None
        best_dual   = dual_summary[0]   if dual_summary   else None

        if best_single and best_dual:
            delta = best_dual["avg_accuracy"] - best_single["avg_accuracy"]
            winner = "DUAL ✅" if delta > 0 else "SINGLE ✅"
            print(f"\n  ▶ {modality.upper()} Winner: {winner} (Δ = {delta:+.4f})")
            print(f"    Best Single: {best_single['key']} ({best_single['avg_accuracy']:.4f})")
            print(f"    Best Dual:   {best_dual['key']} ({best_dual['avg_accuracy']:.4f})")

        all_modality_data[modality] = {
            "single": single_summary,
            "dual": dual_summary,
        }

        final_report[modality] = {
            "best_single": best_single,
            "best_dual": best_dual,
            "all_single": single_summary,
            "all_dual": dual_summary,
        }

    # Cross-modality summary
    print(f"\n{'='*65}")
    print("  CROSS-MODALITY SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Modality':<12} {'Best Single':>12} {'Best Dual':>12} {'Δ':>8} {'Winner':>10}")
    print(f"  {'-'*58}")
    for mod, data in final_report.items():
        bs = data["best_single"]["avg_accuracy"] if data["best_single"] else 0
        bd = data["best_dual"]["avg_accuracy"]   if data["best_dual"]   else 0
        delta = bd - bs
        winner = "DUAL ✅" if delta > 0 else "SINGLE ✅"
        print(f"  {mod.capitalize():<12} {bs:>12.4f} {bd:>12.4f} {delta:>+8.4f} {winner:>10}")

    # Save JSON
    out_json = os.path.join(OUTPUT_DIR, "paper3_thread_a_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
    print(f"\n  💾 Results saved to {out_json}")

    # Plots
    plot_cross_modality(all_modality_data, os.path.join(OUTPUT_DIR, "fig1_cross_modality_summary.png"))
    plot_full_comparison(all_modality_data, os.path.join(OUTPUT_DIR, "fig2_full_comparison.png"))

    print("\n🎉 THREAD A AGGREGATION COMPLETE!")

if __name__ == "__main__":
    main()
