"""
train_all_specialists.py
========================
Runs all 5 BERT specialist trainers sequentially and aggregates results.
Useful for a single overnight training run.

Usage:
    python train_all_specialists.py
    python train_all_specialists.py --dims intent binary   # only selected dims
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

DIMS_ORDER = ["binary", "intent", "technique", "severity", "surface"]
SCRIPT     = os.path.join(os.path.dirname(__file__), "train_specialist.py")
MODELS_BASE= r"f:\AI-IN-THE-LOOP\dataset_pipeline\models"


def run_dim(dim: str) -> dict:
    print(f"\n{'#'*65}")
    print(f"#  STARTING SPECIALIST: {dim.upper():^10}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'#'*65}")
    
    result = subprocess.run(
        [sys.executable, SCRIPT, "--dim", dim],
        capture_output=False,   # stream live to terminal
    )
    
    if result.returncode != 0:
        print(f"❌  [{dim}] FAILED with return code {result.returncode}")
        return {"dimension": dim, "status": "FAILED"}

    # Try to load the saved metrics
    metrics_path = os.path.join(MODELS_BASE, f"specialist_{dim}", "final", "test_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["status"] = "OK"
        return metrics
    return {"dimension": dim, "status": "OK_NO_METRICS"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dims", nargs="+",
        choices=DIMS_ORDER,
        default=DIMS_ORDER,
        help="Which dimensions to train (default: all)"
    )
    args = parser.parse_args()

    start = datetime.now()
    summary = []

    for dim in args.dims:
        result = run_dim(dim)
        summary.append(result)

    elapsed = datetime.now() - start

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ALL SPECIALISTS COMPLETED  [{elapsed}]")
    print(f"{'='*65}")
    print(f"  {'Dimension':<12} {'Status':<8} {'Accuracy':>10} {'F1-W':>8} {'F1-M':>8}")
    print(f"  {'-'*50}")
    for r in summary:
        dim    = r.get("dimension", "?")
        status = r.get("status", "?")
        acc    = r.get("eval_accuracy", r.get("accuracy", float("nan")))
        f1w    = r.get("eval_f1_weighted", r.get("f1_weighted", float("nan")))
        f1m    = r.get("eval_f1_macro",    r.get("f1_macro",    float("nan")))
        try:
            print(f"  {dim:<12} {status:<8} {acc:>10.4f} {f1w:>8.4f} {f1m:>8.4f}")
        except Exception:
            print(f"  {dim:<12} {status:<8} {'N/A':>10} {'N/A':>8} {'N/A':>8}")

    # Save aggregate summary
    summary_path = os.path.join(MODELS_BASE, "all_specialists_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\n  Full results → {summary_path}")


if __name__ == "__main__":
    main()
