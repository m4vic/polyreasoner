"""
Math Prompt Ablation Batch Runner
=================================
Runs repeated math-prompt ablations and persists full reviewer calculations.
"""
import argparse
import datetime
import json
import os
import statistics
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from coder import CoderAgent
from data_loader import get_data
from reviewer import ReviewerAgent
from trainer import execute_agent_code


DEFAULT_DATASETS = ("tabular2", "text")
ALLOWED_DATASETS = {"tabular", "tabular2", "text", "vision"}
N_SAMPLES = 10_000
SAFETY_MAX_ITER = 75
TIMEOUT_ITER = 300
RUN_TAG = "mathprompt"
OLLAMA_BASE = "http://localhost:11434"

# Paper-aligned baseline (existing)
BASE_SINGLE_MODEL = "llama3.1:8b"
BASE_DUAL_REVIEWER = "qwen2.5-coder:14b"
BASE_DUAL_CODER = "deepseek-coder-v2:16b"

# Additional requested conditions
EXTRA_SINGLE_MODEL = "qwen2.5-coder:7b"     # strong coder-style single model
EXTRA_DUAL_REVIEWER = "llama3.1:8b"         # "thinking"/review role
EXTRA_DUAL_CODER = "qwen2.5-coder:3b"       # lightweight coder role


def _safe_model_str(name: str) -> str:
    return name if name.startswith("ollama/") else f"ollama/{name}"


def _slug(text: str) -> str:
    safe = text.replace("ollama/", "").replace(":", "-").replace("/", "-").replace("->", "_to_")
    return safe.replace(" ", "_")


def _print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _save_result(payload: dict, results_dir: str, filename: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n  [Saved] {path}")
    return path


def _count_sce(results: list[dict]) -> int:
    episodes = 0
    streak = 0
    last_family = None
    for item in results:
        if item.get("error"):
            continue
        family = item.get("family", "Unknown")
        is_best = item.get("is_best", False)
        if family == last_family and not is_best:
            streak += 1
            if streak >= 3:
                episodes += 1
                streak = 0
        else:
            streak = 0
        last_family = family
    return episodes


def _compute_math_fields(math_inputs: dict | None) -> dict | None:
    if not isinstance(math_inputs, dict):
        return None
    v_val = int(math_inputs.get("v_last5_valid", 0))
    delta = float(math_inputs.get("delta", 0.0))
    if delta > 0.001:
        m_val = 1.5
        m_reason = "delta > 0.001"
    elif delta < -0.005 or v_val == 0:
        m_val = 0.1
        m_reason = "delta < -0.005 or V=0"
    else:
        m_val = 0.5
        m_reason = "otherwise"
    y_val = (v_val / 5.0) * m_val
    return {
        "v": v_val,
        "delta": delta,
        "m": m_val,
        "y": round(y_val, 6),
        "m_reason": m_reason,
        "math_stop_threshold": 0.45,
        "should_stop_by_math": y_val < 0.45,
    }


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    if len(values) == 1:
        std_val = 0.0
    else:
        std_val = statistics.stdev(values)
    return {
        "mean": round(float(statistics.mean(values)), 4),
        "std": round(float(std_val), 4),
        "min": round(float(min(values)), 4),
        "max": round(float(max(values)), 4),
        "n": len(values),
    }


def build_conditions(include_extra: bool) -> list[dict]:
    conditions = [
        {
            "condition_key": "single_llama3.1-8b_math",
            "label": f"SINGLE-{BASE_SINGLE_MODEL}-MATH",
            "group": "single",
            "reviewer_model": BASE_SINGLE_MODEL,
            "coder_model": BASE_SINGLE_MODEL,
        },
        {
            "condition_key": "dual_qwen14b_deepseek16b_math",
            "label": f"DUAL-{BASE_DUAL_REVIEWER}->{BASE_DUAL_CODER}-MATH",
            "group": "dual",
            "reviewer_model": BASE_DUAL_REVIEWER,
            "coder_model": BASE_DUAL_CODER,
        },
    ]
    if include_extra:
        conditions.extend(
            [
                {
                    "condition_key": "single_qwen2.5-coder-7b_math",
                    "label": f"SINGLE-{EXTRA_SINGLE_MODEL}-MATH",
                    "group": "single",
                    "reviewer_model": EXTRA_SINGLE_MODEL,
                    "coder_model": EXTRA_SINGLE_MODEL,
                },
                {
                    "condition_key": "dual_llama3.1-8b_qwen2.5-coder-3b_math",
                    "label": f"DUAL-{EXTRA_DUAL_REVIEWER}->{EXTRA_DUAL_CODER}-MATH",
                    "group": "dual",
                    "reviewer_model": EXTRA_DUAL_REVIEWER,
                    "coder_model": EXTRA_DUAL_CODER,
                },
            ]
        )
    return conditions


def run_critic_loop(
    reviewer_model_str: str,
    coder_model_str: str,
    dataset_hint: str,
    X_train,
    y_train,
    X_val,
    y_val,
    n_features: int,
    n_classes: int,
    n_train: int,
    n_val: int,
    use_math_prompt: bool,
    label: str,
    condition_key: str,
    repeat_index: int,
    data_seed: int,
    dataset_name: str,
) -> dict:
    reviewer = ReviewerAgent(
        model=reviewer_model_str,
        api_base=OLLAMA_BASE,
        dataset_hint=dataset_hint,
        use_math_prompt=use_math_prompt,
    )
    coder = CoderAgent(
        model=coder_model_str,
        api_base=OLLAMA_BASE,
        dataset_hint=dataset_hint,
    )

    all_results = []
    families_tried = set()
    stop_reason = None
    best_loss = float("inf")
    best_acc = 0.0
    best_code = None
    best_iter = None
    iteration = 0
    start_time = time.time()

    try:
        while iteration < SAFETY_MAX_ITER:
            iteration += 1
            iter_start = time.time()

            print("\n" + "=" * 70)
            print(f"  [{label} | run {repeat_index}] ITERATION {iteration}")
            print("=" * 70)

            print("  [Reviewer] Forming directive...")
            directive = reviewer.get_directive(n_features, n_classes, n_train, n_val, all_results)
            reviewer_raw = reviewer.last_raw_response
            reviewer_math_inputs = reviewer.last_math_inputs
            reviewer_math_calc = _compute_math_fields(reviewer_math_inputs)

            if directive.upper().strip().startswith("STOP"):
                stop_reason = f"Reviewer STOP at iteration {iteration}"
                print("\n  [STOP] Reviewer ordered halt.")
                all_results.append(
                    {
                        "iteration": iteration,
                        "val_accuracy": None,
                        "val_loss": None,
                        "family": "REVIEWER_STOP",
                        "is_best": False,
                        "directive": "STOP",
                        "error": None,
                        "time_seconds": round(time.time() - iter_start, 1),
                        "code": None,
                        "reviewer_raw_response": reviewer_raw,
                        "reviewer_math_inputs": reviewer_math_inputs,
                        "reviewer_math_calculation": reviewer_math_calc,
                    }
                )
                break

            print(f"  [Directive] {directive[:200]}")

            print("  [Coder] Generating code...")
            try:
                code = coder.generate_code(
                    n_features=n_features,
                    n_classes=n_classes,
                    n_train=n_train,
                    n_val=n_val,
                    directive=directive,
                    history=all_results,
                    best_code=best_code,
                    timeout=TIMEOUT_ITER,
                )
            except Exception as exc:
                print(f"  [ERROR] Coder call failed: {exc}")
                all_results.append(
                    {
                        "iteration": iteration,
                        "val_accuracy": None,
                        "val_loss": None,
                        "family": "CODER_ERROR",
                        "is_best": False,
                        "directive": directive,
                        "error": str(exc)[:500],
                        "time_seconds": round(time.time() - iter_start, 1),
                        "code": None,
                        "reviewer_raw_response": reviewer_raw,
                        "reviewer_math_inputs": reviewer_math_inputs,
                        "reviewer_math_calculation": reviewer_math_calc,
                    }
                )
                continue

            family = coder._detect_model_family(code)
            families_tried.add(family)
            print(f"  [Family] {family}")

            print(f"  [Trainer] Executing (timeout={TIMEOUT_ITER}s)...")
            result, error = execute_agent_code(
                code,
                X_train,
                y_train,
                X_val,
                y_val,
                n_classes,
                timeout=TIMEOUT_ITER,
            )
            iter_time = time.time() - iter_start

            if error:
                print(f"  [FAILED] {error[:200]}")
                all_results.append(
                    {
                        "iteration": iteration,
                        "val_accuracy": None,
                        "val_loss": None,
                        "family": family,
                        "is_best": False,
                        "directive": directive,
                        "error": error[:500],
                        "time_seconds": round(iter_time, 1),
                        "code": code,
                        "reviewer_raw_response": reviewer_raw,
                        "reviewer_math_inputs": reviewer_math_inputs,
                        "reviewer_math_calculation": reviewer_math_calc,
                    }
                )
                continue

            val_acc = result["val_accuracy"]
            val_loss = result["val_loss"]
            is_best = False
            if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                is_best = True
                best_acc = val_acc
                best_loss = val_loss
                best_code = code
                best_iter = iteration

            marker = " * NEW BEST *" if is_best else ""
            print(
                f"  [RESULT] Acc: {val_acc:.4f} +/- Loss: {val_loss:.4f} +/- "
                f"Family: {family} | Time: {iter_time:.1f}s{marker}"
            )

            all_results.append(
                {
                    "iteration": iteration,
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "family": family,
                    "is_best": is_best,
                    "directive": directive,
                    "error": None,
                    "time_seconds": round(iter_time, 1),
                    "code": code,
                    "reviewer_raw_response": reviewer_raw,
                    "reviewer_math_inputs": reviewer_math_inputs,
                    "reviewer_math_calculation": reviewer_math_calc,
                }
            )
        else:
            stop_reason = f"Safety cap reached ({SAFETY_MAX_ITER} iterations)"

    except KeyboardInterrupt:
        stop_reason = f"Aborted by user at iteration {iteration}"
        print("\n  [ABORT] Ctrl+C caught.")
    except Exception as exc:
        stop_reason = f"Crashed: {exc}"
        print(f"\n  [ERROR] {traceback.format_exc()}")

    total_time = time.time() - start_time
    sce_count = _count_sce(all_results)
    waste_count = sum(1 for item in all_results if item.get("error"))

    print(
        f"\n  [{label} | run {repeat_index}] Done. "
        f"Iters: {iteration} | Best: {best_acc:.4f} +/- "
        f"SCEs: {sce_count} | Time: {total_time:.1f}s | Stop: {stop_reason}"
    )

    return {
        "condition_key": condition_key,
        "repeat_index": repeat_index,
        "data_seed": data_seed,
        "label": label,
        "reviewer_model": reviewer_model_str,
        "coder_model": coder_model_str,
        "dataset": dataset_name,
        "use_math_prompt": use_math_prompt,
        "run_tag": RUN_TAG,
        "best_accuracy": best_acc,
        "best_loss": best_loss,
        "best_iteration": best_iter,
        "total_iterations": iteration,
        "stop_reason": stop_reason,
        "total_time_seconds": round(total_time, 1),
        "sunk_cost_episodes": sce_count,
        "waste_count": waste_count,
        "model_families": sorted(list(families_tried)),
        "iterations": all_results,
    }


def print_run_table(results: list[dict]) -> None:
    _print_banner("MATH PROMPT ABLATION - RUN-LEVEL RESULTS")
    header = f"{'Label':<50} {'Run':>4} {'Best Acc':>9} {'Iters':>6} {'SCEs':>5} {'Time(s)':>8}  Stop Reason"
    print(header)
    print("-" * len(header))
    for item in results:
        stop = (item.get("stop_reason") or "")[:40]
        print(
            f"{item.get('label','?'):<50} "
            f"{item.get('repeat_index',0):>4} "
            f"{item.get('best_accuracy',0):>9.4f} "
            f"{item.get('total_iterations',0):>6} "
            f"{item.get('sunk_cost_episodes',0):>5} "
            f"{item.get('total_time_seconds',0):>8.1f}  {stop}"
        )


def aggregate_results(results: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in results:
        grouped.setdefault(item["condition_key"], []).append(item)

    summary = []
    for condition_key, rows in grouped.items():
        label = rows[0]["label"]
        acc_vals = [float(r["best_accuracy"]) for r in rows]
        iter_vals = [float(r["total_iterations"]) for r in rows]
        sce_vals = [float(r["sunk_cost_episodes"]) for r in rows]
        time_vals = [float(r["total_time_seconds"]) for r in rows]
        waste_vals = [float(r["waste_count"]) for r in rows]

        stop_counts: dict[str, int] = {}
        for row in rows:
            key = row.get("stop_reason") or "unknown"
            stop_counts[key] = stop_counts.get(key, 0) + 1

        summary.append(
            {
                "condition_key": condition_key,
                "label": label,
                "runs": len(rows),
                "best_accuracy": _mean_std(acc_vals),
                "iterations": _mean_std(iter_vals),
                "sunk_cost_episodes": _mean_std(sce_vals),
                "time_seconds": _mean_std(time_vals),
                "waste_count": _mean_std(waste_vals),
                "stop_reasons": stop_counts,
                "reviewer_model": rows[0]["reviewer_model"],
                "coder_model": rows[0]["coder_model"],
            }
        )

    summary.sort(key=lambda x: (-x["best_accuracy"]["mean"], x["time_seconds"]["mean"]))
    return summary


def print_aggregate_table(summary: list[dict]) -> None:
    _print_banner("MATH PROMPT ABLATION - AGGREGATE (MEAN +/- STD)")
    header = f"{'Label':<50} {'Best Acc':>16} {'Iters':>16} {'SCEs':>16} {'Time(s)':>16}"
    print(header)
    print("-" * len(header))

    def fmt(stat: dict) -> str:
        return f"{stat['mean']:.4f} +/- {stat['std']:.4f}"

    for row in summary:
        print(
            f"{row['label']:<50} "
            f"{fmt(row['best_accuracy']):>16} "
            f"{fmt(row['iterations']):>16} "
            f"{fmt(row['sunk_cost_episodes']):>16} "
            f"{fmt(row['time_seconds']):>16}"
        )

    if summary:
        winner = summary[0]
        print("\nBest by mean accuracy:")
        print(
            f"- {winner['label']} | "
            f"acc={winner['best_accuracy']['mean']:.4f}, "
            f"time={winner['time_seconds']['mean']:.1f}s, "
            f"sce={winner['sunk_cost_episodes']['mean']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Math Prompt Ablation Batch Runner")
    parser.add_argument("--skip-single", action="store_true", help="Skip all single-model conditions")
    parser.add_argument("--skip-dual", action="store_true", help="Skip all dual-model conditions")
    parser.add_argument("--repeats", type=int, default=5, help="Runs per condition (paper-ready default: 5)")
    parser.add_argument("--seed-start", type=int, default=42, help="Base seed; each repeat uses seed_start + repeat_index - 1")
    parser.add_argument("--no-extra", action="store_true", help="Run only the two original baseline conditions")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets from: tabular, tabular2, text, vision (default: tabular2,text)",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Shortcut for --datasets tabular2,text,vision",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES,
        help=f"Training pool size before split (default: {N_SAMPLES})",
    )
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    conditions = build_conditions(include_extra=not args.no_extra)
    if args.skip_single:
        conditions = [c for c in conditions if c["group"] != "single"]
    if args.skip_dual:
        conditions = [c for c in conditions if c["group"] != "dual"]
    if not conditions:
        print("No conditions selected. Remove skip flags or enable conditions.")
        return

    if args.all_datasets:
        datasets = ["tabular2", "text", "vision"]
    else:
        datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    invalid = [item for item in datasets if item not in ALLOWED_DATASETS]
    if invalid:
        raise ValueError(f"Unsupported dataset(s): {invalid}. Allowed: {sorted(ALLOWED_DATASETS)}")
    if not datasets:
        raise ValueError("No datasets selected.")

    _print_banner("MATH PROMPT ABLATION EXPERIMENT")
    print(f"  Datasets:    {', '.join(datasets)}")
    print(f"  Run tag:     {RUN_TAG}")
    print(f"  Timestamp:   {timestamp}")
    print(f"  Math prompt: ENABLED")
    print(f"  Repeats:     {args.repeats}")
    print(f"  n_samples:   {args.n_samples}")
    print(f"  Conditions:  {len(conditions)}")
    for condition in conditions:
        print(f"    - {condition['label']}")

    all_results_global = []

    for dataset_name in datasets:
        _print_banner(f"DATASET BLOCK - {dataset_name}")
        results_dir = os.path.join(os.path.dirname(__file__), "results", dataset_name)
        all_results = []

        for repeat_index in range(1, args.repeats + 1):
            data_seed = args.seed_start + (repeat_index - 1)
            _print_banner(
                f"DATA LOAD - {dataset_name} - REPEAT {repeat_index}/{args.repeats} (seed={data_seed})"
            )
            X_train, y_train, X_val, y_val, n_features, n_classes, dataset_hint = get_data(
                dataset=dataset_name, n_samples=args.n_samples, seed=data_seed
            )
            n_train, n_val = len(X_train), len(X_val)
            print(f"[Data] {n_train} train | {n_val} val | {n_features} features | {n_classes} classes")

            for cond_idx, condition in enumerate(conditions, start=1):
                _print_banner(
                    f"COND {cond_idx}/{len(conditions)} - {dataset_name} - "
                    f"REPEAT {repeat_index}/{args.repeats} - {condition['label']}"
                )
                result = run_critic_loop(
                    reviewer_model_str=_safe_model_str(condition["reviewer_model"]),
                    coder_model_str=_safe_model_str(condition["coder_model"]),
                    dataset_hint=dataset_hint,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    n_features=n_features,
                    n_classes=n_classes,
                    n_train=n_train,
                    n_val=n_val,
                    use_math_prompt=True,
                    label=condition["label"],
                    condition_key=condition["condition_key"],
                    repeat_index=repeat_index,
                    data_seed=data_seed,
                    dataset_name=dataset_name,
                )
                all_results.append(result)
                all_results_global.append(result)
                run_file = (
                    f"math_{_slug(condition['condition_key'])}_{dataset_name}_{timestamp}_run{repeat_index:02d}.json"
                )
                _save_result(result, results_dir, run_file)

        if not all_results:
            print(f"No results produced for dataset '{dataset_name}'.")
            continue

        print_run_table(all_results)
        summary = aggregate_results(all_results)
        print_aggregate_table(summary)

        summary_payload = {
            "timestamp": timestamp,
            "dataset": dataset_name,
            "run_tag": RUN_TAG,
            "repeats": args.repeats,
            "seed_start": args.seed_start,
            "n_samples": args.n_samples,
            "conditions": conditions,
            "aggregate": summary,
            "run_files_count": len(all_results),
        }
        summary_file = f"math_batch_summary_{dataset_name}_{timestamp}.json"
        _save_result(summary_payload, results_dir, summary_file)

    if all_results_global:
        combined_summary_path = os.path.join(
            os.path.dirname(__file__),
            "results",
            f"math_batch_summary_all_{timestamp}.json",
        )
        combined_payload = {
            "timestamp": timestamp,
            "datasets": datasets,
            "run_tag": RUN_TAG,
            "repeats": args.repeats,
            "seed_start": args.seed_start,
            "n_samples": args.n_samples,
            "conditions": conditions,
            "total_run_files_count": len(all_results_global),
            "results": all_results_global,
        }
        with open(combined_summary_path, "w", encoding="utf-8") as handle:
            json.dump(combined_payload, handle, indent=2)
        print(f"\n[Saved Combined] {combined_summary_path}")
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
