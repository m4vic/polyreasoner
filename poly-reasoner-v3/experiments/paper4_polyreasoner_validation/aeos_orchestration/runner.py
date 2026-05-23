"""
AITL V2 — Autonomous ML Runner
Orchestrates the full AITL loop:
  1. Load real dataset (labels stripped)
  2. Agent generates ANY model code
  3. Trainer executes and measures
  4. Feed results back to agent
  5. Agent decides when to stop (no human iteration cap)

Supports both OpenAI API and local Ollama models.

Usage:
    # OpenAI (GPT-4o-mini)
    $env:OPENAI_API_KEY="your-key"
    python runner.py --backend openai --model gpt-4o-mini

    # Local (Qwen 2.5 Coder via Ollama)
    python runner.py --backend ollama --model qwen2.5-coder:7b
"""
import os
import sys
import json
import time
import argparse
import datetime
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load .env file for API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars

from data_loader import get_data
from trainer import execute_agent_code
from agent import AutonomousAgent


# ─── Default Parameters ───
DEFAULT_TIMEOUT_PER_ITERATION = 300  # seconds (5 min — enough for neural networks)


def run_experiment(backend, model_name, api_key=None, dataset='tabular2', n_samples=10000, boundless=False, seed=None, run_tag=''):
    """Run the full AITL V2 experiment."""
    
    # Configure experiment bounds
    if boundless:
        safety_max_iterations = 100
        stagnation_patience = 999999
        min_iterations_before_stop = 5
    else:
        safety_max_iterations = 200
        stagnation_patience = 5
        min_iterations_before_stop = 5
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results", dataset)
    os.makedirs(results_dir, exist_ok=True)
    
    # Clean model name for filenames (remove provider prefixes, slashes, colons)
    model_safe = model_name
    for prefix in ['ollama/', 'openai/', 'anthropic/', 'gemini/']:
        model_safe = model_safe.replace(prefix, '')
    model_safe = model_safe.replace('/', '_').replace(':', '-')
    
    # --- Resume / Skip Logic ---
    import glob
    tag_str = f"_{run_tag}" if run_tag else ""
    existing_files = glob.glob(os.path.join(results_dir, f"exp1_{model_safe}_{dataset}{tag_str}_*.json"))
    if existing_files:
        print(f"  [SKIPPING] Found existing result for {model_safe} on {dataset} with tag '{run_tag}'. Skipping run.")
        return None

    print("=" * 70)
    print("  AEOS — Autonomous Evaluator Orchestration System")
    print(f"  Backend: {backend} | Model: {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Started: {timestamp}")
    print("=" * 70)
    
    # ─── Step 1: Load data ───
    X_train, y_train, X_val, y_val, n_features, n_classes, dataset_hint = get_data(
        dataset=dataset, n_samples=n_samples, seed=42
    )
    n_train, n_val = len(X_train), len(X_val)
    
    # ─── Step 2: Initialize agent ───
    if backend == "ollama":
        if not model_name.startswith("ollama/"):
            model_name = f"ollama/{model_name}"
        agent = AutonomousAgent(
            api_base="http://localhost:11434",
            model=model_name,
            dataset_hint=dataset_hint,
            seed=seed
        )
        print(f"\n  [Config] Using LOCAL model: {model_name} via Ollama")
        print(f"  [Config] Cost: $0")
    elif backend == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            print("ERROR: Set OPENAI_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        if not model_name.startswith("openai/"):
            model_name = f"openai/{model_name}"
        agent = AutonomousAgent(api_key=key, model=model_name, dataset_hint=dataset_hint, seed=seed)
        print(f"\n  [Config] Using OPENAI model: {model_name}")
    elif backend == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("ERROR: Set ANTHROPIC_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        if not model_name.startswith("anthropic/"):
            model_name = f"anthropic/{model_name}"
        agent = AutonomousAgent(api_key=key, model=model_name, dataset_hint=dataset_hint, seed=seed)
        print(f"\n  [Config] Using ANTHROPIC model: {model_name}")
    elif backend == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            print("ERROR: Set GEMINI_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        agent = AutonomousAgent(api_key=key, model=model_name, dataset_hint=dataset_hint, seed=seed)
        print(f"\n  [Config] Using GEMINI model: {model_name}")
    else:
        print(f"ERROR: Unknown backend '{backend}'.")
        sys.exit(1)
    
    # ─── Step 3: The AITL Loop ───
    all_results = []
    stop_reason = None
    start_time = time.time()
    
    print(f"\n  [Loop] Starting autonomous search (agent decides when to stop)")
    print(f"  [Loop] Boundless mode: {boundless}")
    print(f"  [Loop] Safety cap: {safety_max_iterations} iterations (failsafe only)")
    print(f"  [Loop] Min iterations: {min_iterations_before_stop}")
    print("-" * 70)
    
    consecutive_errors = 0
    iteration = 0
    try:
        while iteration < safety_max_iterations:
            iteration += 1
            iter_start = time.time()
            
            print(f"\n{'='*70}")
            print(f"  ITERATION {iteration}")
            print(f"{'='*70}")
            
            # Agent generates code
            try:
                code = agent.generate_code(
                    n_features=n_features,
                    n_classes=n_classes,
                    n_train=n_train,
                    n_val=n_val,
                    iteration=iteration,
                    min_iterations=min_iterations_before_stop,
                    patience=stagnation_patience,
                    timeout=DEFAULT_TIMEOUT_PER_ITERATION
                )
            except Exception as e:
                error_str = str(e).lower()
                # Fatal errors — stop immediately, don't loop
                if any(term in error_str for term in ['authenticationerror', 'invalid api key', 'incorrect api key', 'permission', 'notfounderror', 'not_found_error']):
                    stop_reason = f"FATAL: API error - {str(e)[:200]}"
                    print(f"\n  [FATAL] Unrecoverable API error. Stopping immediately.")
                    all_results.append({
                        "iteration": iteration,
                        "val_accuracy": None,
                        "val_loss": None,
                        "family": "LLM_CALL",
                        "error": str(e)[:500],
                        "time_seconds": round(time.time() - iter_start, 1),
                        "code": None,
                    })
                    break
                print(f"  [ERROR] LLM call failed: {e}")
                agent.add_feedback(iteration, 0, 0, "", "ERROR", error=str(e))
                all_results.append({
                    "iteration": iteration,
                    "val_accuracy": None,
                    "val_loss": None,
                    "family": "LLM_CALL",
                    "error": str(e)[:500],
                    "time_seconds": round(time.time() - iter_start, 1),
                    "code": None,
                })
                continue
            
            # Check for STOP signal
            if code == "STOP":
                stop_reason = f"Agent autonomously stopped at iteration {iteration}"
                print(f"\n  [STOP] Agent decided to stop. Reason: converged.")
                print(f"  [STOP] Families explored: {agent.model_families_tried}")
                break
                
            # Hard cutoff if model can't improve (iters_since_best doesn't reset on pivot)
            if agent.iters_since_best >= 30:
                stop_reason = f"System forced stop (no improvement in {agent.iters_since_best} iterations)"
                print(f"\n  [STOP] {stop_reason}. Giving up after 30 iterations without improvement.")
                break
            
            # Detect model family
            family = agent._detect_model_family(code)
            print(f"  [Model Family] {family}")
            
            # Execute the code
            print(f"  [Trainer] Executing agent code (timeout={DEFAULT_TIMEOUT_PER_ITERATION}s)...")
            result, error = execute_agent_code(
                code, X_train, y_train, X_val, y_val, 
                n_classes, timeout=DEFAULT_TIMEOUT_PER_ITERATION
            )
            
            iter_time = time.time() - iter_start
            
            if error:
                consecutive_errors += 1
                print(f"  [FAILED] {error[:200]}")
                agent.add_feedback(iteration, 0, 0, code, family, error=error)
                all_results.append({
                    "iteration": iteration,
                    "val_accuracy": None,
                    "val_loss": None,
                    "family": family,
                    "error": error[:500],
                    "time_seconds": round(iter_time, 1),
                    "code": code,
                })
                
                if consecutive_errors >= 10:
                    stop_reason = f"System forced stop (10 consecutive errors encountered)"
                    print(f"\n  [STOP] Agent stuck in an error loop. Aborting to prevent infinite looping.")
                    break
                    
                continue
            
            # Reset consecutive errors on successful run
            consecutive_errors = 0
            
            val_acc = result["val_accuracy"]
            val_loss = result["val_loss"]
            
            # Update checkpoint
            is_best = agent.update_checkpoint(iteration, val_loss, val_acc, code)
            agent.add_feedback(iteration, val_loss, val_acc, code, family, is_best=is_best)
            
            marker = " * NEW BEST *" if is_best else ""
            print(f"  [RESULT] Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f} | "
                  f"Family: {family} | Time: {iter_time:.1f}s{marker}")
            
            if is_best:
                print(f"  [BEST] New best at iteration {iteration}!")
            else:
                print(f"  [BEST] Still: iter {agent.best_iteration} "
                      f"(acc={agent.best_acc:.4f}, stagnation={agent.stagnation_counter})")
            
            all_results.append({
                "iteration": iteration,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "family": family,
                "is_best": is_best,
                "time_seconds": round(iter_time, 1),
                "code": code,
            })
            
        else:
            stop_reason = f"Safety cap reached ({safety_max_iterations} iterations)"
            
    except KeyboardInterrupt:
        stop_reason = f"Experiment manually aborted by user at iteration {iteration}"
        print(f"\n  [ABORT] Caught Ctrl+C. Saving partial results...")
    except Exception as e:
        stop_reason = f"Experiment crashed: {e}"
        print(f"\n  [ERROR] {traceback.format_exc()}")
    
    total_time = time.time() - start_time
    
    # ─── Step 4: Summary ───
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Stop reason: {stop_reason}")
    print(f"  Total iterations: {iteration}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best accuracy: {agent.best_acc:.4f}")
    print(f"  Best loss: {agent.best_loss:.4f}")
    print(f"  Best iteration: {agent.best_iteration}")
    print(f"  Model families explored: {agent.model_families_tried}")
    print(f"  Backend: {backend} ({model_name})")
    
    # ─── Step 5: Save results ───
    run_data = {
        "run_id": timestamp,
        "backend": backend,
        "model": model_name,
        "dataset": dataset,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_train": n_train,
        "n_val": n_val,
        "best_accuracy": agent.best_acc,
        "best_loss": agent.best_loss,
        "best_iteration": agent.best_iteration,
        "best_code": agent.best_code,
        "total_iterations": iteration,
        "stop_reason": stop_reason,
        "total_time_seconds": round(total_time, 1),
        "model_families_explored": list(agent.model_families_tried),
        "sunk_cost_episodes": _count_sunk_cost_episodes(all_results),
        "waste_count": sum(1 for r in all_results if r.get('error')),
        "iterations": all_results,
    }
    
    tag_str = f"_{run_tag}" if run_tag else ""
    json_path = os.path.join(results_dir, f"exp1_{model_safe}_{dataset}{tag_str}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(run_data, f, indent=2)
    print(f"\n  [Saved] Results: {json_path}")
    
    # ─── Step 6: Generate plot ───
    _generate_plot(all_results, agent, backend, model_name, results_dir, timestamp, run_tag)
    
    return run_data


def _generate_plot(results, agent, backend, model_name, results_dir, timestamp, run_tag):
    """Generate the AITL improvement frontier plot."""
    successful = [r for r in results if r["val_accuracy"] is not None]
    if not successful:
        print("  [Plot] No successful iterations to plot.")
        return
    
    iterations = [r["iteration"] for r in successful]
    accuracies = [r["val_accuracy"] for r in successful]
    families = [r["family"] for r in successful]
    
    # Best-so-far frontier
    best_so_far = []
    running_best = 0
    for acc in accuracies:
        running_best = max(running_best, acc)
        best_so_far.append(running_best)
    
    # Color map by model family
    family_colors = {
        'RandomForest': '#2ecc71',
        'GradientBoosting': '#e74c3c',
        'ExtraTrees': '#27ae60',
        'SVM': '#8e44ad',
        'LogisticRegression': '#f39c12',
        'KNN': '#1abc9c',
        'DecisionTree': '#d35400',
        'sklearn_MLP': '#2980b9',
        'PyTorch_NN': '#e67e22',
        'AdaBoost': '#c0392b',
        'Ensemble': '#16a085',
        'Unknown': '#95a5a6',
    }
    colors = [family_colors.get(f, '#95a5a6') for f in families]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Scatter by family
    ax.scatter(iterations, accuracies, c=colors, s=80, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=3)
    
    # Best-so-far line
    ax.plot(iterations, best_so_far, 'g--', linewidth=2, alpha=0.8, label='Best-So-Far', zorder=2)
    
    # Mark global best
    if agent.best_iteration:
        ax.scatter([agent.best_iteration], [agent.best_acc], 
                   c='red', s=200, marker='*', zorder=5, 
                   label=f'Best: {agent.best_acc:.4f} (iter {agent.best_iteration})')
    
    # Legend for families
    unique_families = list(set(families))
    for fam in sorted(unique_families):
        color = family_colors.get(fam, '#95a5a6')
        ax.scatter([], [], c=color, s=60, label=fam)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title(f'AITL V2 — Autonomous ML Engineering\n{backend}/{model_name} | '
                 f'Agent-controlled stopping', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    tag_str = f"_{run_tag}" if run_tag else ""
    plot_path = os.path.join(results_dir, f"v2_plot_{backend}{tag_str}_{timestamp}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Plot: {plot_path}")


def _count_sunk_cost_episodes(results):
    """Count sunk-cost episodes: 3+ consecutive iterations with same family and no improvement."""
    episodes = 0
    streak = 0
    last_family = None
    for r in results:
        if r.get('error'):
            continue
        family = r.get('family', 'Unknown')
        is_best = r.get('is_best', False)
        if family == last_family and not is_best:
            streak += 1
            if streak >= 3:
                episodes += 1
                streak = 0
        else:
            streak = 0
        last_family = family
    return episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEOS — Autonomous Evaluator Orchestration System")
    parser.add_argument("--backend", choices=["ollama", "openai", "anthropic", "gemini"], default="ollama",
                        help="LLM backend (default: ollama)")
    parser.add_argument("--model", default="gemma4",
                        help="Model name (e.g., gemma4, gpt-4o-mini, claude-3-5-sonnet-20240620)")
    parser.add_argument("--dataset", choices=["tabular", "tabular2", "text", "vision"], default="tabular",
                        help="Dataset to use (default: tabular)")
    parser.add_argument("--api-key", default=None,
                        help="API key (or set corresponding env var like OPENAI_API_KEY)")
    parser.add_argument("--samples", type=int, default=10000,
                        help="Dataset subsample size (default: 10000)")
    parser.add_argument("--boundless", action="store_true",
                        help="Run in boundless mode (disable forced pivots, up to 100 iterations)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for the LLM to ensure reproducibility")
    parser.add_argument("--run-tag", type=str, default='',
                        help="Optional tag for repeated runs (e.g., run1, run2)")
    
    args = parser.parse_args()
    
    # Set default model based on backend
    if args.backend == "openai" and args.model == "gemma4":
        args.model = "gpt-4o-mini"
    elif args.backend == "anthropic" and args.model == "gemma4":
        args.model = "claude-3-5-haiku-20241022"
    
    run_experiment(
        backend=args.backend,
        model_name=args.model,
        api_key=args.api_key,
        dataset=args.dataset,
        n_samples=args.samples,
        boundless=args.boundless,
        seed=args.seed,
        run_tag=args.run_tag
    )
