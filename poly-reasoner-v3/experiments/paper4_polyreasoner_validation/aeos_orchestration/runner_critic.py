"""
AEOS — Phase 2: Agent-Critic Architecture (EXP 2)
Orchestrates a dual-agent loop:
  1. ReviewerAgent (Critic)  — analyzes history, sets strategy, can STOP the loop.
  2. CoderAgent (Worker)     — executes the strategy (never sees stopping logic).

Supports all backends: ollama, openai, anthropic, gemini

Usage examples:
    # Both local (Ollama)
    python runner_critic.py --reviewer-model qwen2.5-coder:7b --coder-model llama3.1:8b

    # Reviewer local, Coder via API
    python runner_critic.py --reviewer-backend ollama --reviewer-model qwen2.5-coder:7b \
                             --coder-backend openai --coder-model gpt-4o-mini

    # New tabular dataset
    python runner_critic.py --reviewer-model qwen2.5-coder:7b --coder-model llama3.1:8b \
                             --dataset tabular2
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from data_loader import get_data
from trainer import execute_agent_code
from reviewer import ReviewerAgent
from coder import CoderAgent

SAFETY_MAX_ITERATIONS = 75
TIMEOUT_PER_ITERATION = 300


# ─── Agent Initialization ─────────────────────────────────────────────────────

def _build_agent_model_str(backend, model_name):
    """Normalize model string for litellm based on backend."""
    prefixes = ['ollama/', 'openai/', 'anthropic/', 'gemini/']
    for p in prefixes:
        model_name = model_name.replace(p, '')
    if backend == 'ollama' and not model_name.startswith('ollama/'):
        return f'ollama/{model_name}'
    if backend == 'openai' and not model_name.startswith('openai/'):
        return f'openai/{model_name}'
    if backend == 'anthropic' and not model_name.startswith('anthropic/'):
        return f'anthropic/{model_name}'
    return model_name  # gemini / other


def _get_api_key(backend, override=None):
    """Fetch the right API key for a backend."""
    if override:
        return override
    env_map = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'gemini': 'GEMINI_API_KEY',
    }
    return os.environ.get(env_map.get(backend, ''), None)


# ─── Metrics Helpers ──────────────────────────────────────────────────────────

def _count_sunk_cost_episodes(results):
    """Count sunk-cost episodes: 3+ consecutive iters with same family and no improvement."""
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


def _reviewer_efficiency(single_agent_total, single_agent_best_iter, dual_total):
    """
    Reviewer Efficiency: fraction of wasted iterations saved.
    Wasted = iterations after the peak (post-best iterations).
    """
    wasted_single = max(0, single_agent_total - single_agent_best_iter)
    wasted_dual = max(0, dual_total - single_agent_best_iter)  # reference same peak
    if wasted_single == 0:
        return None  # single agent stopped perfectly — no improvement possible
    return round((wasted_single - wasted_dual) / wasted_single * 100, 1)


# ─── Plot Generation ──────────────────────────────────────────────────────────

def _generate_plot(results, best_acc, best_iteration, reviewer_model, coder_model,
                   dataset, results_dir, timestamp, run_tag):
    """Generate the dual-agent improvement frontier plot."""
    successful = [r for r in results if r.get('val_accuracy') is not None and r['val_accuracy'] > 0]
    if not successful:
        print("  [Plot] No successful iterations to plot.")
        return

    iterations = [r['iteration'] for r in successful]
    accuracies = [r['val_accuracy'] for r in successful]
    families = [r.get('family', 'Unknown') for r in successful]

    # Best-so-far frontier
    best_so_far = []
    running_best = 0
    for acc in accuracies:
        running_best = max(running_best, acc)
        best_so_far.append(running_best)

    family_colors = {
        'RandomForest': '#2ecc71', 'GradientBoosting': '#e74c3c',
        'ExtraTrees': '#27ae60', 'SVM': '#8e44ad',
        'LogisticRegression': '#f39c12', 'KNN': '#1abc9c',
        'DecisionTree': '#d35400', 'sklearn_MLP': '#2980b9',
        'PyTorch_NN': '#e67e22', 'AdaBoost': '#c0392b',
        'Ensemble': '#16a085', 'Unknown': '#95a5a6',
    }
    colors = [family_colors.get(f, '#95a5a6') for f in families]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(iterations, accuracies, c=colors, s=80, alpha=0.7,
               edgecolors='white', linewidth=0.5, zorder=3)
    ax.plot(iterations, best_so_far, color='#27ae60', linestyle='--',
            linewidth=2, alpha=0.8, label='Best-So-Far', zorder=2)

    if best_iteration:
        ax.scatter([best_iteration], [best_acc], c='red', s=200, marker='*',
                   zorder=5, label=f'Best: {best_acc:.4f} (iter {best_iteration})')

    # Legend entries for model families
    unique_families = sorted(set(families))
    for fam in unique_families:
        ax.scatter([], [], c=family_colors.get(fam, '#95a5a6'), s=60, label=fam)

    # Clean model names for title
    rev_clean = reviewer_model.replace('ollama/', '').replace('openai/', '').replace('anthropic/', '')
    cod_clean = coder_model.replace('ollama/', '').replace('openai/', '').replace('anthropic/', '')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title(
        f'AEOS EXP2 — Agent-Critic Architecture\n'
        f'Reviewer: {rev_clean}  |  Coder: {cod_clean}  |  Dataset: {dataset}',
        fontsize=12
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add a horizontal dashed line at peak accuracy
    if best_acc > 0:
        ax.axhline(y=best_acc, color='red', linestyle=':', alpha=0.4, linewidth=1)

    rev_safe = rev_clean.replace(':', '-').replace('/', '_')
    cod_safe = cod_clean.replace(':', '-').replace('/', '_')
    tag_str = f"_{run_tag}" if run_tag else ""
    plot_path = os.path.join(results_dir, f'exp2_plot_{rev_safe}_{cod_safe}{tag_str}_{timestamp}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Saved] Plot: {plot_path}')


# ─── Main Experiment ──────────────────────────────────────────────────────────

def run_experiment(
    reviewer_backend='ollama', reviewer_model='qwen2.5-coder:7b', reviewer_api_key=None,
    coder_backend='ollama',    coder_model='llama3.1:8b',           coder_api_key=None,
    dataset='tabular2', n_samples=10000, run_tag=''
):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.path.dirname(__file__), 'results', dataset)
    os.makedirs(results_dir, exist_ok=True)

    reviewer_model_str = _build_agent_model_str(reviewer_backend, reviewer_model)
    coder_model_str    = _build_agent_model_str(coder_backend,    coder_model)

    rev_api_key = _get_api_key(reviewer_backend, reviewer_api_key)
    cod_api_key = _get_api_key(coder_backend, coder_api_key)

    if reviewer_backend != 'ollama' and not rev_api_key:
        print(f'ERROR: No API key for reviewer backend "{reviewer_backend}"')
        sys.exit(1)
    if coder_backend != 'ollama' and not cod_api_key:
        print(f'ERROR: No API key for coder backend "{coder_backend}"')
        sys.exit(1)

    # Clean model names
    rev_clean = reviewer_model.replace('ollama/', '').replace('openai/', '').replace('anthropic/', '')
    cod_clean = coder_model.replace('ollama/', '').replace('openai/', '').replace('anthropic/', '')
    rev_safe = rev_clean.replace(':', '-').replace('/', '_')
    cod_safe = cod_clean.replace(':', '-').replace('/', '_')

    # --- Resume / Skip Logic ---
    import glob
    tag_str = f"_{run_tag}" if run_tag else ""
    existing_files = glob.glob(os.path.join(results_dir, f'exp2_{rev_safe}_{cod_safe}_{dataset}{tag_str}_*.json'))
    if existing_files:
        print(f"  [SKIPPING] Found existing dual result for {rev_safe} + {cod_safe} on {dataset} with tag '{run_tag}'. Skipping run.")
        return None

    print('=' * 70)
    print('  AEOS EXP2 — Agent-Critic Architecture')
    print(f'  Reviewer: [{reviewer_backend}] {reviewer_model}')
    print(f'  Coder:    [{coder_backend}] {coder_model}')
    print(f'  Dataset:  {dataset}')
    print(f'  Started:  {timestamp}')
    print('=' * 70)

    # ─── Load data ───
    X_train, y_train, X_val, y_val, n_features, n_classes, dataset_hint = get_data(
        dataset=dataset, n_samples=n_samples, seed=42
    )
    n_train, n_val = len(X_train), len(X_val)

    # ─── Initialize agents ───
    reviewer = ReviewerAgent(
        model=reviewer_model_str,
        api_key=rev_api_key,
        api_base='http://localhost:11434' if reviewer_backend == 'ollama' else None,
        dataset_hint=dataset_hint
    )
    coder = CoderAgent(
        model=coder_model_str,
        api_key=cod_api_key,
        api_base='http://localhost:11434' if coder_backend == 'ollama' else None,
        dataset_hint=dataset_hint
    )

    # ─── The Dual-Agent Loop ───
    all_results = []
    stop_reason = None
    start_time = time.time()

    best_loss = float('inf')
    best_acc = 0.0
    best_code = None
    best_iteration = None
    model_families_tried = set()

    print(f'\n  [Loop] Safety cap: {SAFETY_MAX_ITERATIONS} iterations')
    print(f'  [Loop] Timeout per iteration: {TIMEOUT_PER_ITERATION}s')
    print('-' * 70)

    iteration = 0
    try:
        while iteration < SAFETY_MAX_ITERATIONS:
            iteration += 1
            iter_start = time.time()

            print(f"\n{'='*70}")
            print(f'  ITERATION {iteration}')
            print(f"{'='*70}")

            # ── PHASE A: REVIEWER (CRITIC) ──
            print('  [Reviewer] Analyzing history and forming directive...')
            directive = reviewer.get_directive(
                n_features, n_classes, n_train, n_val, all_results
            )

            reviewer_stopped = directive.upper().strip().startswith('STOP')
            if reviewer_stopped:
                stop_reason = f'Reviewer autonomously stopped at iteration {iteration}'
                print(f'\n  [STOP] Reviewer ordered loop termination.')
                # Log this STOP event in results
                all_results.append({
                    'iteration': iteration,
                    'val_accuracy': None,
                    'val_loss': None,
                    'family': 'REVIEWER_STOP',
                    'is_best': False,
                    'directive': 'STOP',
                    'reviewer_fired_stop': True,
                    'error': None,
                    'time_seconds': round(time.time() - iter_start, 1),
                    'code': None,
                })
                break

            print(f'  [Directive] {directive[:200]}')

            # ── PHASE B: CODER (WORKER) ──
            print('  [Coder] Generating code based on directive...')
            try:
                code = coder.generate_code(
                    n_features=n_features, n_classes=n_classes,
                    n_train=n_train, n_val=n_val,
                    directive=directive, history=all_results,
                    best_code=best_code, timeout=TIMEOUT_PER_ITERATION
                )
            except Exception as e:
                print(f'  [ERROR] Coder LLM call failed: {e}')
                all_results.append({
                    'iteration': iteration, 'val_accuracy': None, 'val_loss': None,
                    'family': 'CODER_ERROR', 'is_best': False, 'directive': directive,
                    'reviewer_fired_stop': False,
                    'error': str(e)[:500], 'time_seconds': round(time.time() - iter_start, 1),
                    'code': None,
                })
                continue

            family = coder._detect_model_family(code)
            model_families_tried.add(family)
            print(f'  [Model Family] {family}')

            # ── PHASE C: EXECUTION ──
            print(f'  [Trainer] Executing code (timeout={TIMEOUT_PER_ITERATION}s)...')
            result, error = execute_agent_code(
                code, X_train, y_train, X_val, y_val,
                n_classes, timeout=TIMEOUT_PER_ITERATION
            )

            iter_time = time.time() - iter_start

            if error:
                print(f'  [FAILED] {error[:200]}')
                all_results.append({
                    'iteration': iteration, 'val_accuracy': None, 'val_loss': None,
                    'family': family, 'is_best': False,
                    'directive': directive, 'reviewer_fired_stop': False,
                    'error': error[:500], 'time_seconds': round(iter_time, 1), 'code': code,
                })
                continue

            val_acc = result['val_accuracy']
            val_loss = result['val_loss']

            is_best = False
            if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                is_best = True
                best_acc = val_acc
                best_loss = val_loss
                best_code = code
                best_iteration = iteration

            marker = ' * NEW BEST *' if is_best else ''
            print(f'  [RESULT] Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f} | '
                  f'Family: {family} | Time: {iter_time:.1f}s{marker}')
            if not is_best:
                print(f'  [BEST]   Still: iter {best_iteration} (acc={best_acc:.4f})')

            all_results.append({
                'iteration': iteration, 'val_accuracy': val_acc, 'val_loss': val_loss,
                'family': family, 'is_best': is_best,
                'directive': directive, 'reviewer_fired_stop': False,
                'error': None, 'time_seconds': round(iter_time, 1), 'code': code,
            })

        else:
            stop_reason = f'Safety cap reached ({SAFETY_MAX_ITERATIONS} iterations)'

    except KeyboardInterrupt:
        stop_reason = f'Manually aborted at iteration {iteration}'
        print('\n  [ABORT] Ctrl+C caught. Saving partial results...')
    except Exception as e:
        stop_reason = f'Crashed: {e}'
        print(f'\n  [ERROR] {traceback.format_exc()}')

    total_time = time.time() - start_time

    # ─── Summary ───
    print('\n' + '=' * 70)
    print('  EXPERIMENT COMPLETE')
    print('=' * 70)
    print(f'  Stop reason:         {stop_reason}')
    print(f'  Total iterations:    {iteration}')
    print(f'  Best accuracy:       {best_acc:.4f}  (iter {best_iteration})')
    print(f'  Model families:      {model_families_tried}')
    print(f'  Total time:          {total_time:.1f}s ({total_time/60:.1f} min)')

    waste_count = sum(1 for r in all_results if r.get('error'))
    sunk_cost_eps = _count_sunk_cost_episodes(all_results)
    print(f'  Wasted iterations:   {waste_count}')
    print(f'  Sunk-cost episodes:  {sunk_cost_eps}')

    # ─── Save JSON ───
    rev_safe = reviewer_model.replace(':', '-').replace('/', '_')
    cod_safe = coder_model.replace(':', '-').replace('/', '_')

    run_data = {
        'exp': 'EXP2_agent_critic',
        'run_id': timestamp,
        'reviewer_backend': reviewer_backend,
        'reviewer_model': reviewer_model,
        'coder_backend': coder_backend,
        'coder_model': coder_model,
        'dataset': dataset,
        'n_features': n_features,
        'n_classes': n_classes,
        'n_train': n_train,
        'n_val': n_val,
        'best_accuracy': best_acc,
        'best_loss': best_loss,
        'best_iteration': best_iteration,
        'best_code': best_code,
        'total_iterations': iteration,
        'stop_reason': stop_reason,
        'total_time_seconds': round(total_time, 1),
        'model_families_explored': list(model_families_tried),
        'sunk_cost_episodes': sunk_cost_eps,
        'waste_count': waste_count,
        # EXP2-specific metrics (fill after you have single-agent baselines)
        'reviewer_efficiency_pct': None,  # compute post-hoc vs Paper 1 single-agent baseline
        'iterations': all_results,
    }

    tag_str = f"_{run_tag}" if run_tag else ""
    json_path = os.path.join(results_dir, f'exp2_{rev_safe}_{cod_safe}_{dataset}{tag_str}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(run_data, f, indent=2)
    print(f'\n  [Saved] Results: {json_path}')

    # ─── Generate plot ───
    _generate_plot(
        all_results, best_acc, best_iteration,
        reviewer_model_str, coder_model_str,
        dataset, results_dir, timestamp, run_tag
    )

    return run_data


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AEOS EXP2 — Agent-Critic Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All local
  python runner_critic.py --reviewer-model qwen2.5-coder:7b --coder-model llama3.1:8b

  # Local reviewer + API coder
  python runner_critic.py --reviewer-model qwen2.5-coder:7b \\
                           --coder-backend openai --coder-model gpt-4o-mini

  # New tabular2 dataset
  python runner_critic.py --reviewer-model qwen2.5-coder:7b \\
                           --coder-model gemma4 --dataset tabular2
        """
    )

    # Reviewer
    parser.add_argument('--reviewer-backend', default='ollama',
                        choices=['ollama', 'openai', 'anthropic', 'gemini'],
                        help='Backend for the Reviewer/Critic agent (default: ollama)')
    parser.add_argument('--reviewer-model', default='qwen2.5-coder:7b',
                        help='Model name for the Reviewer agent (default: qwen2.5-coder:7b)')
    parser.add_argument('--reviewer-api-key', default=None,
                        help='API key for Reviewer (or set env var like OPENAI_API_KEY)')

    # Coder
    parser.add_argument('--coder-backend', default='ollama',
                        choices=['ollama', 'openai', 'anthropic', 'gemini'],
                        help='Backend for the Coder agent (default: ollama)')
    parser.add_argument('--coder-model', default='llama3.1:8b',
                        help='Model name for the Coder agent (default: llama3.1:8b)')
    parser.add_argument('--coder-api-key', default=None,
                        help='API key for Coder (or set env var)')

    # Experiment
    parser.add_argument('--dataset',
                        choices=['tabular', 'tabular2', 'text', 'vision'],
                        default='tabular2',
                        help='Dataset (default: tabular2 = Dry Bean, new for EXP2)')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Dataset subsample size (default: 10000)')
    parser.add_argument('--run-tag', type=str, default='',
                        help='Optional tag for repeated runs (e.g., run1, run2)')

    args = parser.parse_args()

    run_experiment(
        reviewer_backend=args.reviewer_backend,
        reviewer_model=args.reviewer_model,
        reviewer_api_key=args.reviewer_api_key,
        coder_backend=args.coder_backend,
        coder_model=args.coder_model,
        coder_api_key=args.coder_api_key,
        dataset=args.dataset,
        n_samples=args.samples,
        run_tag=args.run_tag
    )
