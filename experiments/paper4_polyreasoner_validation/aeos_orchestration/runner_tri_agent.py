"""
AEOS — Phase 3: Tri-Agent Architecture (EXP 3)
Orchestrates a competitive generation loop:
  1. JudgeAgent (Reviewer) — analyzes history, sets strategy.
  2. CoderAgent A          — attempts to fulfill directive.
  3. CoderAgent B          — attempts to fulfill directive.
  The system evaluates both, and only the BEST performing code is recorded
  into the official history fed back to the Judge.

Usage examples:
    # All local (Ollama)
    python runner_tri_agent.py --judge-model qwen3.5:9b --coder-a-model qwen2.5-coder:3b --coder-b-model qwen2.5-coder:7b
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
from coder_bert import BertCoderAgent

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


# ─── Plot Generation ──────────────────────────────────────────────────────────

def _generate_plot(results, best_acc, best_iteration, judge_model, coder_a_model, coder_b_model,
                   dataset, results_dir, timestamp, run_tag):
    successful = [r for r in results if r.get('val_accuracy') is not None and r['val_accuracy'] > 0]
    if not successful:
        print("  [Plot] No successful iterations to plot.")
        return

    iterations = [r['iteration'] for r in successful]
    accuracies = [r['val_accuracy'] for r in successful]
    winners = [r.get('winner', 'Unknown') for r in successful]

    best_so_far = []
    running_best = 0
    for acc in accuracies:
        running_best = max(running_best, acc)
        best_so_far.append(running_best)

    # Differentiate winners by color
    colors = ['#3498db' if w == 'A' else '#e74c3c' for w in winners]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(iterations, accuracies, c=colors, s=80, alpha=0.7,
               edgecolors='white', linewidth=0.5, zorder=3)
    ax.plot(iterations, best_so_far, color='#27ae60', linestyle='--',
            linewidth=2, alpha=0.8, label='Best-So-Far', zorder=2)

    if best_iteration:
        ax.scatter([best_iteration], [best_acc], c='#f1c40f', s=250, marker='*',
                   zorder=5, label=f'Best: {best_acc:.4f} (iter {best_iteration})')

    ax.scatter([], [], c='#3498db', s=60, label='Coder A Won')
    ax.scatter([], [], c='#e74c3c', s=60, label='Coder B Won')

    j_clean = judge_model.replace('ollama/', '')
    a_clean = coder_a_model.replace('ollama/', '')
    b_clean = coder_b_model.replace('ollama/', '')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title(
        f'AEOS Architecture D — Tri-Agent Competitive Generation\n'
        f'Judge: {j_clean}  |  Coders: {a_clean} vs {b_clean}  |  Dataset: {dataset}',
        fontsize=12
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    if best_acc > 0:
        ax.axhline(y=best_acc, color='#f1c40f', linestyle=':', alpha=0.6, linewidth=1.5)

    plot_path = os.path.join(results_dir, f'exp3_plot_tri_{dataset}_{run_tag}_{timestamp}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Saved] Plot: {plot_path}')


# ─── Main Experiment ──────────────────────────────────────────────────────────

def run_experiment(
    judge_backend='ollama', judge_model='qwen3.5:9b', judge_api_key=None,
    coder_a_backend='ollama', coder_a_model='qwen2.5-coder:3b', coder_a_api_key=None,
    coder_b_backend='ollama', coder_b_model='qwen2.5-coder:7b', coder_b_api_key=None,
    dataset='tabular2', n_samples=10000, run_tag='',
    coder_a_bert=False
):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.path.dirname(__file__), 'results', dataset)
    os.makedirs(results_dir, exist_ok=True)

    judge_str   = _build_agent_model_str(judge_backend, judge_model)
    coder_a_str = _build_agent_model_str(coder_a_backend, coder_a_model)
    coder_b_str = _build_agent_model_str(coder_b_backend, coder_b_model)

    bert_tag = ' [BERT-SPECIALIST]' if coder_a_bert else ''
    print('=' * 80)
    print('  AEOS Architecture D — Tri-Agent Competitive Generation')
    print(f'  Judge:   [{judge_backend}] {judge_model}')
    print(f'  Coder A: [{coder_a_backend}] {coder_a_model}{bert_tag}')
    print(f'  Coder B: [{coder_b_backend}] {coder_b_model}')
    print(f'  Dataset: {dataset}')
    if coder_a_bert:
        print(f'  MODE:    BERT+LLM vs LLM (Coder A constrained to transformers)')
    print('=' * 80)

    # Load data
    X_train, y_train, X_val, y_val, n_features, n_classes, dataset_hint = get_data(
        dataset=dataset, n_samples=n_samples, seed=42
    )
    n_train, n_val = len(X_train), len(X_val)

    # Initialize agents
    judge = ReviewerAgent(
        model=judge_str, api_key=_get_api_key(judge_backend, judge_api_key),
        api_base='http://localhost:11434' if judge_backend == 'ollama' else None,
        dataset_hint=dataset_hint
    )
    if coder_a_bert:
        coder_a = BertCoderAgent(
            model=coder_a_str, api_key=_get_api_key(coder_a_backend, coder_a_api_key),
            api_base='http://localhost:11434' if coder_a_backend == 'ollama' else None,
            dataset_hint=dataset_hint
        )
        print('  [Init] Coder A initialized as BertCoderAgent (BERT-specialist)')
    else:
        coder_a = CoderAgent(
            model=coder_a_str, api_key=_get_api_key(coder_a_backend, coder_a_api_key),
            api_base='http://localhost:11434' if coder_a_backend == 'ollama' else None,
            dataset_hint=dataset_hint
        )
    coder_b = CoderAgent(
        model=coder_b_str, api_key=_get_api_key(coder_b_backend, coder_b_api_key),
        api_base='http://localhost:11434' if coder_b_backend == 'ollama' else None,
        dataset_hint=dataset_hint
    )

    all_results = []
    stop_reason = None
    start_time = time.time()

    best_loss = float('inf')
    best_acc = 0.0
    best_code = None
    best_iteration = None
    model_families_tried = set()

    iteration = 0
    try:
        while iteration < SAFETY_MAX_ITERATIONS:
            iteration += 1
            iter_start = time.time()

            print(f"\n{'='*80}")
            print(f'  ITERATION {iteration}')
            print(f"{'='*80}")

            # ── PHASE A: JUDGE (CRITIC) ──
            print('  [Judge] Analyzing history and forming directive...')
            directive = judge.get_directive(
                n_features, n_classes, n_train, n_val, all_results
            )

            if directive.upper().strip().startswith('STOP'):
                stop_reason = f'Judge autonomously stopped at iteration {iteration}'
                print(f'\n  [STOP] Judge ordered loop termination.')
                break

            print(f'  [Directive] {directive[:200]}')

            # ── PHASE B & C: COMPETITIVE GENERATION & EXECUTION ──
            def execute_coder(coder_obj, name):
                print(f'  [Coder {name}] Generating and executing...')
                try:
                    code = coder_obj.generate_code(
                        n_features=n_features, n_classes=n_classes,
                        n_train=n_train, n_val=n_val,
                        directive=directive, history=all_results,
                        best_code=best_code, timeout=TIMEOUT_PER_ITERATION
                    )
                    family = coder_obj._detect_model_family(code)
                    result, error = execute_agent_code(
                        code, X_train, y_train, X_val, y_val,
                        n_classes, timeout=TIMEOUT_PER_ITERATION
                    )
                    return {'code': code, 'family': family, 'result': result, 'error': error}
                except Exception as e:
                    return {'code': None, 'family': 'CODER_ERROR', 'result': None, 'error': str(e)}

            res_a = execute_coder(coder_a, 'A')
            res_b = execute_coder(coder_b, 'B')

            # ── PHASE D: SELECTION ──
            def get_score(res):
                if res['error'] or res['result'] is None: return -1.0, float('inf')
                return res['result']['val_accuracy'], res['result']['val_loss']

            acc_a, loss_a = get_score(res_a)
            acc_b, loss_b = get_score(res_b)

            winner_name = None
            winner_res = None
            if acc_a == -1.0 and acc_b == -1.0:
                print('  [RESULT] Both Coders FAILED.')
                winner_name = 'NONE'
                winner_res = res_a # just log A's failure
            elif acc_a > acc_b or (acc_a == acc_b and loss_a <= loss_b):
                print(f"  [RESULT] Coder A WINS! (A: {acc_a:.4f} vs B: {acc_b:.4f})")
                winner_name = 'A'
                winner_res = res_a
            else:
                print(f"  [RESULT] Coder B WINS! (B: {acc_b:.4f} vs A: {acc_a:.4f})")
                winner_name = 'B'
                winner_res = res_b

            iter_time = time.time() - iter_start

            # Record winning result into official history
            if winner_name != 'NONE':
                val_acc = winner_res['result']['val_accuracy']
                val_loss = winner_res['result']['val_loss']
                family = winner_res['family']
                code = winner_res['code']
                model_families_tried.add(family)

                is_best = False
                if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                    is_best = True
                    best_acc = val_acc
                    best_loss = val_loss
                    best_code = code
                    best_iteration = iteration

                all_results.append({
                    'iteration': iteration, 'val_accuracy': val_acc, 'val_loss': val_loss,
                    'family': family, 'is_best': is_best,
                    'directive': directive, 'error': None,
                    'time_seconds': round(iter_time, 1), 'code': code,
                    'winner': winner_name,
                    'acc_a': acc_a if acc_a != -1.0 else None,
                    'acc_b': acc_b if acc_b != -1.0 else None,
                })
                
                marker = ' * NEW OVERALL BEST *' if is_best else ''
                print(f'  [RECORDED] Acc: {val_acc:.4f} | Family: {family} | Time: {iter_time:.1f}s{marker}')
            else:
                # Both failed, record failure to history
                all_results.append({
                    'iteration': iteration, 'val_accuracy': None, 'val_loss': None,
                    'family': 'DUAL_ERROR', 'is_best': False,
                    'directive': directive, 'error': res_a['error'][:500] + " | " + res_b['error'][:500],
                    'time_seconds': round(iter_time, 1), 'code': None,
                    'winner': 'NONE'
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

    print('\n' + '=' * 80)
    print('  EXPERIMENT COMPLETE (Tri-Agent)')
    print('=' * 80)
    print(f'  Stop reason:         {stop_reason}')
    print(f'  Total iterations:    {iteration}')
    print(f'  Best accuracy:       {best_acc:.4f}  (iter {best_iteration})')
    print(f'  Model families:      {model_families_tried}')
    print(f'  Total time:          {total_time:.1f}s ({total_time/60:.1f} min)')

    # Count wins
    wins_a = sum(1 for r in all_results if r.get('winner') == 'A')
    wins_b = sum(1 for r in all_results if r.get('winner') == 'B')
    print(f'  Coder A Wins:        {wins_a}')
    print(f'  Coder B Wins:        {wins_b}')

    run_data = {
        'exp': 'EXP3_tri_agent',
        'run_id': timestamp,
        'judge_model': judge_model,
        'coder_a_model': coder_a_model,
        'coder_a_bert': coder_a_bert,
        'coder_b_model': coder_b_model,
        'dataset': dataset,
        'best_accuracy': best_acc,
        'best_iteration': best_iteration,
        'total_iterations': iteration,
        'stop_reason': stop_reason,
        'total_time_seconds': round(total_time, 1),
        'wins_a': wins_a,
        'wins_b': wins_b,
        'iterations': all_results,
    }

    tag_str = f"_{run_tag}" if run_tag else ""
    bert_suffix = '_bert' if coder_a_bert else ''
    json_path = os.path.join(results_dir, f'exp3_tri_{dataset}{bert_suffix}{tag_str}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(run_data, f, indent=2)
    print(f'\n  [Saved] Results: {json_path}')

    _generate_plot(all_results, best_acc, best_iteration, judge_str, coder_a_str, coder_b_str, dataset, results_dir, timestamp, run_tag)
    return run_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AEOS Architecture D — Tri-Agent Competitive Generation')
    parser.add_argument('--judge-model', default='qwen3.5:9b', help='Judge model')
    parser.add_argument('--coder-a-model', default='qwen2.5-coder:3b', help='Coder A model')
    parser.add_argument('--coder-b-model', default='qwen2.5-coder:7b', help='Coder B model')
    parser.add_argument('--coder-a-bert', action='store_true',
                        help='Use BERT-specialist system prompt for Coder A (BERT+LLM vs LLM mode)')
    parser.add_argument('--dataset', choices=['tabular', 'tabular2', 'text', 'vision'], default='tabular2')
    parser.add_argument('--run-tag', type=str, default='')

    args = parser.parse_args()
    run_experiment(
        judge_model=args.judge_model,
        coder_a_model=args.coder_a_model,
        coder_b_model=args.coder_b_model,
        dataset=args.dataset,
        run_tag=args.run_tag,
        coder_a_bert=args.coder_a_bert
    )
