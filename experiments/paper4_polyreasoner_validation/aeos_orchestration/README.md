# AEOS Experiments — Code & Data Guide

> This directory contains all the experimental code and raw results for Papers 2 and 3.

---

## Core Code Files

| File | Purpose | Used In |
|------|---------|---------|
| `runner.py` | **Single-agent** autonomous loop. The agent writes ML code, the system executes it, returns accuracy, and the agent iterates. | Paper 2 + 3 |
| `runner_critic.py` | **Dual-agent** loop. A Reviewer model oversees a Coder model. The Reviewer can issue `DIRECTIVE: STOP` to terminate the loop. | Paper 3 (Thread A) |
| `runner_tri_agent.py` | **Tri-agent** loop. A Judge selects between two competing Coders each iteration. | Paper 3 (experimental) |
| `agent.py` | LLM wrapper — handles Ollama local models and OpenAI-compatible APIs. | All |
| `coder.py` | Coder agent: receives dataset context, writes `solve()` function. | All |
| `coder_bert.py` | BERT-augmented coder variant. | Experimental |
| `reviewer.py` | Reviewer agent: reads execution history, issues directives (improve/pivot/stop). | Paper 3 |
| `data_loader.py` | Loads datasets: `tabular2` (7-class), `MNIST` (10-class), `20newsgroups` (6-class). | All |
| `trainer.py` | Training harness for ML model evaluation. | All |

---

## Aggregation Scripts

These scripts parse the raw JSON results and produce summary statistics.

| Script | Input | Output |
|--------|-------|--------|
| `aggregate_paper3.py` | All `results/tabular2/*.json`, `results/vision/*.json`, `results/text/*.json` | `paper3_thread_a/paper3_thread_a_results.json` |
| `aggregate_single.py` | `results/tabular2/exp1_*.json` | Console summary of single-agent tabular runs |
| `aggregate_single_vision.py` | `results/vision/exp1_*.json` | Console summary of single-agent vision runs |
| `aggregate_single_text.py` | `results/text/exp1_*.json` | Console summary of single-agent text runs |
| `aggregate_dual_vision.py` | `results/vision/exp2_*.json` | Console summary of dual-agent vision runs |
| `aggregate_dual_text.py` | `results/text/exp2_*.json` | Console summary of dual-agent text runs |

---

## Run Scripts (PowerShell)

| Script | What It Does |
|--------|-------------|
| `run_all.ps1` | Runs all single + dual experiments across all modalities overnight |
| `run_tabular2_n3.ps1` | Runs N=3 single-agent tabular experiments for all 8 models |
| `run_all_vision_overnight.ps1` | Runs all vision experiments overnight |
| `run_all_text_overnight.ps1` | Runs all text experiments overnight |
| `run_exp3_dual_tabular.ps1` | Runs all dual-agent tabular pairings |
| `run_exp3_dual_vision.ps1` | Runs all dual-agent vision pairings |
| `run_exp3_dual_text.ps1` | Runs all dual-agent text pairings |
| `run_exp3_tri_overnight.ps1` | Runs tri-agent experiments overnight |

---

## Results Directory Structure

```
results/
├── tabular2/         # 140 files total
│   ├── exp1_*.json   # Single-agent runs (8 models × 3-7 runs each)
│   ├── exp2_*.json   # Dual-agent runs (10 pairings × 3 runs each)
│   ├── exp3_*.json   # Tri-agent runs (3 runs)
│   └── *.png         # Per-run accuracy plots
├── vision/           # 78 files total
│   ├── exp1_*.json   # Single-agent runs (8 models × 3 runs)
│   ├── exp2_*.json   # Dual-agent runs (5 pairings × 3 runs)
│   └── *.png         # Per-run accuracy plots
└── text/             # 77 files total
    ├── exp1_*.json   # Single-agent runs (8 models × 3 runs)
    ├── exp2_*.json   # Dual-agent runs (5 pairings × 3 runs)
    └── *.png         # Per-run accuracy plots
```

### File Naming Convention
```
exp{TYPE}_{model}_{dataset}_run{N}_{YYYYMMDD_HHMMSS}.json
```
- `exp1` = single-agent
- `exp2` = dual-agent (filename: `exp2_{reviewer}_{coder}_{dataset}_...`)
- `exp3` = tri-agent

### JSON Schema (per run)
```json
{
  "exp": "EXP1_single / EXP2_dual / EXP3_tri_agent",
  "model": "qwen2.5-coder:7b",
  "reviewer_model": "qwen2.5-coder:14b",   // dual/tri only
  "dataset": "tabular2",
  "best_accuracy": 0.9395,
  "best_iteration": 5,
  "total_iterations": 7,
  "stop_reason": "Reviewer autonomously stopped at iteration 7",
  "total_time_seconds": 330.0,
  "iterations": [
    {
      "iteration": 1,
      "val_accuracy": 0.925,
      "val_loss": 0.4065,
      "family": "RandomForest",
      "is_best": true,
      "directive": "Try to improve the current best model",
      "error": null,
      "time_seconds": 86.7,
      "code": "import sklearn..."
    }
  ]
}
```

---

## Thread-Specific Directories

### `paper3_thread_a/`
Contains the master aggregated results JSON produced by `aggregate_paper3.py`:
- `paper3_thread_a_results.json` — All models × all modalities with averages, standard deviations, and stop reasons

### `paper3_thread_b/`
The 12-puzzle MoE benchmark (v2):
- `puzzle_benchmark_v2.py` — Runs 3 panel compositions × 4 configs (single, single+CoT, MoE-Vote, MoE-Synth)
- `puzzle_benchmark_v2_results.json` — Full results with per-puzzle answers

### `paper3_thread_d/`
The 30-puzzle frontier benchmark:
- `thread_d_frontier_benchmark.py` — Benchmarks 8 local panels + 12 frontier APIs
- `analyze_results_v2.py` — Produces summary tables and diversity premium calculation
- `benchmark_data_v3.json` — 30 puzzle definitions (Logic, Math, Trick, Lateral, Constraint)
- `frontier_benchmark_results_v3.json` — Raw results with all model responses
- `README.md` — API key setup instructions

---

## Key Research Documents

| File | Purpose |
|------|---------|
| `findings.md` | Master findings document — summarizes all 4 threads with data tables |
| `questionbook.md` | Living research roadmap — formal metric definitions (SCE, CADS), experiment TODOs, Paper 4 planning |
