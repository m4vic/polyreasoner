# Polyreasoner: Diversity vs Depth

This note explains the core research question behind Polyreasoner experiments:

- **Depth**: spend more inference budget on a single model/prompt path (single, single+CoT).
- **Diversity**: use multiple heterogeneous reasoners and aggregate outputs (MoE-vote, MoE-synth, asymmetric reviewer-coder panels).

## Core Hypothesis

For adversarial and open-ended reasoning tasks, **diversity across model priors and roles** is a stronger reliability lever than only adding depth to one model.

## What We Already Have

## 1) Puzzle Benchmark (12 tasks)

Path:
- `paper4_polyreasoner_validation/moe_puzzle_benchmark_12/`

Contains:
- single baseline
- single + CoT
- MoE-vote
- MoE-synth

Use:
- first evidence for diversity premium over depth-only prompting.

## 2) Frontier Benchmark (30 tasks)

Path:
- `paper4_polyreasoner_validation/frontier_benchmark_30/`

Contains:
- benchmark definitions (`benchmark_data_v3.json`)
- raw model responses (`frontier_benchmark_results_v3.json`)
- analysis (`analyze_results_v2.py`)
- figures

Use:
- stress-tests panel diversity against stronger frontier baselines.

## 3) AEOS Orchestration Transfer

Path:
- `paper4_polyreasoner_validation/aeos_orchestration/`

Use:
- connects reviewer-coder asymmetry ideas (from AEOS) to Polyreasoner panel strategy.

## Is This Paper-Worthy?

Short answer: **yes, with final rigor pass**.

Current strengths:
- multi-benchmark evidence
- ablation-style structure already present (single vs multi-agent variants)
- reproducible scripts + result snapshots

Still needed for stronger publication claim:
1. fixed evaluation protocol (frozen seeds + exact model/version table)
2. confidence intervals/significance reporting for key deltas
3. explicit failure taxonomy (where diversity hurts vs helps)
4. cost/latency normalized comparison (accuracy per cost unit)

## Recommended Claim Language

Use:
- "Diversity can outperform depth-only strategies on selected reasoning benchmarks under fixed compute budgets."

Avoid:
- "Diversity always beats larger or deeper single-model inference."

## Minimum Next Experiments

1. Repeat all 4 modes (single, single+CoT, vote, synth) across 5 seeded runs.
2. Report mean, std, and bootstrap CI on accuracy.
3. Add cost and latency tables for each mode.
4. Add error categories (logic slip, instruction miss, hallucinated premise, adversarial compliance).

## Suggested GitHub Positioning

Treat this repository as:
- **Tool**: Polyreasoner runtime
- **Evidence**: validation experiments
- **Claim**: diversity-vs-depth tradeoff, not absolute superiority
