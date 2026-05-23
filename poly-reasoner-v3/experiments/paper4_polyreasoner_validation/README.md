# Polyreasoner Validation Experiments

This folder holds the experiment assets that were previously associated with "Paper 4".
The intent is to keep them with the Polyreasoner tool, since they validate diversity and MoE-vs-single behavior for this system.

## Structure

- `moe_puzzle_benchmark_12/`
  - Local MoE puzzle benchmark scripts and result JSONs.
- `frontier_benchmark_30/`
  - 30-puzzle benchmark scripts, benchmark datasets, result JSONs, and generated figures.
- `aeos_orchestration/`
  - AEOS loop orchestration scripts, runners, and experiment PowerShell launchers used during the validation phase.
- `security_gatekeeper_pipeline/`
  - DistilBERT specialist and classical-ML training/evaluation scripts migrated from `dataset_pipeline` (without large model/data artifacts).
- `paper_assets/`
  - Comparative frontier images used in the writeup.

## Migration Map

- `aitl-paper/experiments/aeos/aeos_behave/paper3_thread_b/*`
  -> `poly-reasoner-v3/experiments/paper4_polyreasoner_validation/moe_puzzle_benchmark_12/*`

- `aitl-paper/experiments/aeos/aeos_behave/paper3_thread_d/*`
  -> `poly-reasoner-v3/experiments/paper4_polyreasoner_validation/frontier_benchmark_30/*`

- `aitl-paper/experiments/aeos/aeos_behave/{aggregate_*,runner*,run_exp3_*,build_paper3_assets.py,run_math_ablation.py,...}`
  -> `poly-reasoner-v3/experiments/paper4_polyreasoner_validation/aeos_orchestration/*`

- `dataset_pipeline/*.py` and `dataset_pipeline/spaces/*`
  -> `poly-reasoner-v3/experiments/paper4_polyreasoner_validation/security_gatekeeper_pipeline/*`

- `aitl-paper/paper/figures/comparative_frontier.png`
  -> `poly-reasoner-v3/experiments/paper4_polyreasoner_validation/paper_assets/comparative_frontier.png`

- `aitl-paper/paper/figures/comparative_frontier_v2.png`
  -> `poly-reasoner-v3/experiments/paper4_polyreasoner_validation/paper_assets/comparative_frontier_v2.png`

## Positioning

Treat this as Polyreasoner validation, not as a numbered paper continuation.
The recommended direction is:

- Tool-first repo and docs for Polyreasoner
- Experiments as reproducibility evidence
- Optional whitepaper/technical note that references these benchmarks
