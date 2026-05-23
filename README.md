# Polyreasoner

Polyreasoner is a multi-perspective reasoning system focused on diversity-first inference: single-model depth vs multi-agent/model diversity.

This repository now tracks the current codebase only. Older `v1` and `v2` remain accessible through Git history and tags/releases.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
copy .env.example .env
```

3. Run CLI:

```bash
python main.py
```

4. Run web UI:

```bash
python webapp.py
```

## Repository Layout

| Path | Purpose |
|---|---|
| `main.py` | Runtime entry point |
| `polyreasoner.py` | Core orchestration logic |
| `agents.py` | Agent execution and panel logic |
| `prompts.py` and `prompts/` | Prompt templates |
| `backend/` | Backend adapters and specialist MoE helpers |
| `modes/` | Mode-specific orchestration |
| `experiments/` | Validation experiments and paper assets |
| `config/` and `config.py` | Ensemble and runtime configuration |

## Experiments and Evidence

Primary validation assets:

- `experiments/paper4_polyreasoner_validation/`
  - `moe_puzzle_benchmark_12/`
  - `frontier_benchmark_30/`
  - `aeos_orchestration/`
  - `security_gatekeeper_pipeline/`
  - `paper_assets/`

Research framing:

- `experiments/DIVERSITY_VS_DEPTH.md`

## Positioning

Polyreasoner claims are framed as:

- diversity can outperform depth-only strategies on selected reasoning tasks under fixed budgets
- performance depends on benchmark class, cost, and latency tradeoffs

## Notes

- Do not commit local model weights.
- Keep secrets in `.env` only.
- Use tags/releases for versioned milestones.

## License

MIT
