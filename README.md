# Polyreasoner

Polyreasoner is a multi-perspective reasoning tool for complex decisions.  
Instead of forcing one model to think in one lane, it runs multiple perspectives and then synthesizes trade-offs.

## Why This Tool Exists

Single-agent answers are often fast, but for high-stakes decisions they can miss blind spots.

Polyreasoner exists to:

- Reduce one-sided reasoning
- Surface disagreements explicitly
- Separate "quick chat" from "decision analysis"
- Make trade-offs auditable for research and engineering workflows

This mirrors human decision quality. Important choices improve when we hear multiple viewpoints before deciding.

## Human-Style Example

Question: `Should I leave my stable job and start an AI security startup?`

How most people think when the decision matters:

- Business lens: market demand, pricing, competition
- Risk lens: runway, failure scenarios, dependency risk
- Feasibility lens: can we actually build and ship this
- Contrarian lens: strongest argument against doing it now

Polyreasoner reproduces this structure programmatically and returns one synthesis with confidence and unresolved concerns.

## Scope and Non-Goals

This repository is intentionally focused on decision reasoning and evaluation logic.

- In scope: routing, perspective panels, synthesis, and judge outputs
- Not a goal: building a polished UI layer
- SafetyDiff is the complementary regression tool for release-over-release safety comparison

## Core Capabilities

- Intent router: decides whether to chat, search, analyze context, or run multi-perspective mode
- Multi-agent decision modes: career, business, decision, and manual custom panels
- Security judge mode (`/judge`) for attack/response evaluation
- Backend flexibility: local Ollama or API-based inference

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

## Key Commands

- `/career <query>`: career decision analysis
- `/business <query>`: business/startup analysis
- `/decision <query>`: general multi-perspective decision mode
- `/manual <query> --agents risk,business,...`: custom perspective set
- `/judge --attack "..." --response "..."`: security evaluation JSON output
- `/settings`: update persistent backend/model settings

## Repository Layout

| Path | Purpose |
|---|---|
| `main.py` | CLI entrypoint and interactive shell |
| `polyreasoner.py` | Programmatic wrapper |
| `router_agent.py` | Intent-based routing |
| `agents.py` | Agent execution and synthesis support |
| `modes/` | Mode-specific orchestration |
| `backend/` | Backend adapters and MoE helpers |
| `cli/` | Parser, display, and context reader |
| `experiments/` | Validation and benchmark assets |
| `run_tests.py` | Verification suite for routing/tools/backend wiring |

## Experiments and Evidence

Validation assets are in:

- `experiments/paper4_polyreasoner_validation/`
  - `moe_puzzle_benchmark_12/`
  - `frontier_benchmark_30/`
  - `aeos_orchestration/`
  - `security_gatekeeper_pipeline/`
  - `paper_assets/`

## Security and Open-Source Boundary

- Keep secrets in `.env` only.
- Do not commit model weights or local checkpoints.
- If you maintain private judge logic for ASRT, expose only integration contracts in public repos.

## GitHub Release Checklist

- `README.md` updated and accurate
- `.env` ignored
- model weights ignored
- no `__pycache__` committed
- verification suite runs: `python run_tests.py`

## License

MIT
