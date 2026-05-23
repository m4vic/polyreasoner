# AEOS Research Roadmap
> A living questionbook for AEOS, AEOSPocketLab, ASRT, and PolyReasoner experiments.
> **Last Updated:** 2026-05-12 — Major revision addressing peer review critique.

---

# Paper 3: Cognitive Agentic Diversity Over Depth of Reasoning

**Working title:** *Cognitive Agentic Diversity Beats Monolithic Scaling: Asymmetric Reviewer-Coder Ensembles in Autonomous ML Engineering Loops*

> **Framing note:** We deliberately avoid the term "Mixture of Experts" (MoE) to prevent confusion with sparse architectural MoE (Mixtral, DeepSeekMoE). Our contribution is *agentic* diversity — different models playing functionally asymmetric roles (Reviewer vs. Coder) with genuinely different weight priors — distinct from token-routing MoE.

**Unified Thesis Arc (explicit):**
> *Ensemble diversity across specialized agentic roles consistently beats monolithic scaling across reasoning, engineering, and security domains. A panel of small, role-differentiated models with diverse weight priors outperforms a single large model "thinking harder" — because diversity eliminates cognitive blindspots that scale alone cannot.*

---

## Core Claim (Revised)
A dual-agent architecture where a critic/reviewer with **genuinely different weights** oversees a coder can:
1. Reduce sunk-cost path dependency to near-zero
2. Match or exceed large frontier models (30B–405B+) on structured reasoning tasks
3. Do so at a fraction of the inference cost

The "two small models beat one large model" claim is only valid when tested against **actually large** models (≥30B). Thread D provides this evidence.

---

## Formal Metric Definitions

### Sunk-Cost Episode (SCE)
> A **Sunk-Cost Episode** is defined as a block of N ≥ 5 consecutive iterations where:
> - Validation accuracy improvement is < 0.001 (i.e., effectively zero), AND
> - The model family has not been changed (same architecture class), AND
> - The agent does not issue `DIRECTIVE: STOP`
>
> SCE count = total number of such blocks within a single run.

### Cognitive Agentic Diversity Score (CADS)
> CADS = number of distinct model families represented in a panel.
> - Family boundaries: {Qwen-Coder, LLaMA, DeepSeek, Gemma, Phi, Mistral} are distinct families.
> - Same-model control: two instances of the same model = CADS of 1 (the ablation baseline).

---

## Experiment Threads

### Thread A — AEOS Cross-Modality Benchmarks ✅ COMPLETE

| Modality | Dataset | Status | Best Result |
|----------|---------|--------|-------------|
| Tabular | tabular2 (7-class) | ✅ Done | Single llama3.1:8b 94.9%; Dual 14b→16b 93.7% at 10× efficiency, 0 SCE |
| Vision | MNIST | ✅ Done | **Dual qwen3.5:9b→coder:7b 98.4%** (peak 99.1%) — beats best single |
| Text | 20 Newsgroups | ✅ Done | Single llama3.1:8b 89.9%; Dual 81.2% — honest negative |

**Key Thread A findings:**
- Dual-agent eliminates Sunk-Cost completely on Tabular (0 SCE vs 8.7 SCE single)
- Dual-agent wins on Vision (+0.13% avg, +0.78% peak)
- Dual-agent FAILS on Text (−8.7%) — the honest boundary condition

**The qwen3.5:9b Paradox (scheduled for formal investigation):**
- *Worst* reviewer on Tabular: 75-iteration safety cap, 20.3 SCE, 3,427s, 92.9% accuracy
- *Best* reviewer on Vision: 75-iteration safety cap, 10.3 SCE, 6,612s, **98.4% accuracy**
- **Hypothesis:** *Task Dimensionality Dictates Persistence Value.* Tabular solution space is small and exhaustible — persistence is wasteful. Vision loss landscape is non-convex and deep — persistence breaks through local minima that early-stopping reviewers miss. This predicts the Meta-Controller policy in Paper 4.

**Same-Model Control (Ablation for reviewer diversity):**
- `qwen2.5-coder:7b → qwen2.5-coder:7b` on Tabular: 92.8% accuracy, 0 SCE, 6.5 iters
- Vs. asymmetric best: 93.7% (+0.9%), same SCE, similar iters
- *Interpretation:* Some gain comes from having *any* second perspective; the additional gain comes from weight asymmetry. **Needs Vision + Text runs to be a clean ablation.**
- **TODO:** Run same-model control on Vision and Text modalities.

---

### Thread B — Puzzle Benchmark (MoE vs Single Reasoning) ✅ COMPLETE (v2)

| Config | Mixed-Family (CADS=3) | Reasoning-Heavy (CADS=2) | Small-Models (CADS=1) |
|--------|:--------------------:|:-----------------------:|:--------------------:|
| Single | 3/12 (25%) | 3/12 (25%) | 2/12 (17%) |
| Single+CoT | 2/12 (17%) | 3/12 (25%) | 2/12 (17%) |
| **MoE-Vote** | **8/12 (67%)** | **6/12 (50%)** | **5/12 (42%)** |
| MoE-Synth | 0/12 (0%) | 4/12 (33%) | 2/12 (17%) |

**Diversity Gradient confirmed:** CADS=3 → 67%, CADS=2 → 50%, CADS=1 → 42%. More family diversity = better ensemble.

**Caveats to address in paper:**
1. **Statistical thinness:** n=12 puzzles — one flip = 8% change. **Expanding to 30 puzzles.**
2. **CoT conclusion softened:** "CoT provides no consistent improvement in our test conditions" (not "CoT is useless" — CoT is prompt-sensitive and we tested one variant).
3. **Lateral puzzles (0/X all panels):** A clean negative result. Hypothesis: *Token-prediction models follow the highest-probability semantic path. Lateral thinking requires a domain-shift "aha!" that violates this prior — regardless of ensemble size. This is a fundamental limitation of autoregressive generation, not a solvable ensemble problem.*
4. **Human baseline:** Adding estimated human performance column (sourced from published results on these classic puzzles — most humans score ~70-85% on this set).

**Thread B Expansion (TODO):**
- Expand to 30 puzzles (add 18: more logic, math, constraint, analogical reasoning)
- Source additional puzzles from BigBench Hard subsets or curate manually
- Add human baseline column

---

### Thread C — PolyReasoner Security Baselines ⬜ PENDING

Baseline classical ML results on 7-class prompt-injection dataset:

| Model | Accuracy | F1-Macro | Status |
|-------|:--------:|:--------:|--------|
| LogReg | 78.7% | 0.731 | ✅ Done |
| Linear SVM | 78.7% | 0.736 | ✅ Done |
| Random Forest | 78.1% | 0.712 | ✅ Done |
| XGBoost | 73.3% | 0.677 | ✅ Done |
| BERT classifier | — | — | ⬜ Pending |
| BERT+ML Ensemble | — | — | ⬜ Pending |
| BERT+ML+LLM (PolyReasoner) | — | — | ⬜ Pending |
| Large LLM alone (GPT-4o) | — | — | ⬜ Pending |

**Scope clarification:** Thread C belongs to the PolyReasoner / ASRT paper track, NOT Paper 3. It is listed in findings.md for completeness but will not block Paper 3 submission.

---

### Thread D — Frontier Baseline ("The Title Proof") ⬜ PENDING

This thread directly validates the paper's headline claim: "Two Small Models Beat One Large Model."
Without testing against a genuinely large model, the claim is scientifically invalid.

**Script:** `paper3_thread_d/thread_d_frontier_benchmark.py`

**Model Lineup:**

| Tier | Model | Provider | Cost | Params |
|------|-------|----------|------|--------|
| **LOCAL ENSEMBLE** | qwen2.5-coder:7b + llama3.1:8b + deepseek-coder:6.7b | Ollama (local) | Free | 7–17B |
| Open Large | Llama-3.3-70B | Groq | **Free** | 70B |
| Open Large | Qwen-QwQ-32B | Groq | **Free** | 32B |
| Open Large | Qwen3-30B | OpenRouter | **Free** | 30B |
| Open Large | Qwen2.5-72B | Qwen Dashscope | **Free tier** | 72B |
| Open Massive | Llama-3.1-405B | SambaNova | Free credits | 405B |
| Open Massive | Llama-3.1-405B | OpenRouter | **Free** | 405B |
| Frontier Proprietary | GPT-4o-mini | OpenAI | Paid (cheap) | ~? |
| Frontier Proprietary | GPT-4o | OpenAI | Paid | ~? |
| Frontier Proprietary | GPT-4.1 | OpenAI | Paid | ~? |

**Expected hypothesis:** Our local 7-17B ensemble (MoE-Vote, CADS=3) achieved 67% on the 12-puzzle set. We predict this will be competitive with Llama-3.3-70B and Qwen-72B, and within 10-15% of GPT-4o — demonstrating the efficiency argument even if not a strict accuracy win.

**Setup (run once before executing):**
```powershell
# Install openai client
pip install openai

# Set API keys (add to your shell profile for persistence)
$env:GROQ_API_KEY       = "gsk_..."      # groq.com — free
$env:OPENROUTER_API_KEY = "sk-or-..."    # openrouter.ai — free models available
$env:SAMBANOVA_API_KEY  = "..."          # cloud.sambanova.ai — free credits
$env:QWEN_API_KEY       = "sk-..."       # dashscope.aliyuncs.com — free tier
$env:OPENAI_API_KEY     = "sk-..."       # platform.openai.com — paid

# Run Thread D
cd f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\aeos_behave\paper3_thread_d
python thread_d_frontier_benchmark.py
```

**Note:** The script is provider-aware — it skips any provider whose API key is missing. You can start with just Groq (free) and OpenAI keys and it will skip the rest gracefully.

---

## Remaining TODOs for Paper 3 Submission

| Priority | Task | Blocker |
|----------|------|---------|
| 🔴 P0 | Run Thread D (Frontier Baseline) | Needs API keys set |
| 🔴 P0 | Run same-model control on Vision + Text | Local Ollama only |
| 🟡 P1 | Expand Thread B to 30 puzzles + human baseline | Manual curation |
| 🟡 P1 | Write formal SCE definition into paper draft | Done in this doc |
| 🟢 P2 | Thread C BERT+LLM runs | Separate paper track |

---

# Paper 4: The Zero-Human Sandbox

**Working title:** *Closed Loop on Top of Closed Loop: Autonomous Laboratory Directors for AI Systems*

## Layer 3 Meta-Orchestrator

The controller reads previous results, forms hypotheses, chooses model pairs, launches AEOS runs, watches logs, kills stalled runs, writes lab notes, and decides the next experiment.

## Minimal Controller Loop
1. Read `results/*/*.json`.
2. Aggregate accuracy, time, SCE count, waste episodes, and stop reason.
3. Form a hypothesis using the qwen3.5:9b paradox rule: *if dataset dimensionality is high AND reviewer is persistent-type → raise iteration cap; else → enforce strict stop at SCE>3.*
4. Choose the next model pair based on CADS score.
5. Execute the run script.
6. Save a lab-notebook entry explaining the decision.

## Safety Boundaries (Formalized)

The Layer 3 controller is an autonomous agent with the ability to launch subprocesses and write files. Failure modes must be explicitly bounded:

| Risk | Mitigation |
|------|-----------|
| Runaway compute | Hard cap: max $5 / 500 API tokens per hypothesis cycle |
| Bad hypothesis loop | Max 3 AEOS experiments per hypothesis before human review required |
| Destructive actions | Sandboxed execution — no `rm`, no pip installs, no network writes outside designated result dirs |
| Stalled run | Watchdog timer: kill any subprocess exceeding 2× expected runtime |
| Infinite meta-loop | Hard epoch counter: controller self-terminates after 10 cycles without accuracy improvement |

The controller must log every decision in plain language before executing: `"I am launching experiment X because hypothesis Y was formed from evidence Z."`

## Latency/Cost Overhead Analysis (Required for Paper 4)
- Layer 3 adds one LLM call per experiment cycle (hypothesis formation + next-step decision)
- At Groq free tier: ~1-2s per Layer 3 call, $0
- Efficiency break-even: If Layer 3 saves even one 3,432s sunk-cost single-agent run, it pays for hundreds of meta-controller cycles
- **TODO:** Measure and report Layer 3 overhead as % of total experiment runtime

---

# AEOSPocketLab: Mobile Version

**Working title:** *AEOSPocketLab: Offline Autonomous AI Labs on Resource-Constrained Devices*

## Success Criteria (Formalized)
> PocketLab is successful when it achieves **< 5% accuracy degradation** compared to desktop AEOS, operating within a **≤ 2GB RAM budget** and **≤ 30s per iteration** on a mid-range mobile CPU.

## Architecture

| Layer | Desktop AEOS | PocketLab Variant |
|-------|-------------|-------------------|
| Controller | Full CLI meta-orchestrator | Restricted mobile task planner |
| Reviewer | 7B–14B local/API LLM | Quantized 1.5B–3B (e.g., qwen2.5-coder:1.5b) |
| Coder | Local/API coding LLM | Template/codelet generator or 3B LLM |
| Evaluator | Python/sklearn/PyTorch | Tiny benchmark harness or on-device inference |
| Memory | JSON result folders | SQLite / compact JSONL lab notebook |

## Evaluation Questions
1. Does a 1.5B reviewer reduce SCE vs. no reviewer? (minimum useful reviewer size)
2. At what model size does CADS advantage disappear?
3. Can a mobile controller run meaningful science without large-model API calls?

---

# ASRT Track

ASRT begins **only after** Paper 4 demonstrates one complete autonomous lab cycle end-to-end.

## Entry Condition (Hard Gate)
ASRT starts when the AEOS Paper 4 controller completes:
1. Read prior benchmark results
2. Propose a next experiment
3. Launch the correct run script
4. Monitor or recover from the run
5. Write a lab-note summary
6. Decide whether more evidence is needed

…**without human step-by-step input.**

## ASRT Sequence
1. Freeze prompt-injection dataset + threat matrix labels
2. Train baselines: TF-IDF+LogReg, SVM, RandomForest, XGBoost
3. Fine-tune BERT/DistilBERT/RoBERTa classifiers
4. Build ensemble security judge: ML + BERT + rules → verdict
5. Compare vs. decoder-only LLM judges (GPT-4o, local 7B)
6. Add PolyReasoner specialist modules

---

# PolyReasoner & Hybrid Architectures

**Working title:** *PolyReasoner: Multi-Perspective Synthesis and Encoder-Decoder Pipelines vs. Monolithic Scaling*

## Core Experiment
| System | Pipeline |
|--------|---------|
| Large Decoder Alone | Input → large LLM judge (GPT-4o or 70B) |
| BERT + Small Decoder | BERT classifies risk → small decoder explains |
| ML + BERT + Small Decoder | Ensemble → structured security state → decoder narrates |
| PolyReasoner Judge | Multiple specialist judges vote → synthesizer decides |

**Hypothesis:** Hybrid encoder/ML front-end + smaller decoder beats a large decoder alone on accuracy, latency, robustness, and interpretability.

## Judge Modules
| Module | Responsibility |
|--------|---------------|
| `Judge_Lexical` | Suspicious tokens, override phrases, injection templates |
| `Judge_Semantic` | BERT/RoBERTa threat intent classification |
| `Judge_Context` | Role/system/tool boundary violations |
| `Judge_Exfiltration` | Secrets, data leakage, hidden prompt requests |
| `Judge_Code` | Malicious code/tool-use instructions |
| `Judge_Policy` | Maps threat class + severity → allow/block/sanitize |
| `Synthesizer` | Merges votes → final verdict with confidence + rationale |

---

# Execution Order (Updated)

| Step | Task | Status |
|------|------|--------|
| 1 | ✅ Finish AEOS text runs | Done |
| 2 | ✅ Aggregate Paper 3 cross-modality results (Thread A) | Done |
| 3 | ✅ Run Thread B Puzzle Benchmark (v2, 3 panels) | Done |
| 4 | 🔴 Set API keys + Run Thread D Frontier Baseline | **Next** |
| 5 | 🔴 Run same-model ablation (Vision + Text) | Next |
| 6 | 🟡 Expand Thread B to 30 puzzles + human baseline | Soon |
| 7 | 🟡 Build Paper 4 controller skeleton | After Thread D |
| 8 | 🟡 Demonstrate one full closed-loop-on-closed-loop cycle | Paper 4 |
| 9 | ⬜ Freeze ASRT dataset schema | After Paper 4 |
| 10 | ⬜ Train ML + BERT baselines (ASRT) | After Paper 4 |
| 11 | ⬜ Build ensemble security judge | After Paper 4 |
| 12 | ⬜ Run BERT+decoder vs large-decoder experiments | Final |
