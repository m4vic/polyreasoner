# AEOS Benchmark Execution Tracker

This document tracks all completed and pending runs across the three target domains (Tabular, Vision, Text) to prove the "Diversity of Weights" hypothesis for the AITL paper.

---

## 1. Single-Model Baselines (Architecture: Single LLM)
✅ **ALL COMPLETE** — Tabular (3 runs each), Vision (3 runs each), Text (3 runs each)

---

## 2. Dual-Agent Architectures (A, B, C)
✅ **ALL COMPLETE** — All configs across all 3 datasets

---

## 3. Architecture D: Tri-Agent Competitive Generation

*Runner: `runner_tri_agent.py` | Script: `run_exp3_tri_overnight.ps1`*

### Config 1: Asymmetric Small vs Mid (Diverse Coders)
Judge: `qwen3.5:9b` | Coder A: `qwen2.5-coder:3b` | Coder B: `qwen2.5-coder:7b`
- [ ] Tabular × 3 runs
- [ ] Text × 3 runs
- [ ] Vision × 3 runs

### Config 2: Heavyweight Diverse Architectures
Judge: `llama3.1:8b` | Coder A: `qwen2.5-coder:7b` | Coder B: `deepseek-coder-v2:16b`
- [ ] Tabular × 3 runs
- [ ] Text × 3 runs
- [ ] Vision × 3 runs

### Config 3: Small Judge + 2x Same Mid Coder
Judge: `qwen2.5-coder:3b` | Coder A: `qwen2.5-coder:7b` | Coder B: `qwen2.5-coder:7b`
- [ ] Tabular × 3 runs
- [ ] Text × 3 runs
- [ ] Vision × 3 runs

### Config 4: Tiny Judge + 2x Same Big Coder
Judge: `phi3:mini` | Coder A: `qwen2.5-coder:14b` | Coder B: `qwen2.5-coder:14b`
- [ ] Tabular × 3 runs
- [ ] Text × 3 runs
- [ ] Vision × 3 runs

### Config 5: Small Thinker Judge + 2x Same Mid Coder
Judge: `qwen3.5:4b` | Coder A: `qwen2.5-coder:7b` | Coder B: `qwen2.5-coder:7b`
- [ ] Tabular × 3 runs
- [ ] Text × 3 runs
- [ ] Vision × 3 runs

**Total: 45 experiments (5 configs × 3 datasets × 3 runs)**

---

## 4. Execution Checklist

1. [x] Single-agent baselines (all modalities)
2. [x] Dual-agent architectures A, B, C (all modalities)
3. [x] Implement `runner_tri_agent.py`
4. [x] Create overnight batch `run_exp3_tri_overnight.ps1`
5. [ ] **Execute overnight tri-agent experiments**
6. [ ] Aggregate tri-agent results
7. [ ] Update `findings.md` with Architecture D section
