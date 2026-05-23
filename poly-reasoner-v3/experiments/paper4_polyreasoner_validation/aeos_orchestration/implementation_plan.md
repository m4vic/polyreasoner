# Paper 3: Execution Checklist & Test Matrix

This plan outlines the exact sequence of terminal commands we need to run in the `aeos_behave` folder from start to finish. Following this sequentially ensures we build the mathematical ground truth *before* testing the advanced architectures.

## The Core Hypothesis
**Diversity of Weights vs. Depth of Reasoning / Model Size**
Does an ensemble of 2 very small, diverse models (`phi3:mini` + `qwen2.5-coder:3b`) outperform a single large general model (`qwen2.5-coder:14b`) or a deep-thinking model (`deepseek-r1`)?

---

## Phase 1: Establish Single-Agent Ground Truth (Config S)
*Goal: Measure how many iterations the individual base models waste when working alone on `tabular2`.*

#### Step 1.1: The Small Coder Baseline
```bash
python runner.py --backend ollama --model qwen2.5-coder:3b --dataset tabular2
```
#### Step 1.2: The Standard Coder Baseline
```bash
python runner.py --backend ollama --model qwen2.5-coder:7b --dataset tabular2
```
#### Step 1.3: The Small Reviewer Baseline
```bash
python runner.py --backend ollama --model phi3:mini --dataset tabular2
```

---

## Phase 2: Monolithic "Big & Thinking" Baselines (Config A)
*Goal: Test if we can solve the Sunk-Cost Fallacy just by using a bigger model or a specialized "thinking" model.*

#### Step 2.1: The "Deep Thinking" Baseline
```bash
python runner.py --backend ollama --model deepseek-r1:8b --dataset tabular2
```
#### Step 2.2: The "Big Coder" Baselines
```bash
python runner.py --backend ollama --model qwen2.5-coder:14b --dataset tabular2
python runner.py --backend ollama --model deepseek-coder-v2:16b --dataset tabular2
```

---

## Phase 3: The Asymmetric Dual-Agent Test (Config B)
*Goal: Test if 2 small/diverse models collaborating beat the 1 big/thinking model from Phase 2.*

#### Step 3.1: Ultra-Small Diverse Ensemble (~4.1GB Total)
*Hypothesis: `phi3:mini` (2.2GB) Reviewer + `qwen2.5-coder:3b` (1.9GB) Coder will outperform the monolithic `deepseek-r1:8b` (5.2GB).*
```bash
python runner_critic.py --reviewer-model phi3:mini --coder-model qwen2.5-coder:3b --dataset tabular2
```

#### Step 3.2: Standard Diverse Ensemble (~6.9GB Total)
*Hypothesis: `phi3:mini` (2.2GB) Reviewer + `qwen2.5-coder:7b` (4.7GB) Coder will outperform the monolithic `qwen2.5-coder:14b` (9.0GB).*
```bash
python runner_critic.py --reviewer-model phi3:mini --coder-model qwen2.5-coder:7b --dataset tabular2
```

#### Step 3.3: Mid-Size Diverse Ensemble (The "Worst Offender" Fix)
```bash
python runner_critic.py --reviewer-model qwen2.5-coder:7b --coder-model llama3.1:8b --dataset tabular2
```

---

## Phase 4: The Symmetric "Degenerate" Test (Config C)
*Goal: Test if the Dual-Agent gains come from **genuinely diverse weights**, or if just roleplaying as a Critic is enough.*

#### Step 4.1: Homogeneous Ensemble
```bash
python runner_critic.py --reviewer-model qwen2.5-coder:7b --coder-model qwen2.5-coder:7b --dataset tabular2
```

---

## Phase 5: Metrics & Analysis
*Goal: Compare the JSON results.*

We will calculate:
1. **Reviewer Efficiency:** `(Wasted_Phase1 - Wasted_Phase3) / Wasted_Phase1`
2. **David vs Goliath:** Does the Ultra-Small Ensemble (Step 3.1) beat the DeepSeek-R1 (Step 2.1)?
3. **Diversity vs Roleplay:** Does the Diverse Ensemble (Step 3.2) beat the Homogeneous Ensemble (Step 4.1)?
