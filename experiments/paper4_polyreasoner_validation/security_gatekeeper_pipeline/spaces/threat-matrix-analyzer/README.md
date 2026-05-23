---
title: Threat Matrix Analyzer
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
license: apache-2.0
models:
- neuralchemy/distilbert-binary-threat-matrix
- neuralchemy/distilbert-multiclass-threat-matrix
- neuralchemy/distilbert-expert-direct-injection-threat-matrix
- neuralchemy/distilbert-expert-indirect-injection-threat-matrix
- neuralchemy/distilbert-expert-obfuscation-threat-matrix
- neuralchemy/distilbert-expert-role-hijack-threat-matrix
- neuralchemy/distilbert-expert-system-extraction-threat-matrix
- neuralchemy/distilbert-expert-tool-abuse-threat-matrix
- neuralchemy/classical-ml-threat-matrix
datasets:
- neuralchemy/prompt-injection-Threat-Matrix
tags:
- security
- prompt-injection
- threat-detection
- mixture-of-experts
- neuralchemy
---

# Threat Matrix Analyzer

Analyze any text against **all NeurAlchemy security models simultaneously**:

- **Binary DistilBERT** (99.1% accuracy) — Benign vs Malicious
- **Multiclass DistilBERT** (80.9% accuracy) — 7 threat classes
- **6 Expert Models** — One-vs-rest MoE specialists with majority voting
- **4 Classical ML Baselines** — TF-IDF + LogReg/SVM/RF/XGB

## Models Used

| Model | Accuracy | Type |
|-------|:--------:|------|
| distilbert-binary-threat-matrix | 99.1% | Binary gate |
| distilbert-multiclass-threat-matrix | 80.9% | 7-class classifier |
| 6x distilbert-expert-*-threat-matrix | 86-98% | One-vs-rest experts |
| classical-ml-threat-matrix | 73-79% | sklearn pipelines |

## Links

- **Dataset:** [neuralchemy/prompt-injection-Threat-Matrix](https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix)
- **Paper:** [AITL on Zenodo](https://zenodo.org/records/19551173)
- **Code:** [github.com/m4vic/AEOS](https://github.com/m4vic/AEOS)
- **Lab:** [neuralchemy.in](https://neuralchemy.in)
