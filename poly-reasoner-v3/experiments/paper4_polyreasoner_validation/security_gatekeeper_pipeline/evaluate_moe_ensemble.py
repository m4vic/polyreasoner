"""
Paper 3 — Experiment 2: MoE Ensemble Aggregation Evaluation
=============================================================
Loads all 6 trained expert DistilBERT models, runs them on the 
multiclass test set, and compares aggregation strategies.

Strategies:
  1. Max-confidence: Pick the expert with highest positive confidence
  2. Threshold: Flag all experts with confidence > 0.5
  3. Weighted: Weight by each expert's training F1

Also compares against the single 7-class multiclass DistilBERT baseline.

Outputs:
  - Per-strategy metrics (accuracy, macro-F1, per-class F1)
  - Confusion matrices
  - Multi-label detection analysis
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ─── Configuration ────────────────────────────────────────────────────────
DATA_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\output\advanced\clean\splits\multiclass"
EXPERTS_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\experts"
MULTICLASS_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\baseline_multiclass\final"
OUTPUT_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\ensemble_evaluation"

INTENT_NAMES = [
    "benign", "direct_injection", "system_extraction",
    "role_hijack", "obfuscation", "tool_abuse", "indirect_injection"
]

# Expert names (indices 1-6, no benign expert)
EXPERT_NAMES = [
    "direct_injection", "system_extraction", "role_hijack",
    "obfuscation", "tool_abuse", "indirect_injection"
]

# Training F1 scores for weighted voting (from training results)
EXPERT_F1_WEIGHTS = {
    "direct_injection": 0.760,
    "system_extraction": 0.693,
    "role_hijack": 0.831,
    "obfuscation": 0.751,
    "tool_abuse": 0.921,
    "indirect_injection": 0.314,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ─── Data Loading ─────────────────────────────────────────────────────────
def load_jsonl(filepath):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, np.array(labels)

# ─── Model Loading ────────────────────────────────────────────────────────
def load_experts(tokenizer):
    experts = {}
    for name in EXPERT_NAMES:
        model_path = os.path.join(EXPERTS_DIR, name, "final")
        if os.path.exists(model_path):
            print(f"  Loading expert: {name}")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            model.eval()
            experts[name] = model
        else:
            print(f"  ⚠️ Missing expert: {name}")
    return experts

def load_multiclass_baseline():
    if os.path.exists(MULTICLASS_DIR):
        print(f"  Loading multiclass baseline from {MULTICLASS_DIR}")
        model = AutoModelForSequenceClassification.from_pretrained(MULTICLASS_DIR)
        model.to(device)
        model.eval()
        return model
    return None

# ─── Inference ────────────────────────────────────────────────────────────
def get_expert_predictions(texts, tokenizer, experts, batch_size=64):
    """Run all experts on all texts, return dict of {expert_name: [confidence_scores]}."""
    all_confidences = {name: [] for name in experts}
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, 
                          max_length=512, padding=True).to(device)
        
        with torch.no_grad():
            for name, model in experts.items():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                # Class 1 = positive (this threat type detected)
                pos_conf = probs[:, 1].cpu().numpy()
                all_confidences[name].extend(pos_conf.tolist())
        
        if (i // batch_size) % 10 == 0:
            print(f"    Processed {min(i+batch_size, len(texts))}/{len(texts)} samples...")
    
    return all_confidences

def get_multiclass_predictions(texts, tokenizer, model, batch_size=64):
    """Run multiclass model, return predicted labels."""
    all_preds = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                          max_length=512, padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
    
    return np.array(all_preds)

# ─── Aggregation Strategies ──────────────────────────────────────────────
def strategy_max_confidence(confidences, threshold=0.5):
    """Pick the expert with highest confidence. If none > threshold, predict benign (0)."""
    n_samples = len(list(confidences.values())[0])
    predictions = []
    
    for i in range(n_samples):
        best_expert = None
        best_conf = 0.0
        
        for name in EXPERT_NAMES:
            conf = confidences[name][i]
            if conf > best_conf:
                best_conf = conf
                best_expert = name
        
        if best_conf > threshold:
            predictions.append(INTENT_NAMES.index(best_expert))
        else:
            predictions.append(0)  # benign
    
    return np.array(predictions)

def strategy_threshold(confidences, threshold=0.5):
    """Flag all experts above threshold. If multiple, pick highest. If none, benign."""
    return strategy_max_confidence(confidences, threshold)

def strategy_weighted(confidences, threshold=0.5):
    """Weight each expert's confidence by its training F1, then pick max."""
    n_samples = len(list(confidences.values())[0])
    predictions = []
    
    for i in range(n_samples):
        best_expert = None
        best_weighted_conf = 0.0
        
        for name in EXPERT_NAMES:
            raw_conf = confidences[name][i]
            weight = EXPERT_F1_WEIGHTS.get(name, 0.5)
            weighted_conf = raw_conf * weight
            
            if weighted_conf > best_weighted_conf:
                best_weighted_conf = weighted_conf
                best_expert = name
        
        if best_weighted_conf > threshold * EXPERT_F1_WEIGHTS.get(best_expert, 0.5):
            predictions.append(INTENT_NAMES.index(best_expert))
        else:
            predictions.append(0)
    
    return np.array(predictions)

def detect_multilabel(confidences, threshold=0.5):
    """For each sample, list all experts that flagged it (confidence > threshold)."""
    n_samples = len(list(confidences.values())[0])
    multi_flags = []
    
    for i in range(n_samples):
        flags = []
        for name in EXPERT_NAMES:
            if confidences[name][i] > threshold:
                flags.append(name)
        multi_flags.append(flags)
    
    return multi_flags

# ─── Plotting ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title, output_path):
    if not HAS_PLOT:
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=INTENT_NAMES, yticklabels=INTENT_NAMES)
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ─── Evaluation ───────────────────────────────────────────────────────────
def evaluate_predictions(y_true, y_pred, strategy_name):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    report = classification_report(y_true, y_pred, target_names=INTENT_NAMES,
                                    output_dict=True, zero_division=0)
    
    result = {
        "strategy": strategy_name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class": {}
    }
    
    for intent in INTENT_NAMES:
        if intent in report:
            result["per_class"][intent] = {
                "precision": round(report[intent]["precision"], 4),
                "recall": round(report[intent]["recall"], 4),
                "f1": round(report[intent]["f1-score"], 4),
                "support": report[intent]["support"]
            }
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(INTENT_NAMES))))
    return result, cm

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Paper 3 — Experiment 2: MoE Ensemble Evaluation")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n[*] Loading test set...")
    texts, y_true = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
    print(f"    Test samples: {len(texts)}")
    
    # Load tokenizer + models
    print("\n[*] Loading models...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    experts = load_experts(tokenizer)
    multiclass_model = load_multiclass_baseline()
    
    # Get expert predictions
    print("\n[*] Running expert inference...")
    confidences = get_expert_predictions(texts, tokenizer, experts)
    
    # Strategy 1: Max confidence
    print("\n[*] Evaluating Strategy: Max-Confidence...")
    preds_max = strategy_max_confidence(confidences)
    result_max, cm_max = evaluate_predictions(y_true, preds_max, "max_confidence")
    plot_confusion_matrix(cm_max, "MoE — Max Confidence", os.path.join(OUTPUT_DIR, "cm_max_confidence.png"))
    
    # Strategy 2: Threshold (same logic, explicit naming)
    print("[*] Evaluating Strategy: Threshold (0.5)...")
    preds_thresh = strategy_threshold(confidences, threshold=0.5)
    result_thresh, cm_thresh = evaluate_predictions(y_true, preds_thresh, "threshold_0.5")
    plot_confusion_matrix(cm_thresh, "MoE — Threshold 0.5", os.path.join(OUTPUT_DIR, "cm_threshold.png"))
    
    # Strategy 3: Weighted
    print("[*] Evaluating Strategy: Weighted by F1...")
    preds_weighted = strategy_weighted(confidences)
    result_weighted, cm_weighted = evaluate_predictions(y_true, preds_weighted, "weighted_f1")
    plot_confusion_matrix(cm_weighted, "MoE — Weighted by F1", os.path.join(OUTPUT_DIR, "cm_weighted.png"))
    
    # Multiclass baseline
    result_multiclass = None
    if multiclass_model:
        print("[*] Evaluating Multiclass Baseline...")
        preds_multi = get_multiclass_predictions(texts, tokenizer, multiclass_model)
        result_multiclass, cm_multi = evaluate_predictions(y_true, preds_multi, "multiclass_baseline")
        plot_confusion_matrix(cm_multi, "Single Multiclass DistilBERT", os.path.join(OUTPUT_DIR, "cm_multiclass_baseline.png"))
    
    # Multi-label analysis
    print("\n[*] Running Multi-Label Detection Analysis...")
    multi_flags = detect_multilabel(confidences)
    multi_label_count = sum(1 for flags in multi_flags if len(flags) > 1)
    print(f"    Samples with multiple experts flagged: {multi_label_count}/{len(texts)} ({100*multi_label_count/len(texts):.1f}%)")
    
    # Compile results
    all_results = [result_max, result_thresh, result_weighted]
    if result_multiclass:
        all_results.append(result_multiclass)
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY: ALL STRATEGIES vs MULTICLASS BASELINE")
    print("=" * 60)
    print(f"{'Strategy':<25} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>12}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['strategy']:<25} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} {r['f1_weighted']:>12.4f}")
    
    # Save
    summary_path = os.path.join(OUTPUT_DIR, "all_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n💾 Summary saved to {summary_path}")
    
    # Save multi-label stats
    ml_stats = {
        "total_samples": len(texts),
        "multi_label_detections": multi_label_count,
        "multi_label_percentage": round(100 * multi_label_count / len(texts), 2)
    }
    with open(os.path.join(OUTPUT_DIR, "multi_label_stats.json"), "w") as f:
        json.dump(ml_stats, f, indent=4)
    
    print("\n🎉 MoE ENSEMBLE EVALUATION COMPLETE!")

if __name__ == "__main__":
    main()
