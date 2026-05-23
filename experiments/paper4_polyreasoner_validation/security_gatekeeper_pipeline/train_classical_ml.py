"""
Paper 3 — Experiment 1: Classical ML Baselines
================================================
Trains 4 classical ML models on the same 7-class multiclass dataset 
using TF-IDF features. This provides the baseline comparison for 
RQ2: "How do classical ML models compare to BERT?"

Models trained:
  1. Logistic Regression
  2. Linear SVM
  3. Random Forest
  4. XGBoost (Gradient Boosting)

Outputs per model:
  - test_metrics.json (accuracy, macro-F1, weighted-F1, per-class report)
  - confusion_matrix.png
  - Saved sklearn pipeline (vectorizer + model)
"""

import os
import json
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import joblib

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
OUTPUT_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\classical"

INTENT_NAMES = [
    "benign", "direct_injection", "system_extraction",
    "role_hijack", "obfuscation", "tool_abuse", "indirect_injection"
]

# ─── Data Loading ─────────────────────────────────────────────────────────
def load_jsonl(filepath):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, np.array(labels)

# ─── Model Definitions ───────────────────────────────────────────────────
def get_models():
    return {
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial", n_jobs=-1))
        ]),
        "linear_svm": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LinearSVC(max_iter=2000, C=1.0, multi_class="ovr"))
        ]),
        "random_forest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42))
        ]),
        "xgboost": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 1), sublinear_tf=True)),
            ("clf", GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.2, random_state=42))
        ]),
    }

# ─── Plotting ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, model_name, output_path):
    if not HAS_PLOT:
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=INTENT_NAMES, yticklabels=INTENT_NAMES)
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"    📊 Saved confusion matrix to {output_path}")

# ─── Training & Evaluation ───────────────────────────────────────────────
def train_and_evaluate(name, pipeline, X_train, y_train, X_test, y_test):
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING: {name.upper()}")
    print(f"{'='*60}")
    
    model_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Train
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"    Training time: {train_time:.2f}s")
    
    # Predict
    start = time.time()
    y_pred = pipeline.predict(X_test)
    inference_time = time.time() - start
    ms_per_sample = (inference_time / len(X_test)) * 1000
    print(f"    Inference time: {inference_time:.2f}s ({ms_per_sample:.3f} ms/sample)")
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    
    # Per-class report
    report = classification_report(y_test, y_pred, target_names=INTENT_NAMES, 
                                    output_dict=True, zero_division=0)
    
    results = {
        "model": name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "train_time_seconds": round(train_time, 2),
        "inference_ms_per_sample": round(ms_per_sample, 4),
        "per_class": {}
    }
    
    for intent in INTENT_NAMES:
        if intent in report:
            results["per_class"][intent] = {
                "precision": round(report[intent]["precision"], 4),
                "recall": round(report[intent]["recall"], 4),
                "f1": round(report[intent]["f1-score"], 4),
                "support": report[intent]["support"]
            }
    
    # Save metrics
    metrics_path = os.path.join(model_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"    💾 Saved metrics to {metrics_path}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, name, os.path.join(model_dir, "confusion_matrix.png"))
    
    # Save model pipeline
    joblib.dump(pipeline, os.path.join(model_dir, "pipeline.joblib"))
    print(f"    💾 Saved pipeline to {model_dir}/pipeline.joblib")
    
    print(f"\n    Results: Acc={acc:.4f} | F1-macro={f1_macro:.4f} | F1-weighted={f1_weighted:.4f}")
    print(f"    ✅ {name.upper()} COMPLETE")
    
    return results

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Paper 3 — Experiment 1: Classical ML Baselines")
    print("=" * 60)
    
    # Load data
    print("\n[*] Loading datasets...")
    X_train, y_train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    X_test, y_test = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
    print(f"    Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"    Classes: {len(set(y_train))} ({set(y_train)})")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Train all models
    models = get_models()
    all_results = []
    
    for name, pipeline in models.items():
        result = train_and_evaluate(name, pipeline, X_train, y_train, X_test, y_test)
        all_results.append(result)
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("  SUMMARY: ALL CLASSICAL ML BASELINES")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>12} {'ms/sample':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model']:<25} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} {r['f1_weighted']:>12.4f} {r['inference_ms_per_sample']:>10.4f}")
    
    # Save combined summary
    summary_path = os.path.join(OUTPUT_DIR, "all_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n💾 Combined summary saved to {summary_path}")
    print("\n🎉 ALL CLASSICAL ML BASELINES TRAINED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
