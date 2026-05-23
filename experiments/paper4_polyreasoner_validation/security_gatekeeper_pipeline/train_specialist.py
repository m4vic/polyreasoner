"""
train_specialist.py
====================
Trains a DistilBERT specialist for ONE of the 5 security dimensions.
Uses Early Stopping — trains until validation F1 stops improving,
then automatically restores and saves the best checkpoint.

Usage:
    python train_specialist.py --dim intent
    python train_specialist.py --dim intent --max-epochs 20 --patience 4
    python train_specialist.py --dim binary --max-epochs 10 --patience 2

Defaults:
    --max-epochs 15   (ceiling — will stop earlier via early stopping)
    --patience   3    (epochs with no improvement before stopping)

Each run saves the BEST model (by val F1) to:
    models/specialist_<dim>/final/

And writes test_metrics.json alongside the model weights.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

os.environ["WANDB_DISABLED"] = "true"

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import evaluate
import warnings
warnings.filterwarnings("ignore")

# ─── Schema ──────────────────────────────────────────────────────────────────
from schema import (
    INTENTS,
    TECHNIQUES,
    SURFACES,
)

# Dimension → (num_classes, id2label)
DIM_CONFIG = {
    "intent": {
        "num_labels": len(INTENTS),
        "id2label":   {i: n for i, n in enumerate(INTENTS)},
        "label2id":   {n: i for i, n in enumerate(INTENTS)},
        "metric":     "f1_weighted",
        "average":    "weighted",
    },
    "technique": {
        "num_labels": len(TECHNIQUES),
        "id2label":   {i: n for i, n in enumerate(TECHNIQUES)},
        "label2id":   {n: i for i, n in enumerate(TECHNIQUES)},
        "metric":     "f1_weighted",
        "average":    "weighted",
    },
    "severity": {
        # Severity stored as 0-indexed (0=low, 1=moderate, 2=advanced)
        "num_labels": 3,
        "id2label":   {0: "low", 1: "moderate", 2: "advanced"},
        "label2id":   {"low": 0, "moderate": 1, "advanced": 2},
        "metric":     "f1_macro",
        "average":    "macro",
    },
    "surface": {
        "num_labels": len(SURFACES),
        "id2label":   {i: n for i, n in enumerate(SURFACES)},
        "label2id":   {n: i for i, n in enumerate(SURFACES)},
        "metric":     "f1_weighted",
        "average":    "weighted",
    },
    "binary": {
        "num_labels": 2,
        "id2label":   {0: "benign", 1: "malicious"},
        "label2id":   {"benign": 0, "malicious": 1},
        "metric":     "f1_weighted",
        "average":    "weighted",
    },
}

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
DATA_BASE   = r"f:\AI-IN-THE-LOOP\dataset_pipeline\output\advanced\clean\splits"
MODELS_BASE = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models"

BATCH_SIZE    = 32
MAX_EPOCHS    = 15     # hard ceiling — early stopping will kick in before this
PATIENCE      = 3      # epochs with no val-F1 improvement before stopping
LEARNING_RATE = 2e-5

device = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Dataset loading ──────────────────────────────────────────────────────────
def load_split(dim: str, split: str) -> Dataset:
    path = os.path.join(DATA_BASE, dim, f"{split}.jsonl")
    data = {"text": [], "labels": []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            data["text"].append(row["text"])
            data["labels"].append(row["labels"])
    return Dataset.from_dict(data)


# ─── Main training function ───────────────────────────────────────────────────
def train_specialist(dim: str, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE, balance_weights: bool = False):
    cfg = DIM_CONFIG[dim]
    output_dir = os.path.join(MODELS_BASE, f"specialist_{dim}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🧠 Training BERT Specialist: [{dim.upper()}]")
    print(f"     Classes    : {cfg['num_labels']}")
    print(f"     Device     : {device}")
    print(f"     Max epochs : {max_epochs}")
    print(f"     Patience   : {patience} epochs")
    print(f"     Output     : {output_dir}")
    print(f"     Weights    : {'Balanced (Inverse Freq)' if balance_weights else 'Uniform'}")
    print(f"{'='*60}\n")

    # Load splits
    print("📂 Loading datasets...")
    train_ds = load_split(dim, "train")
    val_ds   = load_split(dim, "val")
    test_ds  = load_split(dim, "test")
    print(f"   train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    # Tokenize
    print("🔤 Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=512)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds   = val_ds.map(tokenize, batched=True)
    test_ds  = test_ds.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    print(f"🤖 Loading model ({cfg['num_labels']} classes)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=cfg["num_labels"],
        id2label=cfg["id2label"],
        label2id=cfg["label2id"],
    )

    # Metrics
    accuracy  = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    average   = cfg["average"]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc         = accuracy.compute(predictions=preds, references=labels)
        f1_weighted = f1_metric.compute(predictions=preds, references=labels, average="weighted")
        f1_macro    = f1_metric.compute(predictions=preds, references=labels, average="macro")
        return {
            "accuracy":    acc["accuracy"],
            "f1_weighted": f1_weighted["f1"],
            "f1_macro":    f1_macro["f1"],
        }

    # Training args — early stopping driven
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=max_epochs,      # ceiling only — early stopping fires first
        weight_decay=0.01,
        warmup_ratio=0.06,                # 6% warmup to stabilise early epochs
        lr_scheduler_type="cosine",       # cosine decay for smoother late-training
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,               # keep best + 1 previous so we can always restore
        load_best_model_at_end=True,      # ← restores best checkpoint when training ends
        metric_for_best_model=cfg["metric"],
        greater_is_better=True,           # F1 — higher is better
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    if balance_weights:
        print("⚖️ Calculating balanced class weights...")
        train_labels = train_ds["label"] if "label" in train_ds.column_names else train_ds["labels"]
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        weights = total_samples / (len(class_counts) * class_counts)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"   Weights: {np.round(weights, 3)}")

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss
        
        TrainerClass = CustomTrainer
    else:
        TrainerClass = Trainer

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=0.0005,  # must improve by at least 0.05%
            )
        ],
    )

    print("🚀 Training (early stopping enabled)...")
    train_result = trainer.train()

    # Report which epoch was actually the best
    best_metric = trainer.state.best_metric
    best_epoch  = getattr(trainer.state, 'best_model_checkpoint', 'unknown')
    epochs_run  = int(trainer.state.epoch)
    print(f"\n⏹  Stopped at epoch {epochs_run}/{max_epochs}")
    print(f"   Best val/{cfg['metric']} : {best_metric:.4f}")
    print(f"   Best checkpoint       : {best_epoch}")
    if epochs_run < max_epochs:
        print(f"   ✅ Early stopping fired after {patience} epochs without improvement")
    else:
        print(f"   ⚠  Hit max_epochs ceiling ({max_epochs}) — consider increasing")

    print("\n📊 Evaluating BEST model on test set...")
    test_results = trainer.evaluate(test_ds)
    print("   Test Results:", test_results)

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    metrics_path = os.path.join(final_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "dimension":   dim,
            "epochs_run":  epochs_run,
            "max_epochs":  max_epochs,
            "patience":    patience,
            "best_val_metric": best_metric,
            **test_results
        }, f, indent=4)

    print(f"\n✅ [{dim}] BEST specialist saved → {final_dir}")
    print(f"   Metrics  → {metrics_path}")
    return test_results


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BERT specialist for one security dimension with early stopping."
    )
    parser.add_argument(
        "--dim",
        choices=list(DIM_CONFIG.keys()),
        required=True,
        help="Dimension to train: intent | technique | severity | surface | binary",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Maximum training epochs (default: {MAX_EPOCHS}). Early stopping will fire before this.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help=f"Early stopping patience — epochs with no improvement before stopping (default: {PATIENCE}).",
    )
    parser.add_argument(
        "--balance-weights",
        action="store_true",
        help="Use inverse-frequency class weights in CrossEntropyLoss to heavily penalize missing rare classes (fixes indirect_injection).",
    )
    args = parser.parse_args()
    train_specialist(args.dim, max_epochs=args.max_epochs, patience=args.patience, balance_weights=args.balance_weights)
