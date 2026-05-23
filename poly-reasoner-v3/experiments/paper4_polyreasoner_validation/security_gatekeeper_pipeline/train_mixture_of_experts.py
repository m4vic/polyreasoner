import os
import json
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate
import warnings

os.environ["WANDB_DISABLED"] = "true"  # Disable wandb
warnings.filterwarnings("ignore")

# Schema definitions
from schema import INTENTS

# ─── Configuration ────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
BASE_OUTPUT_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\experts"
DATA_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\output\advanced\clean\splits\multiclass"

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def create_ovr_dataset(file_path, target_intent_id):
    """
    Creates a One-vs-Rest (OvR) binary dataset.
    If the row's label == target_intent_id, new label is 1 (Positive).
    Otherwise, new label is 0 (Negative).
    """
    data = {"text": [], "labels": []}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            # 1 if matches target expert, 0 otherwise
            binary_label = 1 if row["label"] == target_intent_id else 0
            data["text"].append(row["text"])
            data["labels"].append(binary_label)
    return Dataset.from_dict(data)

def train_expert(intent_name, intent_id, tokenizer, data_collator, metrics):
    print(f"\n{'='*50}")
    print(f"🚀 TRAINING EXPERT: {intent_name.upper()} (Class ID: {intent_id})")
    print(f"{'='*50}\n")
    
    expert_output_dir = os.path.join(BASE_OUTPUT_DIR, intent_name)
    os.makedirs(expert_output_dir, exist_ok=True)
    
    print("Loading specialized One-vs-Rest datasets...")
    train_ds = create_ovr_dataset(os.path.join(DATA_DIR, "train.jsonl"), intent_id)
    val_ds   = create_ovr_dataset(os.path.join(DATA_DIR, "val.jsonl"), intent_id)
    test_ds  = create_ovr_dataset(os.path.join(DATA_DIR, "test.jsonl"), intent_id)
    
    print("Tokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=512)

    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val   = val_ds.map(tokenize_function, batched=True)
    tokenized_test  = test_ds.map(tokenize_function, batched=True)
    
    print(f"Loading base model for {intent_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label={0: "other", 1: intent_name},
        label2id={"other": 0, intent_name: 1}
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        acc = metrics["accuracy"].compute(predictions=predictions, references=labels)
        prec = metrics["precision"].compute(predictions=predictions, references=labels, average="binary", zero_division=0)
        rec = metrics["recall"].compute(predictions=predictions, references=labels, average="binary", zero_division=0)
        f1 = metrics["f1"].compute(predictions=predictions, references=labels, average="binary")
        
        return {
            "accuracy": acc["accuracy"],
            "precision": prec["precision"],
            "recall": rec["recall"],
            "f1": f1["f1"],
        }

    training_args = TrainingArguments(
        output_dir=expert_output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Starting training for {intent_name}...")
    trainer.train()

    print(f"Evaluating {intent_name} on test set...")
    test_results = trainer.evaluate(tokenized_test)
    print(f"Test Results for {intent_name}:", test_results)

    final_dir = os.path.join(expert_output_dir, "final")
    print(f"Saving {intent_name} expert model to {final_dir}...")
    trainer.save_model(final_dir)
    
    with open(os.path.join(final_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4)
        
    print(f"✅ {intent_name.upper()} EXPERT TRAINING COMPLETED.\n")

def main():
    print("Initializing MoE Training Pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    metrics = {
        "accuracy": evaluate.load("accuracy"),
        "precision": evaluate.load("precision"),
        "recall": evaluate.load("recall"),
        "f1": evaluate.load("f1")
    }
    
    # Loop through all malicious intents (index 1 to 6)
    # 0 is 'benign', we don't train a "benign expert" since it's the absence of an attack
    malicious_intents = [(name, idx) for idx, name in enumerate(INTENTS) if name != "benign"]
    
    for intent_name, intent_id in malicious_intents:
        train_expert(intent_name, intent_id, tokenizer, data_collator, metrics)
        
    print("🎉 ALL 6 EXPERT MODELS SUCCESSFULLY TRAINED!")

if __name__ == "__main__":
    main()
