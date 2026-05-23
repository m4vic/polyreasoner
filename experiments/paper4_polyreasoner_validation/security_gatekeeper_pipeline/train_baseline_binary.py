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
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb to prevent login prompts

import evaluate
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\baseline_binary"
DATA_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\output\advanced\clean\splits\binary"

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def create_hf_dataset(file_path):
    # Reads jsonl and converts it to a Hugging Face Dataset
    data = {"text": [], "labels": []}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            data["text"].append(row["text"])
            data["labels"].append(row["label"])
    return Dataset.from_dict(data)

def main():
    print("Loading datasets...")
    train_ds = create_hf_dataset(os.path.join(DATA_DIR, "train.jsonl"))
    val_ds   = create_hf_dataset(os.path.join(DATA_DIR, "val.jsonl"))
    test_ds  = create_hf_dataset(os.path.join(DATA_DIR, "test.jsonl"))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=512)

    print("Tokenizing datasets...")
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val   = val_ds.map(tokenize_function, batched=True)
    tokenized_test  = test_ds.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label={0: "benign", 1: "malicious"},
        label2id={"benign": 0, "malicious": 1}
    )

    # Initialize metrics
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        acc = accuracy.compute(predictions=predictions, references=labels)
        prec = precision.compute(predictions=predictions, references=labels, average="binary")
        rec = recall.compute(predictions=predictions, references=labels, average="binary")
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
        
        return {
            "accuracy": acc["accuracy"],
            "precision": prec["precision"],
            "recall": rec["recall"],
            "f1": f1["f1"],
        }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # keep disk usage low by retaining only the newest checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(), # Use mixed precision if on GPU
        logging_steps=100,
        report_to="none" # disable wandb integration
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

    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_test)
    print("Test Results:", test_results)

    print(f"Saving final model to {OUTPUT_DIR}/final...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    
    with open(os.path.join(OUTPUT_DIR, "final", "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4)

    print("✅ Binary baseline training completed.")

if __name__ == "__main__":
    main()
