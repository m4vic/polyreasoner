import json
import os
import tempfile
from collections import defaultdict

from datasets import load_dataset
from huggingface_hub import HfApi

HF_ORG = "neuralchemy"
REPO_NAME = "prompt-injection-Threat-Matrix"
REPO_ID = f"{HF_ORG}/{REPO_NAME}"

ADVANCED_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\output\advanced"
SPLITS_DIR = os.path.join(ADVANCED_DIR, "clean", "splits")

BIN_DATA_DIR = os.path.join(SPLITS_DIR, "binary")
MULTI_DATA_DIR = os.path.join(SPLITS_DIR, "multiclass")
FULL_V2_PATH = os.path.join(ADVANCED_DIR, "full_v2.jsonl")

# Optional baseline model upload paths.
BIN_MODEL_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\baseline_binary\final"
MULTI_MODEL_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\baseline_multiclass\final"

MULTICLASS_UPLOAD_COLUMNS = [
    "text",
    "label",  # alias for intent_label, kept for trainer compatibility
    "binary_label",
    "intent",
    "intent_label",
    "technique",
    "technique_label",
    "severity",
    "surface",
    "surface_label",
    "source",
    "ambiguity",
]
REQUIRED_MULTI_COLUMNS = set(MULTICLASS_UPLOAD_COLUMNS)


def _load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _has_full_multiclass_schema(rows):
    if not rows:
        return False
    return REQUIRED_MULTI_COLUMNS.issubset(rows[0].keys())


def _build_full_lookup():
    if not os.path.exists(FULL_V2_PATH):
        raise FileNotFoundError(
            f"Missing full dataset at {FULL_V2_PATH}. "
            "Cannot enrich multiclass schema for upload."
        )

    lookup = defaultdict(list)
    with open(FULL_V2_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (
                row["text"],
                row["intent_label"],
                row["intent"],
                bool(row.get("ambiguity", False)),
            )
            lookup[key].append(row)
    return lookup


def _enrich_multiclass_rows(rows, lookup, split_name):
    enriched = []
    missing = 0

    for idx, row in enumerate(rows):
        if REQUIRED_MULTI_COLUMNS.issubset(row.keys()):
            # Keep label aligned even when source row already has full schema.
            row["label"] = row["intent_label"]
            enriched.append(row)
            continue

        key = (
            row["text"],
            row["label"],
            row.get("intent"),
            bool(row.get("ambiguity", False)),
        )
        candidates = lookup.get(key)
        if not candidates:
            missing += 1
            raise ValueError(
                f"Could not enrich multiclass row for split='{split_name}' index={idx}. "
                f"Lookup key={key}. Missing rows so far: {missing}"
            )

        full = candidates.pop(0)
        enriched.append(
            {
                "text": full["text"],
                "label": full["intent_label"],
                "binary_label": full["binary_label"],
                "intent": full["intent"],
                "intent_label": full["intent_label"],
                "technique": full["technique"],
                "technique_label": full["technique_label"],
                "severity": full["severity"],
                "surface": full["surface"],
                "surface_label": full["surface_label"],
                "source": full["source"],
                "ambiguity": bool(full.get("ambiguity", False)),
            }
        )

    return enriched


def _prepare_multiclass_files(verbose=True):
    split_to_path = {
        "train": os.path.join(MULTI_DATA_DIR, "train.jsonl"),
        "validation": os.path.join(MULTI_DATA_DIR, "val.jsonl"),
        "test": os.path.join(MULTI_DATA_DIR, "test.jsonl"),
    }

    split_rows = {split: _load_jsonl(path) for split, path in split_to_path.items()}
    needs_enrichment = any(
        not _has_full_multiclass_schema(rows) for rows in split_rows.values()
    )

    if not needs_enrichment:
        if verbose:
            print("Multiclass splits already include full schema columns.")
        return split_to_path, None

    if verbose:
        print("Multiclass splits are minimal. Enriching with full_v2 schema before push...")
    lookup = _build_full_lookup()
    temp_dir = tempfile.TemporaryDirectory()
    enriched_files = {}

    for split, rows in split_rows.items():
        enriched_rows = _enrich_multiclass_rows(rows, lookup, split)
        out_path = os.path.join(temp_dir.name, f"{split}.jsonl")
        _write_jsonl(out_path, enriched_rows)
        enriched_files[split] = out_path
        if verbose:
            print(f"  enriched {split:<10} {len(enriched_rows):>6,} rows")

    return enriched_files, temp_dir


def push_datasets():
    print(f"\n--- Pushing datasets to {REPO_ID} ---")

    print("Loading binary split data...")
    binary_ds = load_dataset(
        "json",
        data_files={
            "train": os.path.join(BIN_DATA_DIR, "train.jsonl"),
            "validation": os.path.join(BIN_DATA_DIR, "val.jsonl"),
            "test": os.path.join(BIN_DATA_DIR, "test.jsonl"),
        },
    )
    binary_ds.push_to_hub(
        REPO_ID,
        config_name="binary",
        commit_message="Upload binary split dataset",
    )

    print("Loading multiclass split data...")
    multi_files, temp_dir = _prepare_multiclass_files()
    try:
        multi_ds = load_dataset("json", data_files=multi_files)
        multi_ds.push_to_hub(
            REPO_ID,
            config_name="multiclass",
            commit_message="Upload multiclass split dataset with full schema",
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def push_models():
    print("\n--- Pushing baseline model folders into the dataset repo ---")
    api = HfApi()

    if os.path.exists(BIN_MODEL_DIR):
        print("Uploading binary baseline...")
        api.upload_folder(
            folder_path=BIN_MODEL_DIR,
            path_in_repo="baselines/distilbert-binary",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add binary baseline model",
        )
    else:
        print("Binary baseline folder not found; skipping.")

    if os.path.exists(MULTI_MODEL_DIR):
        print("Uploading multiclass baseline...")
        api.upload_folder(
            folder_path=MULTI_MODEL_DIR,
            path_in_repo="baselines/distilbert-multiclass",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add multiclass baseline model",
        )
    else:
        print("Multiclass baseline folder not found; skipping.")


def main():
    print("Initializing Hugging Face dataset upload...")
    api = HfApi()

    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=False)
        print(f"Verified repository: {REPO_ID}")
    except Exception as exc:
        print(f"Error accessing Hugging Face Hub: {exc}")
        return

    push_datasets()
    print(
        "\nDataset upload complete.\n"
        "Tip: if HF still shows a legacy merged/default config, run fix_hub_dataset.py once "
        "to recreate the repo cleanly."
    )


if __name__ == "__main__":
    main()
