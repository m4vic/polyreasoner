"""
prepare_multidim_splits.py
==========================
Reads the master multiclass_clean.jsonl, augments every row with
technique / severity / surface labels (derived via the schema heuristics),
then writes stratified train/val/test splits for each of the 5 dimensions:

    output/advanced/clean/splits/intent/     (7 classes, uses existing label)
    output/advanced/clean/splits/technique/  (8 classes, heuristic)
    output/advanced/clean/splits/severity/   (3 classes, heuristic → 0-indexed)
    output/advanced/clean/splits/surface/    (4 classes, heuristic)
    output/advanced/clean/splits/binary/     (2 classes, derived from intent)

Run once before training any specialist model.
"""

import os
import json
import random
from collections import defaultdict

from schema import (
    INTENT_TO_ID,
    TECHNIQUE_TO_ID,
    SURFACE_TO_ID,
    BINARY_LABELS,
    detect_technique,
    calculate_severity,
    get_binary_label,
)

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

SRC_FILE = r"f:\AI-IN-THE-LOOP\dataset_pipeline\output\advanced\clean\multiclass_clean.jsonl"
OUT_BASE  = r"f:\AI-IN-THE-LOOP\dataset_pipeline\output\advanced\clean\splits"

DIMS = ["intent", "technique", "severity", "surface", "binary"]

random.seed(SEED)

# ─── Surface heuristic ────────────────────────────────────────────────────────
def detect_surface(text: str, intent: str) -> str:
    """Simple heuristic: benign → user_input; tool_abuse → tool_output;
    indirect_injection → document; else user_input / api based on length."""
    if intent == "benign":
        return "user_input"
    if intent == "tool_abuse":
        return "tool_output"
    if intent == "indirect_injection":
        return "document"
    # Longer, structured payloads tend to come via API
    if len(text) > 800:
        return "api"
    return "user_input"


# ─── Load & annotate ──────────────────────────────────────────────────────────
def load_and_annotate(src: str):
    rows = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text   = row["text"]
            intent = row["intent"]          # string like "benign"

            technique = detect_technique(text)
            severity  = calculate_severity(text, technique, intent)
            surface   = detect_surface(text, intent)
            binary    = get_binary_label(intent)

            rows.append({
                "text":      text,
                "intent":    INTENT_TO_ID[intent],
                "technique": TECHNIQUE_TO_ID[technique],
                # severity is 1-3, shift to 0-indexed for torch CrossEntropyLoss
                "severity":  severity - 1,
                "surface":   SURFACE_TO_ID[surface],
                "binary":    binary,
            })
    return rows


# ─── Stratified split ─────────────────────────────────────────────────────────
def stratified_split(rows, label_key):
    """Split rows into train/val/test stratified by label_key."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[label_key]].append(r)

    train, val, test = [], [], []
    for label, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_val  = max(1, int(n * VAL_RATIO))
        n_test = max(1, int(n * TEST_RATIO))
        test  += items[:n_test]
        val   += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


# ─── Write split ─────────────────────────────────────────────────────────────
def write_split(rows, dim, split_name, out_base):
    out_dir = os.path.join(out_base, dim)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split_name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            out = {"text": r["text"], "labels": r[dim]}
            f.write(json.dumps(out) + "\n")
    return path, len(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"📖  Loading & annotating: {SRC_FILE}")
    rows = load_and_annotate(SRC_FILE)
    print(f"    Total rows: {len(rows):,}")

    for dim in DIMS:
        print(f"\n🔀  Splitting dimension: '{dim}'")
        train, val, test = stratified_split(rows, dim)

        for split_name, split_rows in [("train", train), ("val", val), ("test", test)]:
            path, n = write_split(split_rows, dim, split_name, OUT_BASE)
            print(f"    [{split_name:5s}] {n:6,} rows → {path}")

    # ── Label distribution report ─────────────────────────────────────────────
    print("\n📊  Label distribution in full dataset:")
    for dim in DIMS:
        dist = defaultdict(int)
        for r in rows:
            dist[r[dim]] += 1
        label_counts = sorted(dist.items())
        total = len(rows)
        counts_str = "  ".join(
            f"{k}:{v} ({100*v/total:.1f}%)" for k, v in label_counts
        )
        print(f"    {dim:10s}: {counts_str}")

    print("\n✅  Multi-dimensional splits ready.")


if __name__ == "__main__":
    main()
