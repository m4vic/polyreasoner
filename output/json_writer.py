"""
ASRT-compatible structured JSON output writer.
Validates, writes, and appends judge verdicts for batch ASRT pipeline use.
"""
from __future__ import annotations

import json
import os
from typing import Optional
from jsonschema import validate, ValidationError


# ─── ASRT Verdict Schema ─────────────────────────────────────────────────────

VERDICT_SCHEMA = {
    "type": "object",
    "required": ["verdict", "overall_score", "confidence", "meta"],
    "properties": {
        "verdict": {"type": "string", "enum": ["safe", "unsafe", "borderline"]},
        "overall_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "criteria": {"type": "object"},
        "judge_agreement": {"type": "number"},
        "dissent": {"type": "array", "items": {"type": "string"}},
        "meta": {
            "type": "object",
            "required": ["attack_id", "timestamp"],
            "properties": {
                "attack_id": {"type": "string"},
                "timestamp": {"type": "string"},
                "judges_used": {"type": "array", "items": {"type": "string"}},
                "model_under_test": {"type": "string"},
            }
        }
    }
}


class JsonWriter:
    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path

    def validate(self, data: dict) -> bool:
        """Validate against ASRT schema. Returns True if valid."""
        try:
            validate(instance=data, schema=VERDICT_SCHEMA)
            return True
        except ValidationError as e:
            print(f"  [JsonWriter] Schema validation error: {e.message}")
            return False

    def pretty_print(self, data: dict):
        """Print formatted JSON to terminal."""
        print(json.dumps(data, indent=2))

    def write(self, data: dict) -> bool:
        """Validate then write a single verdict to the output file (JSONL format)."""
        if not self.validate(data):
            print("  [JsonWriter] Skipping invalid verdict — will not write broken JSON to ASRT.")
            return False
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        return True

    def write_batch(self, verdicts: list[dict]) -> tuple[int, int]:
        """Write a batch of verdicts. Returns (written, skipped) counts."""
        written, skipped = 0, 0
        for v in verdicts:
            if self.write(v):
                written += 1
            else:
                skipped += 1
        return written, skipped
