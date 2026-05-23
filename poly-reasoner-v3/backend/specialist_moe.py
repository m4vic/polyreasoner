"""
specialist_moe.py
=================
Multi-Dimensional BERT Mixture-of-Experts for PolyReasoner.

Replaces the old ExpertEnsemble (6 binary per-type models) with the new
5-dimensional specialist system trained on the full Neuralchemy Threat Matrix:

    Specialist        Classes   What it answers
    ─────────────     ───────   ───────────────────────────────────────────
    binary            2         Is this prompt malicious at all?
    intent            7         WHAT is the attacker trying to achieve?
    technique         8         HOW is the attack constructed?
    severity          3         How dangerous / sophisticated is it?
    surface           4         WHERE does the attack originate?

Output is a structured ThreatVector dict ready for synthesis by an LLM judge.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── Label maps (must match schema.py) ───────────────────────────────────────
LABEL_MAPS = {
    "binary": {0: "benign", 1: "malicious"},
    "intent": {
        0: "benign",
        1: "direct_injection",
        2: "system_extraction",
        3: "role_hijack",
        4: "obfuscation",
        5: "tool_abuse",
        6: "indirect_injection",
    },
    "technique": {
        0: "none",
        1: "keyword_override",
        2: "persona_play",
        3: "encoding",
        4: "payload_splitting",
        5: "context_overflow",
        6: "few_shot_poisoning",
        7: "multilingual",
    },
    "severity": {0: "low", 1: "moderate", 2: "advanced"},
    "surface":  {0: "user_input", 1: "document", 2: "api", 3: "tool_output"},
}

MODELS_BASE = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models"

# Dimension order — binary first so we can short-circuit
DIMS = ["binary", "intent", "technique", "severity", "surface"]


class SpecialistMoE:
    """
    Loads 5 BERT specialists and runs them on a prompt to produce a
    structured ThreatVector covering all security dimensions.

    Usage:
        moe = SpecialistMoE()
        moe.load()                    # call once at startup
        tv  = moe.analyze("...")      # call per prompt
    """

    def __init__(self, models_base: str = MODELS_BASE):
        self.models_base = models_base
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.specialists: dict[str, AutoModelForSequenceClassification] = {}
        self._loaded = False

    # ─── Loading ──────────────────────────────────────────────────────────────
    def load(self):
        """Load tokenizer + all 5 specialist models into GPU/CPU memory."""
        if self._loaded:
            return

        print(f"[SpecialistMoE] Device: {self.device}")
        # All specialists share the same DistilBERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        for dim in DIMS:
            model_path = os.path.join(self.models_base, f"specialist_{dim}", "final")
            if not os.path.exists(model_path):
                print(f"[SpecialistMoE] Missing specialist: {dim} at {model_path}")
                continue
            print(f"[SpecialistMoE] Loading [{dim}] specialist...")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
            self.specialists[dim] = model

        loaded = list(self.specialists.keys())
        # Avoid emoji/non-ASCII to prevent Windows console encoding crashes.
        print(f"[SpecialistMoE] Loaded {len(loaded)}/5 specialists: {loaded}")
        self._loaded = True

    # ─── Inference ────────────────────────────────────────────────────────────
    def analyze(self, prompt: str, binary_threshold: float = 0.5) -> dict:
        """
        Run the prompt through all loaded specialists.

        Returns a ThreatVector dict, for example:
        {
            "binary":    {"label": "malicious", "confidence": 0.97},
            "is_malicious": True,
            "intent":    {"label": "role_hijack",      "confidence": 0.84, "top3": [...]},
            "technique": {"label": "persona_play",     "confidence": 0.91, "top3": [...]},
            "severity":  {"label": "moderate",         "confidence": 0.79},
            "surface":   {"label": "user_input",       "confidence": 0.93, "top3": [...]},
            "threat_score": 0.85,   # composite 0-1
        }
        """
        if not self._loaded:
            self.load()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        result: dict = {}

        with torch.no_grad():
            for dim, model in self.specialists.items():
                outputs = model(**inputs)
                probs   = F.softmax(outputs.logits, dim=-1)[0]  # shape [num_classes]
                label_map = LABEL_MAPS[dim]

                top_idx  = probs.argmax().item()
                top_conf = probs[top_idx].item()
                top_label = label_map.get(top_idx, str(top_idx))

                dim_result = {
                    "label":      top_label,
                    "confidence": round(top_conf, 4),
                }

                # Include top-3 breakdown for multi-class dims
                if len(label_map) > 2:
                    top3_indices = probs.topk(min(3, len(label_map))).indices.tolist()
                    dim_result["top3"] = [
                        {
                            "label":      label_map.get(i, str(i)),
                            "confidence": round(probs[i].item(), 4),
                        }
                        for i in top3_indices
                    ]

                result[dim] = dim_result

        # Convenience boolean
        binary_conf = result.get("binary", {}).get("confidence", 0.0)
        is_malicious = (
            result.get("binary", {}).get("label") == "malicious"
            and binary_conf >= binary_threshold
        )
        result["is_malicious"] = is_malicious

        # Composite threat score (0.0 - 1.0)
        result["threat_score"] = self._compute_threat_score(result)

        return result

    # ─── Threat Score ─────────────────────────────────────────────────────────
    @staticmethod
    def _compute_threat_score(tv: dict) -> float:
        """
        Lightweight composite score combining binary confidence + severity.
        Range: 0.0 (benign) → 1.0 (confirmed, advanced attack).
        """
        if not tv.get("is_malicious", False):
            return round(tv.get("binary", {}).get("confidence", 0.0) * 0.3, 4)

        binary_conf  = tv.get("binary",   {}).get("confidence", 0.5)
        intent_conf  = tv.get("intent",   {}).get("confidence", 0.5)
        severity_lbl = tv.get("severity", {}).get("label", "low")

        severity_weight = {"low": 0.4, "moderate": 0.7, "advanced": 1.0}.get(severity_lbl, 0.5)

        score = (binary_conf * 0.45) + (intent_conf * 0.25) + (severity_weight * 0.30)
        return round(min(score, 1.0), 4)

    # ─── Human-readable summary ───────────────────────────────────────────────
    @staticmethod
    def format_summary(tv: dict) -> str:
        """One-line human-readable threat summary for logging."""
        if not tv.get("is_malicious"):
            conf = tv.get("binary", {}).get("confidence", 0)
            return f"BENIGN  (binary_conf={conf:.2f}, threat_score={tv.get('threat_score', 0):.2f})"

        intent    = tv.get("intent",    {}).get("label", "?")
        technique = tv.get("technique", {}).get("label", "?")
        severity  = tv.get("severity",  {}).get("label", "?")
        surface   = tv.get("surface",   {}).get("label", "?")
        score     = tv.get("threat_score", 0)

        return (
            f"MALICIOUS  intent={intent}  technique={technique}  "
            f"severity={severity}  surface={surface}  threat_score={score:.2f}"
        )
