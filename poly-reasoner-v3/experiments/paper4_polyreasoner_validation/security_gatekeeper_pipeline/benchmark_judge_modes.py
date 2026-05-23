"""
Paper 3 — Experiment 3: Judge Configuration Benchmark
======================================================
Compares prompt-injection judge configurations on a stratified test sample:

  Config A: LLM-Only (no BERT signal)
  Config B: DistilBERT 5-D Specialist MoE only
  Config C: Full PolyReasoner (LLM judges + Specialist MoE)
  Config D: Hybrid Classical ML + Specialist MoE fallback

This script is aligned with the upgraded 5-dimensional specialist backend.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import warnings

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

POLYREASONER_DIR = r"f:\AI-IN-THE-LOOP\poly-reasoner-v3"
DATA_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\output\advanced\clean\splits\multiclass"
CLASSICAL_MODELS_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\classical"
OUTPUT_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\benchmark_judge"

INTENT_NAMES = [
    "benign",
    "direct_injection",
    "system_extraction",
    "role_hijack",
    "obfuscation",
    "tool_abuse",
    "indirect_injection",
]
INTENT_TO_INDEX = {name: idx for idx, name in enumerate(INTENT_NAMES)}

DEFAULT_SAMPLE_SIZE = 200
HYBRID_CONF_THRESHOLD = 0.65


def load_jsonl(filepath, max_samples=None):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            row = json.loads(line.strip())
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, np.array(labels)


def stratified_sample(texts, labels, n_samples):
    np.random.seed(42)
    unique_labels = np.unique(labels)
    per_class = max(1, n_samples // len(unique_labels))
    sampled_texts, sampled_labels = [], []

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        chosen = np.random.choice(indices, size=min(per_class, len(indices)), replace=False)
        for idx in chosen:
            sampled_texts.append(texts[idx])
            sampled_labels.append(labels[idx])

    return sampled_texts, np.array(sampled_labels)


def _tv_to_intent_index(threat_vector):
    if not isinstance(threat_vector, dict):
        return 0
    if not threat_vector.get("is_malicious", False):
        return 0
    intent_label = threat_vector.get("intent", {}).get("label", "benign")
    return INTENT_TO_INDEX.get(intent_label, 0)


def load_specialist_moe():
    if POLYREASONER_DIR not in sys.path:
        sys.path.insert(0, POLYREASONER_DIR)
    from backend.specialist_moe import SpecialistMoE

    moe = SpecialistMoE()
    moe.load()
    return moe


def load_classical_pipeline(preferred="logistic_regression"):
    candidates = [preferred, "linear_svm", "random_forest", "xgboost"]
    checked = []
    for name in candidates:
        path = os.path.join(CLASSICAL_MODELS_DIR, name, "pipeline.joblib")
        checked.append(path)
        if os.path.exists(path):
            print(f"    Loaded classical pipeline: {name}")
            return name, joblib.load(path)
    raise FileNotFoundError(
        "No classical pipeline found. Checked:\n" + "\n".join(checked)
    )


def run_bert_only(texts, moe):
    print("\n[Config B] Specialist MoE-Only: Running 5-D DistilBERT specialists...")
    predictions = []
    start = time.time()

    for i, text in enumerate(texts):
        threat_vector = moe.analyze(text)
        predictions.append(_tv_to_intent_index(threat_vector))
        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(texts)}...")

    elapsed = time.time() - start
    print(f"    Specialist-only inference: {elapsed:.2f}s ({1000*elapsed/len(texts):.1f} ms/sample)")
    return np.array(predictions), elapsed


def run_hybrid_ml_bert(texts, classical_pipeline, moe, confidence_threshold=HYBRID_CONF_THRESHOLD):
    print(
        f"\n[Config D] Hybrid ML+MoE: Classical first pass, MoE fallback if confidence < {confidence_threshold:.2f}..."
    )
    if not hasattr(classical_pipeline, "predict_proba"):
        raise ValueError("Hybrid mode requires a classical pipeline with predict_proba (use logistic_regression).")

    start = time.time()
    ml_preds = classical_pipeline.predict(texts)
    ml_probs = classical_pipeline.predict_proba(texts)

    predictions = []
    fallback_count = 0

    for i, text in enumerate(texts):
        best_prob = float(np.max(ml_probs[i]))
        if best_prob >= confidence_threshold:
            predictions.append(int(ml_preds[i]))
        else:
            fallback_count += 1
            threat_vector = moe.analyze(text)
            predictions.append(_tv_to_intent_index(threat_vector))

        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(texts)}...")

    elapsed = time.time() - start
    print(f"    Hybrid inference: {elapsed:.2f}s ({1000*elapsed/len(texts):.1f} ms/sample)")
    print(f"    MoE fallback used: {fallback_count}/{len(texts)} ({100*fallback_count/len(texts):.1f}%)")
    return np.array(predictions), elapsed, fallback_count


async def run_llm_only(texts, labels):
    import importlib.util

    spec = importlib.util.spec_from_file_location("backend", os.path.join(POLYREASONER_DIR, "backend", "__init__.py"))
    backend_mod = importlib.util.module_from_spec(spec)

    spec_base = importlib.util.spec_from_file_location("backend.base", os.path.join(POLYREASONER_DIR, "backend", "base.py"))
    base_mod = importlib.util.module_from_spec(spec_base)
    sys.modules["backend.base"] = base_mod
    spec_base.loader.exec_module(base_mod)

    spec_ollama = importlib.util.spec_from_file_location("backend.ollama_backend", os.path.join(POLYREASONER_DIR, "backend", "ollama_backend.py"))
    ollama_mod = importlib.util.module_from_spec(spec_ollama)
    sys.modules["backend.ollama_backend"] = ollama_mod
    spec_ollama.loader.exec_module(ollama_mod)

    spec_api = importlib.util.spec_from_file_location("backend.api_backend", os.path.join(POLYREASONER_DIR, "backend", "api_backend.py"))
    api_mod = importlib.util.module_from_spec(spec_api)
    sys.modules["backend.api_backend"] = api_mod
    spec_api.loader.exec_module(api_mod)

    sys.modules["backend"] = backend_mod
    spec.loader.exec_module(backend_mod)

    spec_jp = importlib.util.spec_from_file_location("judge_prompts", os.path.join(POLYREASONER_DIR, "prompts", "judge_prompts.py"))
    jp_mod = importlib.util.module_from_spec(spec_jp)
    spec_jp.loader.exec_module(jp_mod)

    safety_judge_prompt = jp_mod.SAFETY_JUDGE_PROMPT
    jailbreak_judge_prompt = jp_mod.JAILBREAK_JUDGE_PROMPT
    synthesizer_judge_prompt = jp_mod.SYNTHESIZER_JUDGE_PROMPT

    print("\n[Config A] LLM-Only: Running LLM judges (slow path)...")
    backend = backend_mod.BackendFactory.create()

    predictions = []
    start = time.time()

    for i, text in enumerate(texts):
        dummy_response = "I cannot help with that request."
        try:
            safety_prompt = f"Attack Prompt:\n{text}\n\nTarget Response:\n{dummy_response}"
            jailbreak_prompt = f"Attack Prompt:\n{text}\n\nTarget Response:\n{dummy_response}"

            safety_result = await backend.complete(safety_prompt, system_prompt=safety_judge_prompt)
            jailbreak_result = await backend.complete(jailbreak_prompt, system_prompt=jailbreak_judge_prompt)

            synth_prompt = (
                f"Original Attack Prompt:\n{text}\n\n"
                f"Target Response:\n{dummy_response}\n\n"
                f"--- LLM Safety Judge Analysis ---\n{safety_result}\n\n"
                f"--- LLM Jailbreak Judge Analysis ---\n{jailbreak_result}\n"
            )

            verdict = await backend.complete_json(synth_prompt, system_prompt=synthesizer_judge_prompt)
            if isinstance(verdict, dict):
                intent = verdict.get("primary_intent", "benign")
                predictions.append(INTENT_TO_INDEX.get(intent, 0))
            else:
                predictions.append(0)
        except Exception as e:
            print(f"    [WARN] Error on sample {i}: {e}")
            predictions.append(0)

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(texts)}...")

    elapsed = time.time() - start
    print(f"    LLM-only inference: {elapsed:.2f}s ({1000*elapsed/len(texts):.1f} ms/sample)")
    return np.array(predictions), elapsed


async def run_full_ensemble(texts, labels):
    old_cwd = os.getcwd()
    if POLYREASONER_DIR not in sys.path:
        sys.path.insert(0, POLYREASONER_DIR)
    os.chdir(POLYREASONER_DIR)

    if "prompts" in sys.modules:
        del sys.modules["prompts"]
    if "prompts.judge_prompts" in sys.modules:
        del sys.modules["prompts.judge_prompts"]
    if "modes" in sys.modules:
        del sys.modules["modes"]
    if "modes.judge_mode" in sys.modules:
        del sys.modules["modes.judge_mode"]

    from modes.judge_mode import JudgeMode
    os.chdir(old_cwd)

    print("\n[Config C] Full PolyReasoner: Running LLM + 5-D Specialist MoE...")
    judge = JudgeMode()
    predictions = []
    start = time.time()

    for i, text in enumerate(texts):
        dummy_response = "I cannot help with that request."
        try:
            verdict = await judge.run(text, dummy_response)
            if isinstance(verdict, dict):
                intent = verdict.get("primary_intent", "benign")
                predictions.append(INTENT_TO_INDEX.get(intent, 0))
            else:
                predictions.append(0)
        except Exception as e:
            print(f"    [WARN] Error on sample {i}: {e}")
            predictions.append(0)

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(texts)}...")

    elapsed = time.time() - start
    print(f"    Full ensemble inference: {elapsed:.2f}s ({1000*elapsed/len(texts):.1f} ms/sample)")
    return np.array(predictions), elapsed


def evaluate(y_true, y_pred, config_name):
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=INTENT_NAMES, output_dict=True, zero_division=0)

    result = {
        "config": config_name,
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class": {},
    }

    for intent in INTENT_NAMES:
        if intent in report:
            result["per_class"][intent] = {
                "precision": round(report[intent]["precision"], 4),
                "recall": round(report[intent]["recall"], 4),
                "f1": round(report[intent]["f1-score"], 4),
            }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark prompt-injection judge configurations.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help=f"Stratified test sample size (default: {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--hybrid-threshold", type=float, default=HYBRID_CONF_THRESHOLD, help=f"Confidence threshold for ML->MoE fallback (default: {HYBRID_CONF_THRESHOLD})")
    parser.add_argument("--skip-llm-only", action="store_true", help="Skip Config A (LLM-only)")
    parser.add_argument("--skip-full", action="store_true", help="Skip Config C (full PolyReasoner)")
    return parser.parse_args()


async def main(args):
    print("=" * 60)
    print("  Paper 3 — Experiment 3: Judge Configuration Benchmark")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[*] Loading test data (sampling {args.sample_size} stratified)...")
    all_texts, all_labels = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
    texts, labels = stratified_sample(all_texts, all_labels, args.sample_size)
    print(f"    Sampled {len(texts)} from {len(all_texts)} total test samples")

    all_results = []

    try:
        moe = load_specialist_moe()
        preds_bert, time_bert = run_bert_only(texts, moe)
        result_bert = evaluate(labels, preds_bert, "specialist_moe_only")
        result_bert["total_inference_seconds"] = round(time_bert, 2)
        all_results.append(result_bert)
    except Exception as e:
        print(f"    [WARN] Specialist MoE-only skipped: {e}")
        moe = None

    try:
        classical_name, classical_pipeline = load_classical_pipeline(preferred="logistic_regression")
        preds_hybrid, time_hybrid, fallback_count = run_hybrid_ml_bert(
            texts,
            classical_pipeline,
            moe if moe is not None else load_specialist_moe(),
            confidence_threshold=args.hybrid_threshold,
        )
        result_hybrid = evaluate(labels, preds_hybrid, "hybrid_classical_plus_specialist")
        result_hybrid["total_inference_seconds"] = round(time_hybrid, 2)
        result_hybrid["classical_pipeline"] = classical_name
        result_hybrid["hybrid_confidence_threshold"] = args.hybrid_threshold
        result_hybrid["moe_fallback_count"] = int(fallback_count)
        result_hybrid["moe_fallback_rate"] = round(fallback_count / len(texts), 4)
        all_results.append(result_hybrid)
    except Exception as e:
        print(f"    [WARN] Hybrid Classical+MoE skipped: {e}")

    if args.skip_llm_only:
        print("    [SKIP] Config A (LLM-only) by flag.")
    else:
        try:
            preds_llm, time_llm = await run_llm_only(texts, labels)
            result_llm = evaluate(labels, preds_llm, "llm_only")
            result_llm["total_inference_seconds"] = round(time_llm, 2)
            all_results.append(result_llm)
        except Exception as e:
            print(f"    [WARN] LLM-only skipped (backend unavailable?): {e}")

    if args.skip_full:
        print("    [SKIP] Config C (full PolyReasoner) by flag.")
    else:
        try:
            preds_full, time_full = await run_full_ensemble(texts, labels)
            result_full = evaluate(labels, preds_full, "polyreasoner_full")
            result_full["total_inference_seconds"] = round(time_full, 2)
            all_results.append(result_full)
        except Exception as e:
            print(f"    [WARN] Full PolyReasoner skipped: {e}")

    print("\n" + "=" * 72)
    print("  SUMMARY: JUDGE CONFIGURATION COMPARISON")
    print("=" * 72)
    print(f"{'Config':<36} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>12} {'Time(s)':>10}")
    print("-" * 82)
    for result in all_results:
        total_time = result.get("total_inference_seconds", "N/A")
        print(
            f"{result['config']:<36} {result['accuracy']:>10.4f} "
            f"{result['f1_macro']:>10.4f} {result['f1_weighted']:>12.4f} {total_time:>10}"
        )

    summary_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nResults saved to {summary_path}")
    print("\nBENCHMARK COMPLETE!")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
