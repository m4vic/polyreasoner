import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Config
BASE_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models"
RESULTS_FILE = os.path.join(BASE_DIR, "benchmark_judge", "benchmark_results.json")
INTENT_METRICS_FILE = os.path.join(BASE_DIR, "specialist_intent", "final", "test_metrics.json")
OUTPUT_DIR = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\aeos_behave\paper3_thread_d\figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_latency_vs_accuracy():
    print("Generating Latency vs Accuracy chart...")
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    accuracies = []
    latencies = []

    for row in data:
        name_map = {
            "specialist_moe_only": "Specialist MoE",
            "hybrid_classical_plus_specialist": "Hybrid (ML + MoE)",
            "polyreasoner_full": "Full PolyReasoner",
            "llm_only": "LLM Only"
        }
        labels.append(name_map.get(row["config"], row["config"]))
        accuracies.append(row["accuracy"] * 100)  # Convert to percentage
        latencies.append(row["total_inference_seconds"])

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    width = 0.35

    color1 = '#1f77b4'
    rects1 = ax1.bar(x - width/2, accuracies, width, color=color1, label='Accuracy (%)')
    ax1.set_ylabel('Accuracy (%)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    color2 = '#d62728'
    # Log scale for latency because the difference is massive (6s vs 2300s)
    rects2 = ax2.bar(x + width/2, latencies, width, color=color2, label='Total Time (s)')
    ax2.set_ylabel('Total Inference Time (s) [Log Scale]', color=color2, fontsize=12)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
    
    plt.title('Judge Configuration: Accuracy vs Latency (200 samples)', fontsize=14, pad=20)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "latency_vs_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")

def generate_per_class_f1():
    print("Generating Per-Class F1 chart...")
    # Read the ensemble eval metrics which has the per-class breakdown
    eval_file = os.path.join(BASE_DIR, "ensemble_evaluation", "all_results_summary.json")
    if not os.path.exists(eval_file):
         print(f"Error: {eval_file} not found.")
         return
         
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    multiclass_data = None
    for strategy in data:
        if strategy.get("strategy") == "multiclass_baseline":
            multiclass_data = strategy.get("per_class", {})
            break

    if not multiclass_data:
        print("Error: multiclass_baseline not found in summary.")
        return

    classes = list(multiclass_data.keys())
    f1_scores = [multiclass_data[c]["f1"] for c in classes]

    # Sort by F1
    sorted_pairs = sorted(zip(classes, f1_scores), key=lambda x: x[1], reverse=False)
    classes, f1_scores = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Highlight indirect_injection in red to show it's the weak point
    colors = ['#d62728' if c == 'indirect_injection' else '#1f77b4' for c in classes]
    
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, f1_scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title('Specialist MoE: Per-Class F1 Score (Intent Dimension)', fontsize=14, pad=15)
    
    # Add values to bars
    for i, v in enumerate(f1_scores):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=10)

    ax.set_xlim(0, 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "per_class_f1.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_latency_vs_accuracy()
    generate_per_class_f1()
