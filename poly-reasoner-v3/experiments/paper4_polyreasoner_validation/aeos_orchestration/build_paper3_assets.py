"""
Build Paper 3 figure assets from AEOS behave results.

Outputs to:
  aitl-paper/paper/figures/
    - fig2_full_comparison.png
    - fig3_representative_trajectories.png
    - fig4_mathprompt_ablation_summary.png
    - fig5_tri_loop_example.png
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def _paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    results = root / "results"
    figures = root.parents[2] / "paper" / "figures"
    thread_a = root / "paper3_thread_a"
    return {"root": root, "results": results, "figures": figures, "thread_a": thread_a}


def _copy_thread_a_fig2(paths: dict[str, Path]) -> None:
    src = paths["thread_a"] / "fig2_full_comparison.png"
    dst = paths["figures"] / "fig2_full_comparison.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[OK] Copied {dst}")


def _copy_tri_example(paths: dict[str, Path]) -> None:
    src = paths["results"] / "tabular2" / "exp3_plot_tri_tabular2_run1_20260509_234632.png"
    dst = paths["figures"] / "fig5_tri_loop_example.png"
    shutil.copy2(src, dst)
    print(f"[OK] Copied {dst}")


def _build_representative_trajectories(paths: dict[str, Path]) -> None:
    panels = [
        (
            "Tabular2 — Single",
            "llama3.1:8b",
            paths["results"] / "tabular2" / "v2_plot_ollama_run2_20260503_123549.png",
        ),
        (
            "Tabular2 — Dual",
            "qwen14b → deepseek16b",
            paths["results"]
            / "tabular2"
            / "exp2_plot_qwen2.5-coder-14b_deepseek-coder-v2-16b_run2_20260505_162435.png",
        ),
        (
            "Tabular2 — Tri-Agent",
            "qwen3.5 + 3b + 7b",
            paths["results"] / "tabular2" / "exp3_plot_tri_tabular2_run1_20260509_234632.png",
        ),
        (
            "Text — Single",
            "llama3.1:8b",
            paths["results"] / "text" / "v2_plot_ollama_run3_20260507_211134.png",
        ),
        (
            "Text — Dual",
            "qwen14b → deepseek16b",
            paths["results"]
            / "text"
            / "exp2_plot_qwen2.5-coder-14b_deepseek-coder-v2-16b_run2_20260509_052012.png",
        ),
        (
            "Vision — Single",
            "qwen3.5:9b",
            paths["results"] / "vision" / "v2_plot_ollama_run1_20260506_211858.png",
        ),
        (
            "Vision — Dual",
            "qwen3.5:9b → qwen7b",
            paths["results"] / "vision" / "exp2_plot_qwen3.5-9b_qwen2.5-coder-7b_run3_20260507_032643.png",
        ),
    ]

    missing = [str(path) for _, _, path in panels if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing representative plot file(s):\n" + "\n".join(missing))

    fig, axes = plt.subplots(4, 2, figsize=(13, 18))
    for index, ax in enumerate(axes.flatten()):
        if index >= len(panels):
            ax.axis("off")
            continue
        title, subtitle, image_path = panels[index]
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=8)

    fig.suptitle("Representative AEOS Trajectories with Model Labels", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_path = paths["figures"] / "fig3_representative_trajectories.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


def _latest_math_summary(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("math_batch_summary_all_*.json"))
    if not candidates:
        raise FileNotFoundError("No math_batch_summary_all_*.json file found in results directory.")
    return candidates[-1]


def _build_math_ablation_summary(paths: dict[str, Path]) -> None:
    summary_path = _latest_math_summary(paths["results"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No results found in {summary_path.name}")

    labels = [condition["label"] for condition in payload.get("conditions", [])]
    datasets = payload.get("datasets", [])
    if not labels:
        labels = sorted({row["label"] for row in results})
    if not datasets:
        datasets = sorted({row["dataset"] for row in results})

    acc_matrix = np.full((len(datasets), len(labels)), np.nan)
    for i, dataset in enumerate(datasets):
        for j, label in enumerate(labels):
            values = [
                row.get("best_accuracy")
                for row in results
                if row.get("dataset") == dataset
                and row.get("label") == label
                and isinstance(row.get("best_accuracy"), (int, float))
            ]
            if values:
                acc_matrix[i, j] = float(np.mean(values))

    stop_rates = []
    mean_iters = []
    for label in labels:
        rows = [row for row in results if row.get("label") == label]
        stop_rows = [
            row
            for row in rows
            if isinstance(row.get("stop_reason"), str) and "Reviewer STOP" in row["stop_reason"]
        ]
        stop_rates.append(100.0 * len(stop_rows) / len(rows) if rows else 0.0)
        iters = [row["total_iterations"] for row in rows if isinstance(row.get("total_iterations"), (int, float))]
        mean_iters.append(float(np.mean(iters)) if iters else 0.0)

    figure = plt.figure(figsize=(15, 6))

    ax1 = figure.add_subplot(1, 2, 1)
    image = ax1.imshow(
        acc_matrix,
        cmap="viridis",
        aspect="auto",
        vmin=np.nanmin(acc_matrix),
        vmax=np.nanmax(acc_matrix),
    )
    ax1.set_title("Mean Best Accuracy (N=36)", fontsize=12)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(
        [label.replace("-MATH", "").replace("DUAL-", "D:").replace("SINGLE-", "S:") for label in labels],
        rotation=35,
        ha="right",
        fontsize=9,
    )
    ax1.set_yticks(range(len(datasets)))
    ax1.set_yticklabels([dataset.capitalize() for dataset in datasets], fontsize=10)
    for i in range(len(datasets)):
        for j in range(len(labels)):
            if not np.isnan(acc_matrix[i, j]):
                ax1.text(j, i, f"{acc_matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    colorbar = figure.colorbar(image, ax=ax1, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("Accuracy", rotation=90)

    ax2 = figure.add_subplot(1, 2, 2)
    x_axis = np.arange(len(labels))
    bars = ax2.bar(x_axis, stop_rates, color="#d35400", alpha=0.85, label="Reviewer STOP rate (%)")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("STOP rate (%)")
    ax2.set_xticks(x_axis)
    ax2.set_xticklabels(
        [label.replace("-MATH", "").replace("DUAL-", "D:").replace("SINGLE-", "S:") for label in labels],
        rotation=35,
        ha="right",
        fontsize=9,
    )
    ax2.set_title("Autonomous Halting Behavior", fontsize=12)
    for bar, value in zip(bars, stop_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    ax2b = ax2.twinx()
    ax2b.plot(x_axis, mean_iters, color="#2c3e50", marker="o", linewidth=2, label="Mean iterations")
    ax2b.set_ylabel("Mean iterations")
    ax2b.set_ylim(0, max(mean_iters) * 1.35 if mean_iters else 10)
    for xi, value in zip(x_axis, mean_iters):
        ax2b.text(xi, value + 0.6, f"{value:.1f}", color="#2c3e50", ha="center", fontsize=8)

    left_handles, left_labels = ax2.get_legend_handles_labels()
    right_handles, right_labels = ax2b.get_legend_handles_labels()
    ax2.legend(left_handles + right_handles, left_labels + right_labels, loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    figure.suptitle("Math-Prompt Ablation Summary Across Datasets and Architectures", fontsize=14, y=1.02)
    figure.tight_layout()

    out_path = paths["figures"] / "fig4_mathprompt_ablation_summary.png"
    figure.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"[OK] Wrote {out_path} (source: {summary_path.name})")


def main() -> None:
    paths = _paths()
    paths["figures"].mkdir(parents=True, exist_ok=True)
    _copy_thread_a_fig2(paths)
    _copy_tri_example(paths)
    _build_representative_trajectories(paths)
    _build_math_ablation_summary(paths)
    print("[DONE] Paper 3 figure assets refreshed.")


if __name__ == "__main__":
    main()
