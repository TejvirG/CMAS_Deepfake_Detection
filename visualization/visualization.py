"""
visualization/visualization.py — generates plots from real results produced
by evaluate.py / experiments/run_all.py. All plots read from results/*.json
and results/*.csv; nothing here fabricates numbers — if a results file is
missing, the corresponding plot is skipped with a warning instead of being
faked.

Usage:
    python visualization/visualization.py --results_dir results/
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_roc_curve(results_json_path: str, out_path: str):
    if not os.path.exists(results_json_path):
        print(f"[skip] {results_json_path} not found; run evaluate.py first.")
        return
    with open(results_json_path) as f:
        data = json.load(f)
    if "roc_curve" not in data:
        print(f"[skip] No roc_curve in {results_json_path} (single-class split?).")
        return

    fpr = data["roc_curve"]["fpr"]
    tpr = data["roc_curve"]["tpr"]
    auc = data.get("roc_auc")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})" if auc else "ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def plot_confusion_matrix(results_json_path: str, out_path: str):
    if not os.path.exists(results_json_path):
        print(f"[skip] {results_json_path} not found; run evaluate.py first.")
        return
    with open(results_json_path) as f:
        data = json.load(f)
    cm = np.array(data["confusion_matrix"])

    plt.figure(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def plot_cmas_comparison(cmas_table_path: str, out_path: str):
    if not os.path.exists(cmas_table_path):
        print(f"[skip] {cmas_table_path} not found; run experiments/exp4_cmas_eval.py first.")
        return
    df = pd.read_csv(cmas_table_path)
    df = df.dropna(subset=["mean_cmas"])
    if df.empty:
        print(f"[skip] {cmas_table_path} has no valid CMAS rows to plot.")
        return

    plt.figure(figsize=(7, 5))
    bars = plt.bar(df["method"], df["mean_cmas"], yerr=df["std_cmas"], capsize=5, color="#4C72B0")
    plt.ylabel("Mean CMAS")
    plt.title("CMAS by Attribution Method")
    plt.ylim(0, 1.05)
    for bar, val in zip(bars, df["mean_cmas"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def plot_experiment_comparison(comparison_csv_path: str, out_path: str):
    if not os.path.exists(comparison_csv_path):
        print(f"[skip] {comparison_csv_path} not found; run experiments/run_all.py first.")
        return
    df = pd.read_csv(comparison_csv_path)
    metrics = ["accuracy", "roc_auc", "f1"]
    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, df[metric], width, label=metric)
    plt.xticks(x + width, df["method"])
    plt.ylabel("Score")
    plt.title("Experiment Comparison: Visual-only vs Audio-only vs Multimodal")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def plot_attribution_examples(results_json_path: str, out_path: str, n_examples: int = 6):
    if not os.path.exists(results_json_path):
        print(f"[skip] {results_json_path} not found; run evaluate.py first.")
        return
    with open(results_json_path) as f:
        data = json.load(f)
    samples = data.get("cmas_per_sample", [])
    if not samples:
        print(f"[skip] No cmas_per_sample entries in {results_json_path}.")
        return

    samples = samples[:n_examples]
    labels = [os.path.basename(s["video_path"]) for s in samples]
    visual_pct = [s["visual_contribution_pct"] for s in samples]
    audio_pct = [s["audio_contribution_pct"] for s in samples]

    x = np.arange(len(labels))
    plt.figure(figsize=(9, 5))
    plt.bar(x, visual_pct, label="Visual contribution %", color="#DD8452")
    plt.bar(x, audio_pct, bottom=visual_pct, label="Audio contribution %", color="#55A868")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Contribution %")
    plt.title("Modality Attribution — Sample Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/")
    args = parser.parse_args()

    r = args.results_dir
    plot_roc_curve(os.path.join(r, "results.json"), os.path.join(r, "roc_curve.png"))
    plot_confusion_matrix(os.path.join(r, "results.json"), os.path.join(r, "confusion_matrix.png"))
    plot_cmas_comparison(os.path.join(r, "cmas_table.csv"), os.path.join(r, "cmas_comparison.png"))
    plot_experiment_comparison(os.path.join(r, "experiment_comparison.csv"), os.path.join(r, "experiment_comparison.png"))
    plot_attribution_examples(os.path.join(r, "results.json"), os.path.join(r, "attribution_examples.png"))


if __name__ == "__main__":
    main()
