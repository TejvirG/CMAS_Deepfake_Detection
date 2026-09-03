"""
experiments/run_all.py — runs Experiments 1-4 end to end and writes
results/experiment_comparison.csv (Accuracy/AUC/F1 across visual-only,
audio-only, multimodal) plus results/cmas_table.csv (Experiment 4).

Usage:
    python experiments/run_all.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

import experiments.exp1_visual_only as exp1
import experiments.exp2_audio_only as exp2
import experiments.exp3_multimodal as exp3
import experiments.exp4_cmas_eval as exp4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all experiments (useful for quick smoke runs).")
    parser.add_argument("--skip_cmas", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print("Running Experiment 1: Visual-only model")
    print("=" * 70)
    m1 = exp1.run(args.config, args.epochs)

    print("=" * 70)
    print("Running Experiment 2: Audio-only model")
    print("=" * 70)
    m2 = exp2.run(args.config, args.epochs)

    print("=" * 70)
    print("Running Experiment 3: Multimodal model")
    print("=" * 70)
    m3 = exp3.run(args.config, args.epochs)

    comparison_rows = [
        {"method": "Visual-only", "accuracy": m1["accuracy"], "roc_auc": m1["roc_auc"], "f1": m1["f1"]},
        {"method": "Audio-only", "accuracy": m2["accuracy"], "roc_auc": m2["roc_auc"], "f1": m2["f1"]},
        {"method": "Multimodal (CMAS)", "accuracy": m3["accuracy"], "roc_auc": m3["roc_auc"], "f1": m3["f1"]},
    ]
    out_path = os.path.join(cfg["paths"]["results_dir"], "experiment_comparison.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "accuracy", "roc_auc", "f1"])
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Wrote {out_path}")

    if not args.skip_cmas:
        print("=" * 70)
        print("Running Experiment 4: CMAS evaluation")
        print("=" * 70)
        ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model_multimodal.pt")
        exp4.run(args.config, ckpt_path)

    print("All experiments complete. See results/ for outputs.")


if __name__ == "__main__":
    main()
