"""
experiments/exp1_visual_only.py — trains and evaluates the visual-only
ablation of the detector (Experiment 1). Real metrics only; no fabricated
numbers. Writes results/exp1_visual_only.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from evaluate import main as evaluate_main  # noqa: E402


def run(config_path: str, epochs: int = None):
    """Trains a visual_only model then evaluates it on the test split,
    returning the parsed test-set metrics dict."""
    cmd = [sys.executable, "train.py", "--config", config_path, "--mode", "visual_only", "--run_name", "exp1_visual_only"]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    print(f"[Experiment 1] Training visual-only model: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parents[1]))

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model_visual_only.pt")

    eval_cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", ckpt_path, "--split", "test", "--config", config_path, "--no_cmas",
    ]
    print(f"[Experiment 1] Evaluating: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True, cwd=str(Path(__file__).resolve().parents[1]))

    results_path = os.path.join(cfg["paths"]["results_dir"], "results.json")
    with open(results_path) as f:
        metrics = json.load(f)

    out_path = os.path.join(cfg["paths"]["results_dir"], "exp1_visual_only.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Experiment 1] Wrote {out_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    run(args.config, args.epochs)
