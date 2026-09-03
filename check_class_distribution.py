"""
check_class_distribution.py — direct, independent confirmation of class
balance in each manifest (no model involved). Run this once against your
already-generated manifests to confirm the test-set imbalance is what the
evaluation-metrics audit implies it is, rather than trusting the arithmetic
derivation alone.

Usage:
    python check_class_distribution.py --config config.yaml
"""
from __future__ import annotations

import argparse

import pandas as pd
import yaml


def report(name: str, path: str) -> None:
    df = pd.read_csv(path)
    n = len(df)
    print(f"\n{name} ({path}) — {n} samples")
    print("  label:")
    for label, count in df["label"].value_counts().items():
        print(f"    {label:6s}: {count:6d}  ({100*count/n:.2f}%)")
    print("  manipulated_modality:")
    for mod, count in df["manipulated_modality"].value_counts().items():
        print(f"    {mod:6s}: {count:6d}  ({100*count/n:.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for split in ["train", "val", "test"]:
        report(split, cfg["paths"][f"{split}_manifest"])

    print(
        "\nExpected FAKE fraction across the full FakeAVCeleb release is ~97.68% "
        "(21044/21544). Each split above should be reasonably close to that — "
        "identity-based splitting can cause some drift, but large deviations "
        "(e.g. a split with 0% or >99.5% of one class) would indicate the split "
        "itself needs attention, not just the model or the decision threshold."
    )


if __name__ == "__main__":
    main()
