"""
prepare_fakeavceleb.py — builds train/val/test manifest CSVs from a local
FakeAVCeleb release.

FakeAVCeleb must be obtained directly from the dataset authors (research-use
agreement required): https://sites.google.com/view/fakeavcelebdataset

Expected raw layout (official release):
    FakeAVCeleb/
      RealVideo-RealAudio/<id>/*.mp4
      FakeVideo-RealAudio/<id>/*.mp4      # visual manipulated, audio real
      RealVideo-FakeAudio/<id>/*.mp4      # audio manipulated, visual real
      FakeVideo-FakeAudio/<id>/*.mp4      # both manipulated

If your local copy differs from this layout, adjust `CATEGORY_DIRS` below —
that's the only thing this script assumes about directory structure.

Usage:
    python prepare_fakeavceleb.py --raw_dir /path/to/FakeAVCeleb \
        --out_dir data/fakeavceleb --val_ratio 0.15 --test_ratio 0.15
"""
from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path

import pandas as pd

# category folder name -> (label, manipulated_modality)
CATEGORY_DIRS = {
    "RealVideo-RealAudio": ("REAL", "none"),
    "FakeVideo-RealAudio": ("FAKE", "video"),
    "RealVideo-FakeAudio": ("FAKE", "audio"),
    "FakeVideo-FakeAudio": ("FAKE", "both"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to the raw FakeAVCeleb release.")
    parser.add_argument("--out_dir", type=str, default="data/fakeavceleb")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_ext", type=str, default="mp4")
    return parser.parse_args()


def collect_videos(raw_dir: str, video_ext: str) -> pd.DataFrame:
    rows = []
    for category, (label, modality) in CATEGORY_DIRS.items():
        pattern = os.path.join(raw_dir, category, "**", f"*.{video_ext}")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            print(f"[warn] No videos found under {os.path.join(raw_dir, category)} — "
                  f"check --raw_dir layout matches CATEGORY_DIRS in this script.")
        for path in matches:
            rows.append({"video_path": os.path.abspath(path), "label": label, "manipulated_modality": modality})
    if not rows:
        raise RuntimeError(
            f"No videos found under {raw_dir} for any known category. "
            f"Verify the FakeAVCeleb directory layout, or edit CATEGORY_DIRS in this script."
        )
    return pd.DataFrame(rows)


def split_by_identity(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int):
    """Splits by (best-effort) identity folder name, to prevent the same
    speaker's face/voice from leaking across train/val/test. Falls back to a
    random per-video split if identity can't be inferred from the path."""

    def infer_identity(path: str) -> str:
        parts = Path(path).parts
        # Heuristic: the identity folder is typically two levels above the
        # video file in the official FakeAVCeleb layout (category/id/*.mp4).
        return parts[-2] if len(parts) >= 2 else path

    df = df.copy()
    df["identity"] = df["video_path"].apply(infer_identity)

    identities = df["identity"].unique().tolist()
    random.Random(seed).shuffle(identities)

    n_val = max(1, int(len(identities) * val_ratio))
    n_test = max(1, int(len(identities) * test_ratio))

    val_ids = set(identities[:n_val])
    test_ids = set(identities[n_val : n_val + n_test])
    train_ids = set(identities[n_val + n_test :])

    train_df = df[df["identity"].isin(train_ids)].drop(columns=["identity"])
    val_df = df[df["identity"].isin(val_ids)].drop(columns=["identity"])
    test_df = df[df["identity"].isin(test_ids)].drop(columns=["identity"])
    return train_df, val_df, test_df


def main():
    args = parse_args()
    df = collect_videos(args.raw_dir, args.video_ext)
    print(f"Found {len(df)} videos across {df['label'].nunique()} labels, "
          f"{df['manipulated_modality'].value_counts().to_dict()}")

    train_df, val_df, test_df = split_by_identity(df, args.val_ratio, args.test_ratio, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train_manifest.csv", index=False)
    val_df.to_csv(out_dir / "val_manifest.csv", index=False)
    test_df.to_csv(out_dir / "test_manifest.csv", index=False)

    print(f"train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    print(f"Wrote manifests to {out_dir}/")


if __name__ == "__main__":
    main()
