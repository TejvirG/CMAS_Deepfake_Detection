"""
validate_videos.py — proactively scans every video path referenced in your
manifests (or a raw dataset directory) for files that are unreadable/
corrupted (e.g. truncated during an interrupted extraction — the "moov atom
not found" class of error). Run this once after extracting the dataset,
BEFORE a multi-hour training/evaluation run, so a bad file surfaces in
seconds rather than crashing (or, since dataset.py now skips bad files
gracefully, silently shrinking) a run that's already an hour in.

This does NOT fix corrupted files — a truncated video's data is actually
gone, not recoverable by code. It just tells you which files (if any) are
affected, how many, and gives you the choice: re-extract fresh (often fixes
interrupted-extraction corruption), exclude them from the manifest, or
accept dataset.py's automatic skip-and-continue behavior.

Usage:
    python validate_videos.py --config config.yaml
    python validate_videos.py --raw_dir /content/FakeAVCeleb_raw/FakeAVCeleb_v1.2
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import pandas as pd
import yaml
from tqdm import tqdm


def check_video(path: str) -> str | None:
    """Returns None if the video opens and has at least one readable frame,
    otherwise a short reason string."""
    if not os.path.exists(path):
        return "file does not exist"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return "could not open (corrupted/truncated container)"
    ret, _ = cap.read()
    cap.release()
    if not ret:
        return "opened but no readable frames"
    return None


def collect_paths_from_manifests(cfg: dict) -> list[str]:
    paths = []
    for split in ["train", "val", "test"]:
        manifest_path = cfg["paths"][f"{split}_manifest"]
        if not os.path.exists(manifest_path):
            print(f"[skip] {manifest_path} not found.")
            continue
        df = pd.read_csv(manifest_path)
        paths.extend(df["video_path"].tolist())
    return paths


def collect_paths_from_raw_dir(raw_dir: str) -> list[str]:
    return glob.glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Scan every path in this config's train/val/test manifests.")
    parser.add_argument("--raw_dir", type=str, default=None, help="Scan every .mp4 under this directory instead of manifests.")
    args = parser.parse_args()

    if not args.config and not args.raw_dir:
        raise SystemExit("Pass either --config (scan manifests) or --raw_dir (scan a raw dataset folder).")

    if args.raw_dir:
        paths = collect_paths_from_raw_dir(args.raw_dir)
        print(f"Scanning {len(paths)} videos under {args.raw_dir} ...")
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        paths = collect_paths_from_manifests(cfg)
        print(f"Scanning {len(paths)} videos referenced in {args.config}'s manifests ...")

    bad = []
    for path in tqdm(paths, desc="Validating"):
        reason = check_video(path)
        if reason is not None:
            bad.append((path, reason))

    print(f"\n{len(bad)} / {len(paths)} video(s) failed validation.")
    if bad:
        print("\nBad files:")
        for path, reason in bad:
            print(f"  {path}\n    -> {reason}")
        print(
            "\nSuggested next steps:\n"
            "  1. Try re-extracting the dataset zip fresh (fixes interrupted-extraction corruption):\n"
            "     !unzip -o -q \"<your dataset zip in Drive>\" -d <raw_dir>\n"
            "  2. Re-run this script — if the same files still fail, they're likely corrupted in the\n"
            "     source zip itself, not just this extraction.\n"
            "  3. Either way, dataset.py now skips unreadable files automatically during training/\n"
            "     evaluation rather than crashing (logs a warning per skipped file) — so a run will\n"
            "     complete even without fixing these, just on slightly fewer samples than the full count."
        )
    else:
        print("All videos passed validation.")


if __name__ == "__main__":
    main()
