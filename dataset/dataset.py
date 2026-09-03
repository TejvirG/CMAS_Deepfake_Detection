"""
FakeAVCeleb Dataset for CMAS training/eval.

Expects a manifest CSV with columns:
    video_path, label, manipulated_modality
where label in {REAL, FAKE} and manipulated_modality in {none, audio, video, both}.

Build manifests with `prepare_fakeavceleb.py`.

Features:
  - train / val / test splits (one Dataset instance per manifest)
  - balanced sampling via `make_balanced_sampler`
  - visual + audio augmentation (train split only)
  - disk caching of extracted face frames + raw audio waveforms, keyed by
    video path hash, so repeated epochs don't re-run OpenCV/MediaPipe/ffmpeg.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from models.audio_encoder import extract_audio_ffmpeg, pad_or_trim
from models.visual_encoder import FaceFrameExtractor

logger = logging.getLogger(__name__)

LABEL2IDX = {"REAL": 0, "FAKE": 1}
MODALITY2VEC = {
    # [visual_manipulated_weight, audio_manipulated_weight] ground-truth vector used by CMAS
    "none": [0.0, 0.0],   # REAL sample, no manipulated modality
    "video": [1.0, 0.0],
    "audio": [0.0, 1.0],
    "both": [0.5, 0.5],
}


def _cache_key(video_path: str, *variant_parts: object) -> str:
    """Hashes the video path together with every extraction parameter that
    affects the cached array's shape/content (num_frames, image_size,
    sample_rate, max_duration_sec).

    REVIEW FIX: the original version hashed only `video_path`, so cached
    frames/audio from a run with e.g. num_frames=8 would be silently reused
    by a later run configured with num_frames=16 (same video, different
    array shape) — either crashing downstream batching with a shape
    mismatch, or in the worse case where shapes happened to coincide,
    silently feeding the wrong-resolution/wrong-length data through
    training without any error. Including the config-relevant parameters in
    the hash means a config change naturally invalidates stale cache entries
    (a new cache file is written) instead of reusing a stale one.
    """
    key_str = video_path + "|" + "|".join(str(p) for p in variant_parts)
    return hashlib.sha1(key_str.encode("utf-8")).hexdigest()


class FakeAVCelebDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        cache_dir: str = "cache/",
        num_frames: int = 8,
        image_size: int = 224,
        sample_rate: int = 16000,
        max_duration_sec: float = 3.0,
        split: str = "train",  # train | val | test  (controls whether augmentation is applied)
        augment_cfg: Optional[dict] = None,
        use_cache: bool = True,
    ):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Run prepare_fakeavceleb.py first "
                f"to build train/val/test manifests from the raw FakeAVCeleb release."
            )
        self.df = pd.read_csv(manifest_path)
        required_cols = {"video_path", "label", "manipulated_modality"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest {manifest_path} is missing required columns: {missing}")

        # Validate manipulated_modality / label values up front with a clear
        # error, rather than letting a malformed manifest surface as an
        # opaque KeyError deep inside __getitem__ (previously: MODALITY2VEC[row["manipulated_modality"]]
        # would raise KeyError with no indication of which row or what the
        # valid values are).
        bad_modality_mask = ~self.df["manipulated_modality"].isin(MODALITY2VEC.keys())
        if bad_modality_mask.any():
            bad_values = sorted(self.df.loc[bad_modality_mask, "manipulated_modality"].unique().tolist())
            raise ValueError(
                f"Manifest {manifest_path} contains unrecognized manipulated_modality value(s) "
                f"{bad_values} in {bad_modality_mask.sum()} row(s). Valid values are "
                f"{sorted(MODALITY2VEC.keys())}. Check prepare_fakeavceleb.py's CATEGORY_DIRS mapping "
                f"or fix the manifest CSV directly."
            )
        bad_label_mask = ~self.df["label"].isin(LABEL2IDX.keys())
        if bad_label_mask.any():
            bad_values = sorted(self.df.loc[bad_label_mask, "label"].unique().tolist())
            raise ValueError(
                f"Manifest {manifest_path} contains unrecognized label value(s) {bad_values}. "
                f"Valid values are {sorted(LABEL2IDX.keys())}."
            )
        inconsistent = ((self.df["label"] == "REAL") & (self.df["manipulated_modality"] != "none")) | (
            (self.df["label"] == "FAKE") & (self.df["manipulated_modality"] == "none")
        )
        if inconsistent.any():
            logger.warning(
                f"Manifest {manifest_path} has {inconsistent.sum()} row(s) where label/"
                f"manipulated_modality look inconsistent (REAL with a non-'none' modality, or FAKE "
                f"with modality 'none'). This won't crash, but double-check prepare_fakeavceleb.py's "
                f"category mapping if this wasn't intentional."
            )

        self.split = split
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

        self.face_extractor = FaceFrameExtractor(num_frames=num_frames, image_size=image_size)
        self.num_frames = num_frames
        self.image_size = image_size
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.augment_cfg = augment_cfg or {}

    def __len__(self) -> int:
        return len(self.df)

    # ---------------------------------------------------------------- cache
    def _cached_frames_path(self, video_path: str) -> Path:
        key = _cache_key(video_path, "frames", self.num_frames, self.image_size)
        return self.cache_dir / f"{key}_frames.npy"

    def _cached_audio_path(self, video_path: str) -> Path:
        key = _cache_key(video_path, "audio", self.sample_rate, self.max_duration_sec)
        return self.cache_dir / f"{key}_audio.npy"

    def _get_frames(self, video_path: str) -> np.ndarray:
        cpath = self._cached_frames_path(video_path)
        if self.use_cache and cpath.exists():
            return np.load(cpath)
        frames = self.face_extractor.extract(video_path)
        if self.use_cache:
            np.save(cpath, frames)
        return frames

    def _get_audio(self, video_path: str) -> np.ndarray:
        cpath = self._cached_audio_path(video_path)
        if self.use_cache and cpath.exists():
            return np.load(cpath)
        waveform = extract_audio_ffmpeg(video_path, sample_rate=self.sample_rate, mono=True)
        waveform = pad_or_trim(waveform, self.sample_rate, self.max_duration_sec)
        if self.use_cache:
            np.save(cpath, waveform)
        return waveform

    # ------------------------------------------------------------ augment
    def _augment_frames(self, frames: np.ndarray) -> np.ndarray:
        """frames: (T, H, W, 3) uint8. Applied only when split == 'train'."""
        cfg = self.augment_cfg.get("visual", {})
        if self.split != "train":
            return frames

        if np.random.rand() < cfg.get("horizontal_flip_prob", 0.0):
            frames = frames[:, :, ::-1, :].copy()

        if cfg.get("color_jitter", False) and np.random.rand() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        if np.random.rand() < cfg.get("gaussian_blur_prob", 0.0):
            frames = np.stack([cv2.GaussianBlur(f, (3, 3), 0) for f in frames])

        scale_range = cfg.get("random_crop_scale", None)
        if scale_range:
            scale = np.random.uniform(*scale_range)
            h, w = frames.shape[1:3]
            ch, cw = int(h * scale), int(w * scale)
            top = np.random.randint(0, h - ch + 1)
            left = np.random.randint(0, w - cw + 1)
            cropped = frames[:, top : top + ch, left : left + cw, :]
            frames = np.stack([cv2.resize(f, (w, h)) for f in cropped])

        return frames

    def _augment_audio(self, waveform: np.ndarray) -> np.ndarray:
        cfg = self.augment_cfg.get("audio", {})
        if self.split != "train":
            return waveform

        if np.random.rand() < cfg.get("add_noise_prob", 0.0):
            snr_low, snr_high = cfg.get("noise_snr_db", [10, 30])
            snr_db = np.random.uniform(snr_low, snr_high)
            signal_power = np.mean(waveform ** 2) + 1e-10
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
            waveform = waveform + noise

        if np.random.rand() < cfg.get("time_shift_prob", 0.0):
            max_shift_samples = int(cfg.get("time_shift_max_sec", 0.2) * self.sample_rate)
            shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
            waveform = np.roll(waveform, shift)

        return waveform

    def __getitem__(self, idx: int) -> Optional[dict]:
        """Returns None (instead of raising) if the video file is unreadable
        — e.g. truncated/corrupted ('moov atom not found' from ffmpeg/OpenCV,
        typically caused by an interrupted download or extraction on a large
        dataset pulled across multiple sessions). A single bad file
        previously crashed the entire training/evaluation run partway
        through, often after many minutes of otherwise-successful work; now
        it's logged and skipped, and collate_fn (below) filters out the
        resulting None entries before batching. Genuinely missing/corrupted
        files are a data-integrity problem worth knowing about (see the
        warning below and validate_videos.py), not something to silently
        paper over — but a full run shouldn't die because of one such file
        deep into a multi-hour job."""
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        label_str = row["label"]
        modality = row["manipulated_modality"]

        try:
            frames = self._get_frames(video_path)
            frames = self._augment_frames(frames)
            waveform = self._get_audio(video_path)
            waveform = self._augment_audio(waveform)
        except (IOError, OSError, RuntimeError) as e:
            logger.warning(f"Skipping unreadable video (idx={idx}): {video_path} ({e})")
            return None

        label = LABEL2IDX[label_str]
        modality_vec = torch.tensor(MODALITY2VEC[modality], dtype=torch.float32)

        return {
            "frames": torch.from_numpy(frames.copy()),         # (T, H, W, 3) uint8
            "waveform": torch.from_numpy(waveform.copy()).float(),  # (num_samples,)
            "label": torch.tensor(label, dtype=torch.long),
            "modality_gt": modality_vec,                        # for CMAS ground truth
            "video_path": video_path,
        }


def make_balanced_sampler(dataset: FakeAVCelebDataset) -> WeightedRandomSampler:
    """Inverse-frequency class weighting so REAL/FAKE are seen with equal
    expected frequency per epoch, regardless of raw class imbalance."""
    labels = dataset.df["label"].map(LABEL2IDX).values
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def collate_fn(batch: list) -> Optional[dict]:
    """Filters out None entries (unreadable/corrupted videos — see
    __getitem__) before stacking. Returns None if the ENTIRE batch was bad
    (astronomically unlikely unless something is very wrong dataset-wide);
    callers should skip a None batch rather than crash on it — DataLoader
    itself doesn't filter these, so this must happen here."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "frames": torch.stack([b["frames"] for b in batch]),
        "waveform": torch.stack([b["waveform"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "modality_gt": torch.stack([b["modality_gt"] for b in batch]),
        "video_path": [b["video_path"] for b in batch],
    }
