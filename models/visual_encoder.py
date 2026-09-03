"""
Visual branch: face-frame extraction (OpenCV + MediaPipe) and
EfficientNet-B0/B4 embedding extraction, averaged across sampled frames.

REVIEW NOTE (dependency audit): this file previously used the legacy
`mediapipe.solutions.face_detection` API. That API's support ended on March
1, 2023 (per MediaPipe's own docs/changelog) in favor of the "MediaPipe
Tasks" API; it doesn't exist as a working, supported code path in any
current MediaPipe release, and pinning an old MediaPipe version to keep it
alive doesn't work either — the specific old version that had it
(0.10.9) isn't even in the current package index anymore. Rewritten below to
use `mediapipe.tasks.python.vision.FaceDetector`, which is the current,
actively maintained API (verified against the installed package's own
source in this environment, and against Google's current documentation).
"""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)

# Official model asset for the Tasks API's short-range face detector (BlazeFace).
# Source: https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python
_FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_DEFAULT_MODEL_CACHE_DIR = os.path.expanduser("~/.cache/cmas_deepfake/mediapipe_models")

try:
    import mediapipe as mp

    # Probe for the Tasks API explicitly (mp.tasks.vision.FaceDetector) rather
    # than assuming import success implies it exists — mirrors the same
    # defensive pattern used elsewhere in this module for the model download.
    _MP_TASKS_AVAILABLE = (
        hasattr(mp, "tasks")
        and hasattr(mp.tasks, "vision")
        and hasattr(mp.tasks.vision, "FaceDetector")
        and hasattr(mp, "Image")
        and hasattr(mp, "ImageFormat")
    )
    if not _MP_TASKS_AVAILABLE:
        logger.warning(
            "mediapipe is installed but the Tasks API (mediapipe.tasks.vision.FaceDetector) "
            "is unavailable in this build; face cropping will fall back to full-frame center crop."
        )
except ImportError:  # pragma: no cover
    _MP_TASKS_AVAILABLE = False
    logger.warning("mediapipe not installed; face cropping will fall back to full-frame center crop.")


def _ensure_face_detector_model(cache_dir: str = _DEFAULT_MODEL_CACHE_DIR, timeout_sec: float = 30.0) -> Optional[str]:
    """Downloads the BlazeFace short-range .task model on first use and caches
    it locally (~230KB, one-time download). Returns the local path, or None
    if the download fails for any reason (no internet, blocked domain,
    timeout, etc.) — callers must treat None as "fall back to center crop",
    the same robustness contract the rest of this module already follows for
    corrupt videos / missing faces."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(cache_dir, "blaze_face_short_range.tflite")
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path
    try:
        tmp_path = model_path + ".tmp"
        urllib.request.urlretrieve(_FACE_DETECTOR_MODEL_URL, tmp_path)  # nosec - fixed, hardcoded Google-owned URL
        os.replace(tmp_path, model_path)  # atomic on POSIX, avoids partial-file races
        return model_path
    except Exception as e:  # noqa: BLE001 - deliberately broad: any failure here must degrade, not crash
        logger.warning(
            f"Could not download MediaPipe face detector model from {_FACE_DETECTOR_MODEL_URL} "
            f"({e}); face cropping will fall back to full-frame center crop. This is expected in "
            f"network-restricted environments and is not fatal — training/inference still runs."
        )
        return None


class FaceFrameExtractor:
    """Samples N frames from a video, detects a face in each with MediaPipe's
    Tasks API (BlazeFace short-range detector), and returns cropped+resized
    face RGB arrays. Falls back to a center crop of the full frame if no face
    is found, if the model can't be downloaded, or if the Tasks API isn't
    available in the installed mediapipe build — keeps the pipeline robust to
    low-quality clips and restricted environments instead of crashing
    training."""

    def __init__(
        self,
        num_frames: int = 8,
        image_size: int = 224,
        min_detection_confidence: float = 0.5,
        model_cache_dir: str = _DEFAULT_MODEL_CACHE_DIR,
    ):
        self.num_frames = num_frames
        self.image_size = image_size
        self._detector = None

        if _MP_TASKS_AVAILABLE:
            model_path = _ensure_face_detector_model(model_cache_dir)
            if model_path is not None:
                try:
                    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
                    options = mp.tasks.vision.FaceDetectorOptions(
                        base_options=base_options,
                        running_mode=mp.tasks.vision.RunningMode.IMAGE,
                        min_detection_confidence=min_detection_confidence,
                    )
                    self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)
                except Exception as e:  # noqa: BLE001 - degrade to center-crop on any setup failure
                    logger.warning(
                        f"Failed to initialize MediaPipe FaceDetector ({e}); "
                        f"face cropping will fall back to full-frame center crop."
                    )
                    self._detector = None

    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            return []
        n = min(self.num_frames, total_frames)
        return np.linspace(0, total_frames - 1, num=n, dtype=int).tolist()

    def _detect_and_crop(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self._detector is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_image)
            if result.detections:
                # Use the highest-confidence detection. Category.score is the
                # Tasks API's confidence field (see mediapipe.tasks.python.
                # components.containers.category.Category).
                det = max(result.detections, key=lambda d: d.categories[0].score if d.categories else 0.0)
                box = det.bounding_box  # BoundingBox(origin_x, origin_y, width, height) — ABSOLUTE pixels,
                # unlike the legacy API's relative_bounding_box (0-1 normalized). No w/h multiplication needed.
                x1, y1, bw, bh = box.origin_x, box.origin_y, box.width, box.height
                # Pad the box by 20% for context around the face
                pad_x, pad_y = int(bw * 0.2), int(bh * 0.2)
                x1 = max(x1 - pad_x, 0)
                y1 = max(y1 - pad_y, 0)
                x2 = min(x1 + bw + 2 * pad_x, w)
                y2 = min(y1 + bh + 2 * pad_y, h)
                if x2 > x1 and y2 > y1:
                    face = frame_bgr[y1:y2, x1:x2]
                    return cv2.resize(face, (self.image_size, self.image_size))
        # Fallback: center crop
        side = min(h, w)
        cy, cx = h // 2, w // 2
        crop = frame_bgr[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]
        return cv2.resize(crop, (self.image_size, self.image_size))

    def extract(self, video_path: str) -> np.ndarray:
        """Returns an array of shape (num_frames, H, W, 3) in RGB, uint8."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._sample_frame_indices(total_frames)
        frames = []
        idx_set = set(indices)
        cur = 0
        while cap.isOpened() and len(frames) < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
            if cur in idx_set:
                face = self._detect_and_crop(frame)
                frames.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            cur += 1
        cap.release()

        if len(frames) == 0:
            # Degenerate video (corrupt / zero frames) - return a black frame
            # so downstream batching doesn't crash; flagged via logger.
            logger.warning(f"No frames extracted from {video_path}; using blank frame.")
            frames = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)]

        # Pad by repeating the last frame if the video was shorter than num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return np.stack(frames[: self.num_frames], axis=0)


class VisualEncoder(nn.Module):
    """EfficientNet-B0/B4 (ImageNet-pretrained) feature extractor. Consumes a
    batch of sampled face frames per video, shape (B, T, 3, H, W), and returns
    a single embedding per video by averaging per-frame embeddings."""

    _EMBED_DIMS = {"efficientnet_b0": 1280, "efficientnet_b4": 1792}

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, freeze: bool = False):
        super().__init__()
        if backbone not in self._EMBED_DIMS:
            raise ValueError(f"Unsupported visual backbone: {backbone}")
        self.backbone_name = backbone
        self.embed_dim = self._EMBED_DIMS[backbone]

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)  # global-pooled features
        self.set_trainable(not freeze)

        # ImageNet normalization stats
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def set_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def preprocess(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        """frames_uint8: (B, T, H, W, 3) uint8 -> normalized (B, T, 3, H, W) float."""
        x = frames_uint8.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # B,T,3,H,W
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = (x - self.mean) / self.std
        return x.view(b, t, c, h, w)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, 3, H, W) already-normalized float tensor.
        Returns: (B, embed_dim) — mean-pooled across the T sampled frames.
        """
        b, t, c, h, w = frames.shape
        flat = frames.view(b * t, c, h, w)
        feats = self.backbone(flat)  # (B*T, embed_dim)
        feats = feats.view(b, t, -1)
        return feats.mean(dim=1)  # average frame embeddings -> (B, embed_dim)
