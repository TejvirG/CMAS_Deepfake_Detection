"""
Audio branch: ffmpeg-based extraction of mono 16kHz audio from video, and
Wav2Vec2-base (HuggingFace) embedding extraction.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

logger = logging.getLogger(__name__)


def extract_audio_ffmpeg(video_path: str, sample_rate: int = 16000, mono: bool = True) -> np.ndarray:
    """Extracts audio from a video file via ffmpeg, resampled to `sample_rate`
    Hz mono, and returns a float32 waveform array. Requires the `ffmpeg`
    binary to be available on PATH."""
    channels = 1 if mono else 2
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", str(sample_rate), "-ac", str(channels),
        "-vn", "-loglevel", "error", tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        waveform, sr = sf.read(tmp_path, dtype="float32")
        assert sr == sample_rate
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"ffmpeg audio extraction failed for {video_path} ({e}); returning silence.")
        waveform = np.zeros(sample_rate, dtype=np.float32)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)  # downmix to mono if extraction produced stereo
    return waveform


def pad_or_trim(waveform: np.ndarray, sample_rate: int, max_duration_sec: float) -> np.ndarray:
    target_len = int(sample_rate * max_duration_sec)
    if len(waveform) >= target_len:
        return waveform[:target_len]
    pad = np.zeros(target_len - len(waveform), dtype=waveform.dtype)
    return np.concatenate([waveform, pad])


class AudioEncoder(nn.Module):
    """Wav2Vec2-base (facebook/wav2vec2-base) wrapper producing a single
    pooled embedding per clip (mean-pooled over the time dimension of the
    last hidden state)."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base", freeze: bool = False):
        super().__init__()
        self.model_name = model_name
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size  # 768 for wav2vec2-base
        self.set_trainable(not freeze)

    def set_trainable(self, trainable: bool) -> None:
        for p in self.model.parameters():
            p.requires_grad = trainable

    def preprocess(self, waveforms: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """waveforms: (B, num_samples) numpy array -> normalized input tensor for wav2vec2."""
        inputs = self.feature_extractor(
            [w for w in waveforms], sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        return inputs.input_values  # (B, num_samples)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        input_values: (B, num_samples) preprocessed waveform tensor.
        Returns: (B, embed_dim) mean-pooled hidden states.
        """
        outputs = self.model(input_values)
        hidden = outputs.last_hidden_state  # (B, T, embed_dim)
        return hidden.mean(dim=1)
