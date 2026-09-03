"""
verify_environment.py — run this right after `pip install -r requirements.txt`
(and after any runtime restart Colab asks for) to confirm the environment is
actually usable before you touch real data or spend GPU time.

Checks, in order:
  1. Every required library imports cleanly.
  2. Version floors from requirements.txt are actually met (catches the case
     where an old cached wheel or a conflicting install shadowed what pip
     reported it installed).
  3. CUDA/GPU is visible to torch (warns, doesn't fail, if not — CPU still
     works, just slowly).
  4. The system `ffmpeg` binary is on PATH (not a pip package — see
     requirements.txt's note on this).
  5. MediaPipe's Tasks API (mediapipe.tasks.vision.FaceDetector) is present
     and its model asset can be fetched from the network — the actual
     face-detection path, not just "mediapipe imports".
  6. A real forward pass through the visual encoder (EfficientNet) and, if
     reachable, the audio encoder (Wav2Vec2) — catches ABI mismatches that
     "it imports" alone wouldn't (e.g. a torch/torchvision build mismatch).

Usage:
    python verify_environment.py

Exit code is 0 if every check passes, 1 otherwise — safe to use in a CI
step or Colab cell (`!python verify_environment.py`) and act on the result.
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    warning_only: bool = False


RESULTS: List[CheckResult] = []


def _run_check(name: str, fn: Callable[[], str], warning_only: bool = False) -> None:
    try:
        message = fn()
        RESULTS.append(CheckResult(name, True, message, warning_only))
        print(f"  [OK]   {name}: {message}")
    except Exception as e:  # noqa: BLE001 - deliberately broad, this IS the error-collection layer
        RESULTS.append(CheckResult(name, False, str(e), warning_only))
        tag = "WARN" if warning_only else "FAIL"
        print(f"  [{tag}] {name}: {e}")


# ------------------------------------------------------------- 1+2. imports
_REQUIRED_LIBS = [
    # (import name, pip package name, minimum version or None)
    ("torch", "torch", "2.4.0"),
    ("torchvision", "torchvision", "0.19.0"),
    ("transformers", "transformers", "4.46.0"),
    ("timm", "timm", "1.0.0"),
    ("captum", "captum", "0.7.0"),
    ("cv2", "opencv-contrib-python", "4.9.0"),
    ("mediapipe", "mediapipe", "0.10.14"),
    ("numpy", "numpy", "1.26.0"),
    ("pandas", "pandas", "2.1.0"),
    ("sklearn", "scikit-learn", "1.4.0"),
    ("matplotlib", "matplotlib", "3.8.0"),
    ("seaborn", "seaborn", "0.13.0"),
    ("tqdm", "tqdm", "4.66.0"),
    ("yaml", "pyyaml", "6.0"),
    ("gradio", "gradio", "5.0.0"),
    ("tensorboard", "tensorboard", "2.16.0"),
    ("soundfile", "soundfile", "0.12.1"),
    ("pytest", "pytest", "7.4.0"),
]


def _parse_version(v: str) -> tuple:
    parts = []
    for p in v.split(".")[:3]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def check_import_and_version(import_name: str, pip_name: str, min_version: Optional[str]) -> str:
    mod = importlib.import_module(import_name)
    version = getattr(mod, "__version__", None)
    if version is None:
        try:
            import importlib.metadata as im

            version = im.version(pip_name)
        except Exception:
            version = "unknown"
    if min_version is not None and version != "unknown":
        if _parse_version(str(version)) < _parse_version(min_version):
            raise RuntimeError(
                f"imported OK but version {version} < required floor {min_version} "
                f"(run: pip install -U '{pip_name}>={min_version}')"
            )
    return f"version {version}"


# ---------------------------------------------------------------- 3. CUDA
def check_cuda() -> str:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA available — {name}"
    return "CUDA NOT available — will run on CPU (slow, but functional; check Runtime > Change runtime type on Colab)"


# --------------------------------------------------------------- 4. ffmpeg
def check_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "system `ffmpeg` binary not found on PATH. Install it: "
            "`apt-get install -y ffmpeg` (Colab/Ubuntu/Debian) or `brew install ffmpeg` (macOS). "
            "Not a pip package — pip installing 'ffmpeg-python' will NOT fix this."
        )
    result = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=10)
    first_line = result.stdout.splitlines()[0] if result.stdout else "version string unavailable"
    return f"found at {path} ({first_line})"


# ------------------------------------------------------- 5. MediaPipe Tasks
def check_mediapipe_tasks_api() -> str:
    import mediapipe as mp

    has_tasks = (
        hasattr(mp, "tasks")
        and hasattr(mp.tasks, "vision")
        and hasattr(mp.tasks.vision, "FaceDetector")
        and hasattr(mp, "Image")
        and hasattr(mp, "ImageFormat")
    )
    if not has_tasks:
        raise RuntimeError(
            "mediapipe is installed but the Tasks API (mediapipe.tasks.vision.FaceDetector) is "
            "missing — this is the API models/visual_encoder.py relies on. This would be unusual "
            "for any mediapipe>=0.10.14; check `pip show mediapipe` for the actual installed version."
        )
    return "mediapipe.tasks.vision.FaceDetector is present"


def check_mediapipe_model_download() -> str:
    """Downloads (or confirms already-cached) the actual face-detector model
    asset. This is the check most likely to legitimately fail in a
    network-restricted environment — that's expected and non-fatal, the
    pipeline degrades to center-crop (see models/visual_encoder.py), but you
    should know about it before training rather than discovering it later."""
    sys.path.insert(0, ".")
    from models.visual_encoder import _ensure_face_detector_model  # local import: needs repo on path

    model_path = _ensure_face_detector_model()
    if model_path is None:
        raise RuntimeError(
            "could not download the MediaPipe face-detector model asset (network-restricted "
            "environment, or storage.googleapis.com is blocked). Training/inference will still "
            "run — face cropping falls back to a full-frame center crop — but you will not get "
            "real face detection until this succeeds. Check firewall/proxy settings if this is "
            "unexpected on your machine."
        )
    return f"model cached at {model_path}"


# --------------------------------------------------------- 6. real forward pass
def check_visual_encoder_forward() -> str:
    import torch

    from models.visual_encoder import VisualEncoder

    model = VisualEncoder(backbone="efficientnet_b0", pretrained=False)
    model.eval()
    frames = torch.randint(0, 256, (1, 2, 96, 96, 3), dtype=torch.uint8)
    x = model.preprocess(frames)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, model.embed_dim), f"unexpected output shape {out.shape}"
    return f"EfficientNet-B0 forward pass OK, embed_dim={model.embed_dim}"


def check_audio_encoder_forward() -> str:
    import numpy as np

    from models.audio_encoder import AudioEncoder

    model = AudioEncoder("facebook/wav2vec2-base")
    model.eval()
    waveform = np.random.randn(1, 16000).astype(np.float32)
    x = model.preprocess(waveform, sample_rate=16000)
    import torch

    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, model.embed_dim), f"unexpected output shape {out.shape}"
    return f"Wav2Vec2 forward pass OK, embed_dim={model.embed_dim}"


def main() -> int:
    print("=" * 70)
    print("CMAS-Deepfake-Detection — environment verification")
    print("=" * 70)

    print("\n[1/6] Library imports + version floors")
    for import_name, pip_name, min_version in _REQUIRED_LIBS:
        _run_check(f"{pip_name}", lambda i=import_name, p=pip_name, m=min_version: check_import_and_version(i, p, m))

    print("\n[2/6] GPU / CUDA")
    _run_check("CUDA", check_cuda, warning_only=True)

    print("\n[3/6] System ffmpeg binary")
    _run_check("ffmpeg", check_ffmpeg)

    print("\n[4/6] MediaPipe Tasks API")
    _run_check("mediapipe.tasks API surface", check_mediapipe_tasks_api)

    print("\n[5/6] MediaPipe face-detector model download")
    _run_check("mediapipe model asset", check_mediapipe_model_download, warning_only=True)

    print("\n[6/6] Real forward passes (catches ABI/build mismatches that imports alone miss)")
    _run_check("VisualEncoder (EfficientNet-B0) forward pass", check_visual_encoder_forward)
    _run_check("AudioEncoder (Wav2Vec2) forward pass", check_audio_encoder_forward, warning_only=True)
    # audio check is warning_only because it requires reaching huggingface.co
    # to fetch pretrained weights on first use — a legitimate network
    # restriction, not a broken environment, if it fails.

    print("\n" + "=" * 70)
    hard_failures = [r for r in RESULTS if not r.ok and not r.warning_only]
    warnings = [r for r in RESULTS if not r.ok and r.warning_only]

    if hard_failures:
        print(f"RESULT: {len(hard_failures)} check(s) FAILED, {len(warnings)} warning(s).")
        print("Fix the FAIL items above before running train.py — WARN items are informational")
        print("(e.g. no GPU, or a network-restricted download) and won't crash the pipeline.")
        return 1
    elif warnings:
        print(f"RESULT: all required checks passed. {len(warnings)} warning(s) — see WARN lines above.")
        return 0
    else:
        print("RESULT: all checks passed cleanly. Environment is ready.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
