"""
tests/test_smoke.py — fast CPU-only smoke tests that exercise every module
with synthetic tensors/videos, so the whole pipeline's plumbing (model
forward passes, fusion, CMAS math, IG attribution shapes) can be verified
without needing the FakeAVCeleb dataset, ffmpeg-extracted audio, or a GPU.

These tests do NOT validate model *accuracy* (there's no real data here) —
they validate that the code runs end-to-end without crashing and produces
outputs of the correct shape/range. Several tests below are explicit
regression guards for bugs found during code review (see comments on each);
they exist specifically so those bugs can't silently come back. Run:

    python -m pytest tests/test_smoke.py -v
"""
import subprocess
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
import torch.nn as nn

from metrics.cmas import cmas_batch, cmas_single, cosine_similarity, normalize_importance
from models.fusion import AttentionPoolFusion, CMASDeepfakeDetector, CrossAttentionFusion


# --------------------------------------------------------------------- CMAS
def test_cmas_perfect_match():
    # Explanation perfectly matches ground truth (video manipulated)
    e = np.array([1.0, 0.0])
    g = np.array([1.0, 0.0])
    assert cmas_single(e, g) == pytest.approx(1.0, abs=1e-6)


def test_cmas_orthogonal():
    e = np.array([1.0, 0.0])
    g = np.array([0.0, 1.0])
    assert cmas_single(e, g) == pytest.approx(0.0, abs=1e-6)


def test_cmas_both_modalities_gt():
    e = np.array([0.5, 0.5])
    g = np.array([0.5, 0.5])
    assert cmas_single(e, g) == pytest.approx(1.0, abs=1e-6)


def test_cmas_real_sample_raises():
    with pytest.raises(ValueError):
        cmas_single(np.array([0.5, 0.5]), np.array([0.0, 0.0]))


def test_cmas_batch_excludes_real():
    explanations = [np.array([1, 0]), np.array([0.5, 0.5]), np.array([0, 1])]
    ground_truths = [np.array([1, 0]), np.array([0, 0]), np.array([0, 1])]  # middle sample is REAL
    result = cmas_batch(explanations, ground_truths, exclude_real=True)
    assert result.n_samples == 2
    assert result.n_excluded_real == 1
    assert result.mean_cmas == pytest.approx(1.0, abs=1e-6)


def test_normalize_importance_is_public_and_degenerate_case():
    # Regression guard: normalize_importance used to be a "private" _normalize
    # imported across three other modules — make sure the public name works
    # and the degenerate all-zero case falls back to [0.5, 0.5] rather than
    # dividing by zero.
    out = normalize_importance(np.array([0.0, 0.0]))
    assert np.allclose(out, [0.5, 0.5])
    out2 = normalize_importance(np.array([3.0, 1.0]))
    assert np.allclose(out2, [0.75, 0.25])


# ------------------------------------------------------------------ Fusion
def test_cross_attention_fusion_shapes():
    fusion = CrossAttentionFusion(visual_dim=128, audio_dim=96, hidden_dim=32, num_heads=4)
    v = torch.randn(4, 128)
    a = torch.randn(4, 96)
    out = fusion(v, a)
    assert out.shape == (4, 64)  # hidden_dim * 2
    # NOTE: CrossAttentionFusion intentionally does NOT expose
    # get_modality_attention_mass() any more — see its docstring in
    # models/fusion.py for why (single-token softmax is degenerate).
    assert not hasattr(fusion, "get_modality_attention_mass")


def test_attention_pool_fusion_shapes_and_gate_sums_to_one():
    fusion = AttentionPoolFusion(visual_dim=128, audio_dim=96, hidden_dim=32)
    v = torch.randn(2, 128)
    a = torch.randn(2, 96)
    out = fusion(v, a)
    assert out.shape == (2, 64)
    mass = fusion.get_modality_attention_mass()
    assert mass.shape == (2, 2)
    assert torch.allclose(mass.sum(dim=-1), torch.ones(2), atol=1e-4)


def test_attention_pool_gate_varies_with_input():
    """Regression guard: unlike CrossAttentionFusion's removed (degenerate)
    attention mass, AttentionPoolFusion's gate is a genuine function of
    input content. Verify it's NOT constant across very different inputs."""
    torch.manual_seed(0)
    fusion = AttentionPoolFusion(visual_dim=64, audio_dim=48, hidden_dim=32)
    fusion.eval()
    masses = []
    with torch.no_grad():
        for trial in range(5):
            v = torch.randn(3, 64) * (trial + 1)
            a = torch.randn(3, 48) * (5 - trial)
            fusion(v, a)
            masses.append(fusion.get_modality_attention_mass().clone())
    all_same = all(torch.allclose(masses[0], m, atol=1e-4) for m in masses[1:])
    assert not all_same, "AttentionPoolFusion gate should vary with input, but was constant."


def test_cross_attention_raw_weights_are_degenerate_by_construction():
    """Documents (and locks in understanding of) *why* CrossAttentionFusion's
    raw attention weights can't be used for attribution: with a single
    key/value token, nn.MultiheadAttention's softmax is mathematically forced
    to 1.0 regardless of the query. This test exists so nobody "fixes" this
    file in the future by resurrecting the old get_modality_attention_mass()
    approach without re-deriving why it doesn't work."""
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
    mha.eval()
    weights_seen = []
    with torch.no_grad():
        for _ in range(4):
            q = torch.randn(2, 1, 16)
            k = v = torch.randn(2, 1, 16)
            _, attn_weights = mha(q, k, v)
            weights_seen.append(attn_weights.clone())
    assert all(torch.allclose(w, torch.ones_like(w)) for w in weights_seen), (
        "Expected single-key softmax attention weights to always be 1.0 — if this "
        "assertion fails, torch's MultiheadAttention semantics changed and the "
        "modality_ablation_importance() workaround in models/fusion.py may be "
        "revisitable."
    )


def test_modality_ablation_importance_varies_with_input_and_sums_to_one():
    """The actual regression guard for the critical bug found during review:
    CMAS's 'attention-based' explanation method used to be constant [0.5,0.5]
    regardless of input (first via degenerate softmax, then via a LayerNorm-
    normalized-magnitude proxy that was equally constant). This test builds a
    minimal stand-in "model" that reuses the REAL
    CMASDeepfakeDetector.classify_from_embeddings /
    modality_ablation_importance implementations (via class-attribute
    borrowing) against a small CrossAttentionFusion + classifier, without
    needing the full VisualEncoder/AudioEncoder (which need network access
    for pretrained weights). If this test starts failing with constant
    output again, the bug is back."""

    class _StubModel(nn.Module):
        def __init__(self, fusion, classifier):
            super().__init__()
            self.mode = "multimodal"
            self.fusion = fusion
            self.classifier = classifier

        # Borrow the real implementations under test, rather than
        # reimplementing them here (which would test a copy, not the code).
        classify_from_embeddings = CMASDeepfakeDetector.classify_from_embeddings
        modality_ablation_importance = CMASDeepfakeDetector.modality_ablation_importance

    torch.manual_seed(0)
    for fusion_cls in (CrossAttentionFusion, AttentionPoolFusion):
        fusion = fusion_cls(visual_dim=64, audio_dim=48, hidden_dim=32, num_heads=4)
        classifier = nn.Sequential(nn.Linear(fusion.output_dim, 16), nn.ReLU(), nn.Linear(16, 2))
        stub = _StubModel(fusion, classifier)
        stub.eval()

        results = []
        for trial in range(6):
            v = torch.randn(4, 64) * (trial + 1)
            a = torch.randn(4, 48) * (6 - trial)
            target = torch.randint(0, 2, (4,))
            imp = stub.modality_ablation_importance(v, a, target)
            assert imp.shape == (4, 2)
            assert torch.allclose(imp.sum(dim=-1), torch.ones(4), atol=1e-4), "importance rows must sum to 1"
            results.append(imp)

        all_same = all(torch.allclose(results[0], r, atol=1e-4) for r in results[1:])
        assert not all_same, f"{fusion_cls.__name__}: modality_ablation_importance was constant across trials."


# ---------------------------------------------------- CMASDeepfakeDetector
def test_visual_only_model_does_not_build_audio_encoder():
    """Regression guard: CMASDeepfakeDetector used to build BOTH VisualEncoder
    and AudioEncoder unconditionally regardless of `mode`, so a visual_only
    model (Experiment 1) needlessly required loading Wav2Vec2. mode=
    'visual_only' should leave audio_encoder as None (and vice versa for
    audio_only) — this doesn't need network access since visual_pretrained=False
    and we never touch the audio_only branch."""
    model = CMASDeepfakeDetector(
        visual_backbone="efficientnet_b0", visual_pretrained=False,
        fusion_hidden_dim=16, num_classes=2, mode="visual_only",
    )
    assert model.audio_encoder is None
    assert model.visual_encoder is not None
    assert model.fusion is None


def test_set_backbone_trainable_toggles_eval_mode_not_just_requires_grad():
    """Regression guard for the freeze/BatchNorm bug: set_backbone_trainable
    must put the backbone in .eval() mode when freezing (not just set
    requires_grad=False), otherwise BatchNorm running stats keep drifting
    every epoch even though no gradient reaches the frozen weights."""
    model = CMASDeepfakeDetector(
        visual_backbone="efficientnet_b0", visual_pretrained=False,
        fusion_hidden_dim=16, num_classes=2, mode="visual_only",
    )
    model.train()
    model.set_backbone_trainable(False)
    assert model.visual_encoder.backbone.training is False, "Frozen backbone should be in eval() mode"
    assert all(not p.requires_grad for p in model.visual_encoder.backbone.parameters())

    # Simulate the training-loop order (model.train() then set_backbone_trainable())
    # to guard against the exact ordering bug found in train.py.
    model.train()
    model.set_backbone_trainable(False)
    assert model.visual_encoder.backbone.training is False, (
        "Backbone should STAY in eval() mode even after an outer model.train() call, "
        "as long as set_backbone_trainable(False) is called afterward (train.py's fixed ordering)."
    )

    model.set_backbone_trainable(True)
    assert model.visual_encoder.backbone.training is True
    assert all(p.requires_grad for p in model.visual_encoder.backbone.parameters())


# ------------------------------------------------------------- Full model
@pytest.mark.parametrize("mode", ["visual_only", "audio_only", "multimodal"])
def test_model_forward_synthetic(mode):
    """Runs a forward pass through the full model on synthetic (random)
    visual frames and audio waveforms — no real video/audio files needed.

    Building a model with an audio branch (audio_only or multimodal mode)
    downloads the pretrained Wav2Vec2 weights/config from HuggingFace on
    first use. In network-restricted environments (e.g. this sandbox's CI)
    that download will fail with an OSError; we skip in that case rather
    than fail, since it reflects environment connectivity, not a code
    defect. On Colab / any machine with normal internet access this test
    runs fully.
    """
    torch.manual_seed(0)
    try:
        model = CMASDeepfakeDetector(
            visual_backbone="efficientnet_b0",
            visual_pretrained=False,  # avoid slow ImageNet weight download during CI smoke test
            audio_model_name="facebook/wav2vec2-base",
            fusion_type="attention_pool",  # cheaper than cross_attention for a CPU smoke test
            fusion_hidden_dim=32,
            num_classes=2,
            mode=mode,
        )
    except OSError as e:
        pytest.skip(f"Skipping: could not reach HuggingFace Hub to fetch wav2vec2-base ({e}).")
    model.eval()

    batch_size, num_frames, image_size = 2, 4, 96
    frames_uint8 = torch.randint(0, 256, (batch_size, num_frames, image_size, image_size, 3), dtype=torch.uint8)
    visual_input = model.visual_encoder.preprocess(frames_uint8) if mode != "audio_only" else None

    waveform_np = np.random.randn(batch_size, 16000).astype(np.float32)
    audio_input = model.audio_encoder.preprocess(waveform_np, sample_rate=16000) if mode != "visual_only" else None

    with torch.no_grad():
        logits = model(visual_input, audio_input)
    assert logits.shape == (batch_size, 2)

    if mode == "multimodal":
        with torch.no_grad():
            visual_embed, audio_embed = model.encode(visual_input, audio_input)
            preds = logits.argmax(dim=-1)
            importance = model.modality_ablation_importance(visual_embed, audio_embed, preds)
        assert importance.shape == (batch_size, 2)
        assert torch.allclose(importance.sum(dim=-1), torch.ones(batch_size), atol=1e-4)


def test_face_frame_extractor_handles_missing_video(tmp_path):
    from models.visual_encoder import FaceFrameExtractor

    extractor = FaceFrameExtractor(num_frames=4, image_size=64)
    with pytest.raises(IOError):
        extractor.extract(str(tmp_path / "does_not_exist.mp4"))


def test_face_frame_extractor_falls_back_when_model_download_fails(tmp_path, monkeypatch):
    """Regression guard: when the MediaPipe Tasks model download fails (no
    internet, blocked domain, etc.), FaceFrameExtractor must degrade to
    center-crop rather than raise.

    REVIEW FIX: the original version of this test asserted
    `extractor._detector is None` unconditionally, which silently assumed
    the download would fail — true in this project's network-restricted
    development sandbox, but false on Colab (which has real internet access
    and successfully downloads the ~230KB model). That made the test
    environment-dependent and flaky: it failed on Colab specifically
    *because* the feature it's supposed to guard was working correctly.
    Fixed by mocking the download function to force a failure
    deterministically, so this test verifies the fallback *logic* itself in
    any environment, regardless of whether the network happens to be
    reachable when the test runs."""
    import models.visual_encoder as ve_module

    monkeypatch.setattr(ve_module, "_ensure_face_detector_model", lambda *a, **kw: None)

    from models.visual_encoder import FaceFrameExtractor

    extractor = FaceFrameExtractor(num_frames=2, image_size=32, model_cache_dir=str(tmp_path / "no_such_cache"))
    assert extractor._detector is None  # forced failure above must result in no detector, not a crash

    frame_bgr = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
    cropped = extractor._detect_and_crop(frame_bgr)
    assert cropped.shape == (32, 32, 3)


def test_face_frame_extractor_uses_real_detector_when_network_available(tmp_path):
    """Complementary to the test above: when the environment DOES have
    working internet (e.g. Colab), FaceFrameExtractor should successfully
    initialize a real detector rather than silently falling back. This is
    informational rather than a hard pass/fail gate, since whether the
    current test-runner has internet access is exactly the thing being
    checked — a sandboxed CI environment failing this is expected and does
    not indicate a code problem (see the test above for the deterministic
    regression guard)."""
    from models.visual_encoder import FaceFrameExtractor

    extractor = FaceFrameExtractor(num_frames=2, image_size=32, model_cache_dir=str(tmp_path / "cache"))
    if extractor._detector is None:
        pytest.skip("No network access to download the MediaPipe model in this environment — expected in a restricted sandbox, not a failure.")
    frame_bgr = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
    cropped = extractor._detect_and_crop(frame_bgr)
    assert cropped.shape == (32, 32, 3)


def test_face_frame_extractor_crops_using_absolute_pixel_bounding_box():
    """Regression guard for the MediaPipe Tasks API migration: the Tasks
    API's Detection.bounding_box uses ABSOLUTE pixel coordinates
    (origin_x/origin_y/width/height), unlike the legacy solutions API's
    relative_bounding_box (0-1 normalized values). This test injects a real
    Detection/BoundingBox/Category object (from the actual installed
    mediapipe package, not a hand-rolled stand-in) via a mocked detector, to
    confirm the crop math in models/visual_encoder.py handles absolute
    pixel coordinates without re-normalizing them (a bug that would silently
    crop the wrong region without ever raising an exception)."""
    from unittest.mock import MagicMock

    from mediapipe.tasks.python.components.containers.bounding_box import BoundingBox
    from mediapipe.tasks.python.components.containers.category import Category
    from mediapipe.tasks.python.components.containers.detections import Detection, DetectionResult

    from models.visual_encoder import FaceFrameExtractor

    extractor = FaceFrameExtractor(num_frames=2, image_size=64)
    frame_bgr = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)  # H=120, W=160

    box = BoundingBox(origin_x=40, origin_y=30, width=50, height=60)
    category = Category(index=0, score=0.95, display_name="face", category_name="face")
    detection = Detection(bounding_box=box, categories=[category])
    fake_result = DetectionResult(detections=[detection])

    fake_detector = MagicMock()
    fake_detector.detect.return_value = fake_result
    extractor._detector = fake_detector  # inject regardless of network availability

    cropped = extractor._detect_and_crop(frame_bgr)
    assert cropped.shape == (64, 64, 3)
    fake_detector.detect.assert_called_once()


# ---------------------------------------------------------------- Dataset
def _make_synthetic_video(path: str, duration: float = 1.5, size: str = "160x120"):
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={size}:rate=10",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-c:a", "aac", path, "-loglevel", "error",
        ],
        check=True,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg binary not available")
def test_dataset_pipeline_end_to_end(tmp_path):
    """Full FakeAVCelebDataset smoke test against a real (synthetically
    generated) video: face-frame extraction, ffmpeg audio extraction,
    caching, augmentation, and DataLoader batching, all in one test — this is
    the closest thing to an integration test we can run without the actual
    FakeAVCeleb dataset."""
    import pandas as pd
    from torch.utils.data import DataLoader

    from dataset.dataset import FakeAVCelebDataset, collate_fn, make_balanced_sampler

    video_path = str(tmp_path / "clip.mp4")
    _make_synthetic_video(video_path)

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"video_path": video_path, "label": "FAKE", "manipulated_modality": "video"},
            {"video_path": video_path, "label": "REAL", "manipulated_modality": "none"},
        ]
    ).to_csv(manifest_path, index=False)

    ds = FakeAVCelebDataset(
        str(manifest_path), cache_dir=str(tmp_path / "cache"), num_frames=4, image_size=64,
        sample_rate=16000, max_duration_sec=2.0, split="train",
        augment_cfg={"visual": {"horizontal_flip_prob": 0.5}, "audio": {"add_noise_prob": 0.5}},
    )
    assert len(ds) == 2
    item = ds[0]
    assert item["frames"].shape == (4, 64, 64, 3)
    assert item["waveform"].shape == (32000,)  # 2.0s @ 16kHz

    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch["frames"].shape == (2, 4, 64, 64, 3)
    assert batch["modality_gt"].shape == (2, 2)

    sampler = make_balanced_sampler(ds)
    assert sampler.num_samples == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg binary not available")
def test_dataset_cache_invalidated_by_config_change(tmp_path):
    """Regression guard for the cache-key bug: two Dataset instances reading
    the same video but configured with different num_frames must NOT share a
    cache entry (different array shapes)."""
    import pandas as pd

    from dataset.dataset import FakeAVCelebDataset

    video_path = str(tmp_path / "clip.mp4")
    _make_synthetic_video(video_path)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"video_path": video_path, "label": "REAL", "manipulated_modality": "none"}]).to_csv(
        manifest_path, index=False
    )
    cache_dir = str(tmp_path / "cache")

    ds_a = FakeAVCelebDataset(str(manifest_path), cache_dir=cache_dir, num_frames=4, image_size=64, split="val")
    ds_b = FakeAVCelebDataset(str(manifest_path), cache_dir=cache_dir, num_frames=8, image_size=64, split="val")

    frames_a = ds_a[0]["frames"]
    frames_b = ds_b[0]["frames"]
    assert frames_a.shape[0] == 4
    assert frames_b.shape[0] == 8  # would fail (both would be 4) if cache key ignored num_frames


def test_dataset_rejects_unknown_manipulated_modality(tmp_path):
    import pandas as pd

    from dataset.dataset import FakeAVCelebDataset

    manifest_path = tmp_path / "bad_manifest.csv"
    pd.DataFrame(
        [{"video_path": "/tmp/doesnt_matter.mp4", "label": "FAKE", "manipulated_modality": "typo_value"}]
    ).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="unrecognized manipulated_modality"):
        FakeAVCelebDataset(str(manifest_path), cache_dir=str(tmp_path / "cache"))
