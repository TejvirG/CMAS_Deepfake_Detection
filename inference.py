"""
inference.py — runs a trained model on a single video file: prediction,
both class probabilities, and per-modality contribution (visual vs audio)
for FAKE predictions.

Usage:
    python inference.py --video path/to/clip.mp4 --checkpoint checkpoints/best_model_multimodal.pt
    python inference.py --video path/to/clip.mp4 --checkpoint checkpoints/best_model_multimodal.pt --threshold 0.3

`load_model()` and `predict_single()` are split apart specifically so
long-running callers (e.g. app.py's Gradio demo) can load the checkpoint and
build the model ONCE and reuse it across many predictions, instead of
re-loading EfficientNet + Wav2Vec2 weights from disk on every single call —
see the REVIEW NOTE in app.py for the bug this fixes.

NOTE on CMAS: this file intentionally does NOT report a CMAS score for
arbitrary uploaded videos. CMAS is defined against a ground-truth manipulated-
modality label (see metrics/cmas.py), which an arbitrary upload doesn't have
— reporting a number labeled "CMAS" here would misrepresent what's actually
being measured. Real CMAS, computed against FakeAVCeleb's labels, stays in
evaluate.py / experiments/exp4_cmas_eval.py. This file reports visual/audio
contribution % only, which is well-defined without a ground-truth label.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml

from explainability.integrated_gradients import integrated_gradients_modality_importance
from metrics.cmas import normalize_importance
from models.fusion import CMASDeepfakeDetector
from models.audio_encoder import extract_audio_ffmpeg, pad_or_trim
from models.visual_encoder import FaceFrameExtractor
from train import build_model
from utils.seed import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single video.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold on P(FAKE). If omitted, uses argmax (equivalent to 0.5). "
             "Run `evaluate.py --tune_threshold` on your checkpoint first to find a better "
             "value for your model, then pass it here — see metrics/threshold.py.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> CMASDeepfakeDetector:
    """Loads a checkpoint into a freshly-built model, once. Callers that will
    run many predictions (app.py) should call this a single time and pass the
    result to predict_single() repeatedly, rather than calling
    predict_single(..., checkpoint_path=...) directly each time."""
    # weights_only=False: safe here because these checkpoints are self-generated
    # by train.py in this same repo, not arbitrary/untrusted files. See the
    # matching comment in evaluate.py for the full explanation (PyTorch >=2.6
    # defaults to weights_only=True, which is too strict for our config/
    # val_metrics payload and was reproduced failing on this exact checkpoint format).
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mode = ckpt.get("mode", "multimodal")
    model = build_model(cfg, mode, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_single(video_path: str, model: CMASDeepfakeDetector, cfg: dict, device: torch.device, threshold: float = None) -> dict:
    """Runs inference on one video using an already-loaded `model` (see
    load_model()). This function itself does zero disk/network I/O for model
    weights — only the video's frames/audio are read from disk.

    threshold: decision threshold on P(FAKE). None -> argmax (equiv. to 0.5).
    Passing a validation-tuned threshold (see evaluate.py --tune_threshold)
    does not change the model's probability estimates, only which side of
    them counts as REAL vs FAKE — no retraining involved.
    """
    mode = model.mode

    face_extractor = FaceFrameExtractor(
        num_frames=cfg["model"]["num_frames"],
        image_size=224,
    )
    frames_np = face_extractor.extract(video_path)  # (T,H,W,3) uint8
    frames = torch.from_numpy(frames_np).unsqueeze(0).to(device)  # (1,T,H,W,3)

    waveform_np = extract_audio_ffmpeg(video_path, sample_rate=cfg["audio"]["sample_rate"], mono=True)
    waveform_np = pad_or_trim(waveform_np, cfg["audio"]["sample_rate"], cfg["audio"]["max_duration_sec"])

    with torch.no_grad():
        visual_input = model.visual_encoder.preprocess(frames) if mode != "audio_only" else None
        audio_input = (
            model.audio_encoder.preprocess(waveform_np[None, :], sample_rate=cfg["audio"]["sample_rate"]).to(device)
            if mode != "visual_only"
            else None
        )
        visual_embed, audio_embed = model.encode(visual_input, audio_input)
        # classify_from_embeddings is the same single source of truth used by
        # forward()/evaluate.py, instead of a hand-duplicated fusion/mode
        # if/elif chain (that duplication was found and removed during review).
        logits = model.classify_from_embeddings(visual_embed, audio_embed)

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        real_prob = float(probs[0].item())
        fake_prob = float(probs[1].item())

        if threshold is None:
            pred_idx = int(probs.argmax().item())
        else:
            pred_idx = 1 if fake_prob >= threshold else 0
        pred_label = ["REAL", "FAKE"][pred_idx]

    result = {
        "video": video_path,
        "prediction": pred_label,
        "real_probability": round(real_prob, 4),
        "fake_probability": round(fake_prob, 4),
        "decision_threshold_used": threshold if threshold is not None else 0.5,
    }

    if mode == "multimodal" and pred_label == "FAKE":
        pred_idx_t = torch.tensor([pred_idx], device=device)
        # Ablation-based contribution (cheap, always computed; see
        # models/fusion.py's modality_ablation_importance docstring — this
        # replaced an earlier, broken "attention-based" method that returned
        # a constant regardless of input).
        ablation_mass = model.modality_ablation_importance(visual_embed, audio_embed, pred_idx_t).squeeze(0).cpu().numpy()

        # Integrated Gradients contribution (more expensive, gradient-based)
        with torch.enable_grad():
            ve = visual_embed.clone().detach().requires_grad_(True)
            ae = audio_embed.clone().detach().requires_grad_(True)
            v_imp, a_imp = integrated_gradients_modality_importance(
                model, ve.squeeze(0), ae.squeeze(0), target_class=pred_idx,
                n_steps=cfg["explainability"]["ig_n_steps"],
            )
        ig_mass = normalize_importance(np.array([v_imp, a_imp]))

        combined = normalize_importance(0.5 * ablation_mass + 0.5 * ig_mass)
        result["visual_contribution_pct"] = round(float(combined[0]) * 100, 1)
        result["audio_contribution_pct"] = round(float(combined[1]) * 100, 1)
    else:
        result["visual_contribution_pct"] = None
        result["audio_contribution_pct"] = None
        if mode != "multimodal":
            result["note"] = f"Model mode is '{mode}' — per-modality contribution requires the multimodal checkpoint."

    return result


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = get_device(cfg["device"]["auto_detect"])
    model = load_model(args.checkpoint, cfg, device)
    result = predict_single(args.video, model, cfg, device, threshold=args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
