"""
Modality-ablation attribution + Integrated Gradients + combined modality
contribution analysis.

Combines two attribution sources into a single explanation vector per sample:
  1. Ablation-based: `model.modality_ablation_importance()` — leave-one-
     modality-out ablation on the classifier logit (see models/fusion.py).
     Cheap (only re-runs the small fusion+classifier head, not the encoders),
     always available for the multimodal model. This method used to be
     called "attention-based" and read raw cross-attention weights directly;
     that was found during review to be constant regardless of input (see
     CrossAttentionFusion's docstring in models/fusion.py) and was replaced.
  2. Integrated Gradients: from explainability/integrated_gradients.py —
     more expensive, gradient-based.

`explain_batch` produces, per sample: visual_contribution_pct,
audio_contribution_pct, and the CMAS score against the sample's ground-truth
modality vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from explainability.integrated_gradients import batch_integrated_gradients_importance
from metrics.cmas import cmas_batch, cmas_single, normalize_importance


@dataclass
class SampleAttribution:
    video_path: str
    predicted_label: int
    true_label: int
    visual_contribution_pct: float
    audio_contribution_pct: float
    cmas_score: Optional[float]
    method: str


def ablation_modality_importance(model: nn.Module, visual_embed: torch.Tensor, audio_embed: torch.Tensor, target_class: torch.Tensor) -> np.ndarray:
    """Wraps model.modality_ablation_importance(). Returns (B, 2) array of
    [visual, audio] importance, already normalized to sum to 1 per sample."""
    mass = model.modality_ablation_importance(visual_embed, audio_embed, target_class)  # (B, 2), already normalized
    return mass.cpu().numpy()


# Backward-compatible alias for the old (misleading) name.
attention_modality_importance = ablation_modality_importance


def explain_batch(
    model: nn.Module,
    visual_embed: torch.Tensor,
    audio_embed: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    modality_gt: torch.Tensor,
    video_paths: List[str],
    method: str = "both",  # integrated_gradients | ablation | both  (also accepts legacy alias "attention")
    ig_n_steps: int = 50,
    cmas_exclude_real: bool = True,
) -> List[SampleAttribution]:
    """Produces per-sample modality contribution % and CMAS score for one
    batch. `method='both'` averages the (normalized) IG and ablation
    explanation vectors before scoring CMAS."""
    if method == "attention":  # legacy alias from before the rename
        method = "ablation"

    preds = logits.argmax(dim=-1).cpu().numpy()
    preds_t = torch.as_tensor(preds, device=logits.device)
    labels_np = labels.cpu().numpy()
    gt_np = modality_gt.cpu().numpy()

    explanation_vectors = np.zeros((len(video_paths), 2))

    if method in ("ablation", "both") and model.mode == "multimodal":
        ablation_importance = ablation_modality_importance(model, visual_embed, audio_embed, preds_t)
        explanation_vectors += ablation_importance if method == "ablation" else 0.5 * ablation_importance

    if method in ("integrated_gradients", "both"):
        ig_importance_raw = batch_integrated_gradients_importance(
            model, visual_embed, audio_embed, preds_t, n_steps=ig_n_steps,
        )
        ig_normalized = np.stack([normalize_importance(row) for row in ig_importance_raw])
        explanation_vectors += ig_normalized if method == "integrated_gradients" else 0.5 * ig_normalized

    # Re-normalize final combined vector per sample (rows already sum ~1 for
    # single-method cases; guards against drift when averaging two methods).
    row_sums = explanation_vectors.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-8] = 1.0
    explanation_vectors = explanation_vectors / row_sums

    cmas_scores = []
    for e, g in zip(explanation_vectors, gt_np):
        if g.sum() <= 1e-8:
            cmas_scores.append(None)  # REAL sample, CMAS undefined
        else:
            cmas_scores.append(cmas_single(e, g, normalize_explanation=False))

    results = []
    for i, path in enumerate(video_paths):
        results.append(
            SampleAttribution(
                video_path=path,
                predicted_label=int(preds[i]),
                true_label=int(labels_np[i]),
                visual_contribution_pct=float(explanation_vectors[i, 0] * 100),
                audio_contribution_pct=float(explanation_vectors[i, 1] * 100),
                cmas_score=cmas_scores[i],
                method=method,
            )
        )
    return results


def aggregate_cmas(results: List[SampleAttribution]) -> dict:
    """Aggregates a list of SampleAttribution (e.g. across a full eval split)
    into summary CMAS statistics via metrics.cmas.cmas_batch-style reduction."""
    valid_scores = [r.cmas_score for r in results if r.cmas_score is not None]
    if not valid_scores:
        return {"mean_cmas": None, "std_cmas": None, "n_samples": 0, "n_excluded_real": len(results)}
    arr = np.array(valid_scores)
    return {
        "mean_cmas": float(arr.mean()),
        "std_cmas": float(arr.std()),
        "n_samples": len(valid_scores),
        "n_excluded_real": len(results) - len(valid_scores),
    }
