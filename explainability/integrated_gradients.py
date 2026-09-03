"""
Integrated Gradients attribution (via Captum) computed w.r.t. the visual and
audio embedding vectors, used to derive a per-modality importance score for
CMAS.

We attribute the predicted-class logit back to the *fused input embeddings*
(visual_embed, audio_embed) rather than raw pixels/waveform samples — this is
both cheaper and directly gives a per-modality scalar (sum of |attribution|
per embedding) needed for CMAS, without an extra aggregation step over pixel-
or sample-level attributions.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients


class _EmbeddingLevelWrapper(nn.Module):
    """Wraps the fusion+classifier head so Captum can attribute w.r.t. the
    two embedding tensors directly (Captum requires a single forward(*inputs)
    -> tensor signature)."""

    def __init__(self, model: nn.Module, target_class: int):
        super().__init__()
        self.model = model
        self.target_class = target_class

    def forward(self, visual_embed: torch.Tensor, audio_embed: torch.Tensor) -> torch.Tensor:
        # Delegates to the same classify_from_embeddings() used by
        # forward()/evaluate.py/inference.py, so IG attributes w.r.t. exactly
        # the function actually used at inference time, not a
        # separately-maintained copy of the mode-branching logic (an earlier
        # version of this file duplicated that branching here).
        return self.model.classify_from_embeddings(visual_embed, audio_embed)


def integrated_gradients_modality_importance(
    model: nn.Module,
    visual_embed: torch.Tensor,
    audio_embed: torch.Tensor,
    target_class: int,
    n_steps: int = 50,
) -> Tuple[float, float]:
    """
    Computes Integrated Gradients attributions for a single sample's visual
    and audio embeddings w.r.t. the target class logit, then reduces each
    modality's attribution to a scalar importance via sum of absolute values.

    Returns: (visual_importance, audio_importance) — non-negative floats,
    NOT yet normalized to sum to 1 (normalization happens in metrics/cmas.py
    or attribution.py's aggregate helper).
    """
    model.eval()
    wrapper = _EmbeddingLevelWrapper(model, target_class)
    ig = IntegratedGradients(wrapper)

    v = visual_embed.unsqueeze(0).clone().detach().requires_grad_(True)
    a = audio_embed.unsqueeze(0).clone().detach().requires_grad_(True)

    baselines = (torch.zeros_like(v), torch.zeros_like(a))

    attributions = ig.attribute(
        inputs=(v, a),
        baselines=baselines,
        target=target_class,
        n_steps=n_steps,
    )
    v_attr, a_attr = attributions
    visual_importance = v_attr.abs().sum().item()
    audio_importance = a_attr.abs().sum().item()
    return visual_importance, audio_importance


def batch_integrated_gradients_importance(
    model: nn.Module,
    visual_embeds: torch.Tensor,
    audio_embeds: torch.Tensor,
    target_classes: torch.Tensor,
    n_steps: int = 50,
) -> np.ndarray:
    """Runs IG per-sample over a batch (Captum's batched IG over a scalar
    target-per-sample is more fragile than looping for small eval batches, so
    we loop explicitly here for clarity/robustness).

    Returns: (B, 2) array of [visual_importance, audio_importance] rows.
    """
    results = []
    for i in range(visual_embeds.shape[0]):
        vi, ai = integrated_gradients_modality_importance(
            model, visual_embeds[i], audio_embeds[i], int(target_classes[i].item()), n_steps=n_steps
        )
        results.append([vi, ai])
    return np.array(results)
