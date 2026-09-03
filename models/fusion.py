"""
Fusion module combining visual and audio embeddings, plus the full
audio-visual deepfake classifier.

Two fusion strategies are implemented:
  - CrossAttentionFusion (preferred): visual and audio embeddings attend to
    each other via multi-head cross-attention before being concatenated and
    classified.
  - AttentionPoolFusion (lighter-weight fallback): a single learned gate over
    the two modality embeddings (attention-based feature fusion), useful when
    cross-attention is too expensive (e.g. CPU-only environments).

Neither fusion class exposes a "modality attention mass" any more — see the
REVIEW NOTE in CrossAttentionFusion's docstring for why that was removed.
Modality-importance attribution for CMAS is instead computed uniformly for
both fusion types by `CMASDeepfakeDetector.modality_ablation_importance()`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_encoder import AudioEncoder
from models.visual_encoder import VisualEncoder


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention: visual attends to audio and vice versa,
    then the two attended representations are concatenated and projected.

    REVIEW NOTE — two bugs found and fixed here during review, kept as a
    record since they're easy to reintroduce:

    (1) Because the visual and audio branches each collapse to a *single*
    pooled embedding before fusion (per the required architecture: "average
    frame embeddings" / one audio embedding per clip), `v2a_attn`/`a2v_attn`
    each attend over exactly one key/value token. Softmax over a single key
    is mathematically forced to output the constant 1.0 regardless of the
    query, so the raw attention *probabilities* returned by
    nn.MultiheadAttention here carry zero information about the input
    (empirically verified: identical [0.5, 0.5] "mass" for wildly different
    random inputs when this was exposed as get_modality_attention_mass()).

    (2) The first attempted fix substituted the L2 norm of the post-attention
    representations (v_out, a_out) as a proxy — but those pass through
    LayerNorm, which normalizes per-sample vector norms to a near-constant
    scale, so that proxy was *also* empirically constant across inputs.

    Fix actually applied: this class no longer exposes any modality-mass
    method at all. Modality importance is instead computed via leave-one-
    modality-out ablation on the actual classifier logit — see
    `CMASDeepfakeDetector.modality_ablation_importance()` — which is
    architecture-agnostic (works identically for both fusion classes) and is
    provably input-dependent since it reads real decision-function outputs
    rather than an intermediate representation's norm or a degenerate
    softmax. See paper_outline.md Limitations for the honest writeup of (1).
    """

    def __init__(self, visual_dim: int, audio_dim: int, hidden_dim: int = 512, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        self.v2a_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.a2v_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)

        self.output_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_embed: torch.Tensor, audio_embed: torch.Tensor) -> torch.Tensor:
        """visual_embed: (B, visual_dim), audio_embed: (B, audio_dim) -> (B, hidden_dim*2)"""
        v = self.visual_proj(visual_embed).unsqueeze(1)  # (B, 1, H)
        a = self.audio_proj(audio_embed).unsqueeze(1)    # (B, 1, H)

        # NOTE: with a single key/value token, attention weights returned
        # here are always exactly 1.0 and carry no signal — see class
        # docstring. Discarded deliberately.
        v_attended, _ = self.v2a_attn(query=v, key=a, value=a)  # visual attends to audio
        a_attended, _ = self.a2v_attn(query=a, key=v, value=v)  # audio attends to visual

        v_out = self.norm_v(v.squeeze(1) + v_attended.squeeze(1))
        a_out = self.norm_a(a.squeeze(1) + a_attended.squeeze(1))

        fused = torch.cat([v_out, a_out], dim=-1)
        return self.dropout(fused)


class AttentionPoolFusion(nn.Module):
    """Lightweight fallback: projects both modalities to a shared space and
    learns a scalar gate per modality (softmax over 2 values) to weight their
    contribution before concatenation. Cheaper than full cross-attention.

    Unlike CrossAttentionFusion's degenerate single-token attention (see that
    class's docstring), this gate is a genuine function of both modalities'
    content (softmax of an MLP over the concatenated projections) and *is*
    empirically input-dependent (verified in
    tests/test_smoke.py::test_attention_pool_gate_varies_with_input).
    `get_modality_attention_mass()` is kept here as a valid, cheap diagnostic,
    but the main CMAS pipeline uses `CMASDeepfakeDetector.
    modality_ablation_importance()` for both fusion types uniformly so that
    Experiment 4's results are computed the same way regardless of which
    fusion_type is configured — otherwise switching config.yaml's
    fusion_type would silently change what "attention-based attribution"
    even means, which would confound any cross-run comparison.
    """

    def __init__(self, visual_dim: int, audio_dim: int, hidden_dim: int = 512, dropout: float = 0.3, **kwargs):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.output_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self._last_gate_weights: Optional[torch.Tensor] = None

    def forward(self, visual_embed: torch.Tensor, audio_embed: torch.Tensor) -> torch.Tensor:
        v = self.visual_proj(visual_embed)
        a = self.audio_proj(audio_embed)
        gate_logits = self.gate(torch.cat([v, a], dim=-1))
        weights = F.softmax(gate_logits, dim=-1)  # (B, 2)
        self._last_gate_weights = weights.detach()
        fused = torch.cat([v * weights[:, 0:1], a * weights[:, 1:2]], dim=-1)
        return self.dropout(fused)

    def get_modality_attention_mass(self) -> torch.Tensor:
        """Diagnostic only — not used by the main CMAS pipeline (see class
        docstring). Returns the learned gate weights from the last forward()."""
        if self._last_gate_weights is None:
            raise RuntimeError("forward() must be called before get_modality_attention_mass().")
        return self._last_gate_weights


class CMASDeepfakeDetector(nn.Module):
    """Full audio-visual deepfake classifier: VisualEncoder + AudioEncoder +
    Fusion + classification head. Supports visual-only / audio-only ablation
    modes for Experiments 1 and 2."""

    def __init__(
        self,
        visual_backbone: str = "efficientnet_b0",
        visual_pretrained: bool = True,
        audio_model_name: str = "facebook/wav2vec2-base",
        fusion_type: str = "cross_attention",
        fusion_hidden_dim: int = 512,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.3,
        num_classes: int = 2,
        mode: str = "multimodal",  # multimodal | visual_only | audio_only
        freeze_backbones: bool = False,
    ):
        super().__init__()
        assert mode in {"multimodal", "visual_only", "audio_only"}
        self.mode = mode

        # Only build the encoder(s) this mode actually needs. The original
        # version of this constructor built both VisualEncoder AND
        # AudioEncoder unconditionally regardless of `mode` — so
        # mode="visual_only" (Experiment 1) still downloaded/loaded the full
        # Wav2Vec2 model it would never use, wasting memory and making the
        # visual-only ablation needlessly dependent on audio-model
        # availability (e.g. it would fail in a network-restricted
        # environment purely because of the unused audio branch).
        self.visual_encoder = VisualEncoder(visual_backbone, visual_pretrained, freeze=freeze_backbones) if mode != "audio_only" else None
        self.audio_encoder = AudioEncoder(audio_model_name, freeze=freeze_backbones) if mode != "visual_only" else None

        if mode == "multimodal":
            fusion_cls = CrossAttentionFusion if fusion_type == "cross_attention" else AttentionPoolFusion
            self.fusion = fusion_cls(
                visual_dim=self.visual_encoder.embed_dim,
                audio_dim=self.audio_encoder.embed_dim,
                hidden_dim=fusion_hidden_dim,
                num_heads=fusion_num_heads,
                dropout=fusion_dropout,
            )
            classifier_in = self.fusion.output_dim
        elif mode == "visual_only":
            self.fusion = None
            classifier_in = self.visual_encoder.embed_dim
        else:  # audio_only
            self.fusion = None
            classifier_in = self.audio_encoder.embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(256, num_classes),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Used for gradual unfreezing during training (see train.py).

        Toggles requires_grad AND train()/eval() mode on the encoder
        submodules. Setting requires_grad=False alone is not sufficient to
        "freeze" a backbone that contains BatchNorm layers (as EfficientNet
        does): BatchNorm updates its running mean/var from the current batch
        statistics during forward() any time the module is in .train() mode,
        regardless of requires_grad. Without the explicit eval() call here,
        a frozen backbone's BN running stats would keep drifting every epoch
        even though no gradient reaches its weights — silently defeating the
        point of freezing. Note this must be called *after* the outer
        `model.train()` in the training loop, since `model.train()` recursively
        re-enables train() on all submodules including these.
        """
        if self.visual_encoder is not None:
            self.visual_encoder.set_trainable(trainable)
            self.visual_encoder.backbone.train(trainable)
        if self.audio_encoder is not None:
            self.audio_encoder.set_trainable(trainable)
            self.audio_encoder.model.train(trainable)

    def encode(self, visual_frames: torch.Tensor, audio_input: torch.Tensor):
        visual_embed = self.visual_encoder(visual_frames) if self.mode != "audio_only" else None
        audio_embed = self.audio_encoder(audio_input) if self.mode != "visual_only" else None
        return visual_embed, audio_embed

    def classify_from_embeddings(self, visual_embed: Optional[torch.Tensor], audio_embed: Optional[torch.Tensor]) -> torch.Tensor:
        """Single source of truth for the mode-dependent fusion+classifier
        branching. `forward()`, `evaluate.py`, `inference.py`, and the
        Integrated Gradients wrapper all call this instead of re-implementing
        the if/elif chain, so a change to fusion logic can't silently drift
        out of sync between training and evaluation (this DRY refactor fixed
        exactly that kind of duplication in an earlier version of this repo)."""
        if self.mode == "multimodal":
            fused = self.fusion(visual_embed, audio_embed)
            return self.classifier(fused)
        elif self.mode == "visual_only":
            return self.classifier(visual_embed)
        else:
            return self.classifier(audio_embed)

    def forward(self, visual_frames: Optional[torch.Tensor] = None, audio_input: Optional[torch.Tensor] = None):
        """
        visual_frames: (B, T, 3, H, W) preprocessed frames (required unless mode == audio_only)
        audio_input: (B, num_samples) preprocessed waveform (required unless mode == visual_only)
        Returns: logits (B, num_classes)
        """
        visual_embed, audio_embed = self.encode(visual_frames, audio_input)
        return self.classify_from_embeddings(visual_embed, audio_embed)

    def modality_ablation_importance(
        self, visual_embed: torch.Tensor, audio_embed: torch.Tensor, target_class: torch.Tensor
    ) -> torch.Tensor:
        """Fast, architecture-agnostic modality-importance attribution via
        leave-one-modality-out ablation on the actual classifier logit for
        `target_class`:

            importance_visual = logit_target(both) - logit_target(audio-only, visual zeroed)
            importance_audio  = logit_target(both) - logit_target(visual-only, audio zeroed)

        A modality that matters more for the prediction causes a bigger drop
        in the target logit when it's zeroed out, so it gets a bigger
        importance score. Both terms are clamped to >= 0 (a modality whose
        removal *increases* the target logit contributed nothing positive)
        and normalized to sum to 1; if neither ablation reduces the logit at
        all (row sum ~0), falls back to an uninformative [0.5, 0.5] rather
        than dividing by ~0.

        Replaces an earlier "attention-based" method that read raw
        cross-attention probabilities / representation norms directly from
        the fusion module — both were found during review to be constant
        regardless of input (see CrossAttentionFusion's docstring). This
        method only needs 2 extra forward passes through the lightweight
        fusion+classifier head (not the expensive visual/audio encoders), so
        it stays cheap.

        Only defined for mode == 'multimodal'. Returns (B, 2) [visual, audio].
        """
        if self.mode != "multimodal":
            raise RuntimeError("modality_ablation_importance is only defined for the multimodal model.")

        with torch.no_grad():
            logits_full = self.classify_from_embeddings(visual_embed, audio_embed)
            logits_no_visual = self.classify_from_embeddings(torch.zeros_like(visual_embed), audio_embed)
            logits_no_audio = self.classify_from_embeddings(visual_embed, torch.zeros_like(audio_embed))

        batch_idx = torch.arange(visual_embed.shape[0], device=visual_embed.device)
        tc = target_class.to(visual_embed.device).long()

        full_t = logits_full[batch_idx, tc]
        no_visual_t = logits_no_visual[batch_idx, tc]
        no_audio_t = logits_no_audio[batch_idx, tc]

        visual_importance = (full_t - no_visual_t).clamp_min(0)
        audio_importance = (full_t - no_audio_t).clamp_min(0)

        stacked = torch.stack([visual_importance, audio_importance], dim=-1)
        row_sum = stacked.sum(dim=-1, keepdim=True)
        fallback = torch.full_like(stacked, 0.5)
        return torch.where(row_sum > 1e-6, stacked / row_sum.clamp_min(1e-6), fallback)
