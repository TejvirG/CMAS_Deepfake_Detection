"""
metrics/threshold.py — decision-threshold tuning, computed WITHOUT touching
any model weights.

Context: evaluating the multimodal/visual/audio checkpoints with the naive
argmax (equivalent to a 0.5 probability threshold) decision rule produced
precision=1.0 with mediocre recall/accuracy on all three models, while
ROC-AUC stayed decent-to-strong (0.74-0.99). That combination — good AUC,
poor accuracy at the default threshold — is the signature of a
well-calibrated *ranking* model whose default decision boundary is a bad
fit, not a broken model. Root cause traced to config.yaml having BOTH
balanced sampling (which already equalizes REAL/FAKE exposure per batch)
AND ~42x inverse-frequency class-weighted loss active simultaneously during
training, which pushed the model to become unusually conservative about
predicting FAKE. That's a training-time bias, not something a validation-
tuned threshold can perfectly undo — but since the AUC shows the underlying
probability estimates ARE discriminative, picking a better operating point
recovers most of the practical accuracy without retraining anything.

REVIEW FIX #2: the original version only supported F1-maximizing threshold
search. On the audio-only model (the weakest of the three, ROC-AUC ~0.79),
F1-maximization found a THRESHOLD SO LOW IT PREDICTED "FAKE" FOR EVERY
SINGLE SAMPLE (confusion matrix [[0,75],[0,3080]] — zero REAL samples
correctly identified). This scored ~98.8% F1 purely because ~97.6% of the
test set genuinely is FAKE — a trivial majority-class classifier, not a
working detector, and F1 alone doesn't penalize it because F1 (as commonly
computed here, w.r.t. the positive/FAKE class) is insensitive to how badly
the negative/REAL class is doing when the negative class is rare. Fixed by
adding a 'youden' objective (Youden's J statistic = TPR - FPR = sensitivity
+ specificity - 1, equivalent to 2*balanced_accuracy - 1), which explicitly
penalizes ignoring either class and cannot be gamed by an always-predict-
majority-class threshold: an all-FAKE prediction gives specificity=0,
driving J to a low score rather than a high one. 'youden' is now the
default. A degeneracy check also now runs regardless of which metric is
used, and its result is returned as a field on ThresholdResult (not just a
console warning) specifically so it can't be missed in a large log dump —
that's exactly how the audio degeneracy went unnoticed originally.

This module intentionally does not depend on torch/dataset — it is pure
numpy so it can be reused by evaluate.py (tunes on val, applies to test) and
later, if wanted, by an interactive tool.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.metrics import f1_score, recall_score


@dataclass
class ThresholdResult:
    threshold: float
    metric_name: str
    metric_value_at_threshold: float
    n_thresholds_tried: int
    is_degenerate: bool  # True if the resulting predictions ignore one class entirely
    recall_real_at_threshold: float  # a.k.a. specificity w.r.t. FAKE-as-positive
    recall_fake_at_threshold: float


def _score_at_threshold(labels_arr: np.ndarray, probs_arr: np.ndarray, t: float, metric: str) -> float:
    preds = (probs_arr >= t).astype(int)
    if metric == "f1":
        return f1_score(labels_arr, preds, zero_division=0)
    elif metric == "youden":
        recall_fake = recall_score(labels_arr, preds, pos_label=1, zero_division=0)
        recall_real = recall_score(labels_arr, preds, pos_label=0, zero_division=0)
        return recall_fake + recall_real - 1.0  # Youden's J
    else:
        raise NotImplementedError(f"Unknown metric '{metric}'; use 'f1' or 'youden'.")


def find_best_threshold(
    labels: Sequence[int],
    probs: Sequence[float],
    metric: str = "youden",
    n_steps: int = 199,
) -> ThresholdResult:
    """Sweeps decision thresholds on P(FAKE) over (0, 1) and returns the one
    that maximizes the chosen metric.

    metric='youden' (default): Youden's J = recall(FAKE) + recall(REAL) - 1.
    Robust to class imbalance — cannot be maximized by a degenerate
    always-predict-one-class threshold. Use this unless you have a specific
    reason to want raw F1.

    metric='f1': maximizes F1 w.r.t. the FAKE class specifically. Kept
    available for comparison, but see the module docstring for why this
    produced a degenerate all-FAKE threshold on the audio-only model in
    practice — always check `.is_degenerate` on the result before trusting it.

    IMPORTANT: call this on a VALIDATION split's (labels, probs), never on
    the test split — the whole point of a validation-tuned threshold is that
    the number being reported for test is not itself the thing that was
    optimized. evaluate.py enforces this by construction (see --tune_threshold).
    """
    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)

    if len(set(labels_arr.tolist())) < 2:
        raise ValueError(
            "Cannot tune a threshold on a split with only one class present. "
            "Check check_class_distribution.py output for this split."
        )

    # Extended low-end range (see module docstring, REVIEW FIX #1): a fixed
    # floor of 0.01 previously clipped the true optimum on models whose
    # P(FAKE) outputs are heavily compressed toward 0.
    low_range = np.geomspace(1e-4, 0.05, n_steps // 2)
    mid_high_range = np.linspace(0.05, 0.99, n_steps - len(low_range))
    candidate_thresholds = np.unique(np.concatenate([low_range, mid_high_range]))

    best_threshold = 0.5
    best_score = -np.inf
    for t in candidate_thresholds:
        score = _score_at_threshold(labels_arr, probs_arr, t, metric)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    final_preds = (probs_arr >= best_threshold).astype(int)
    recall_fake = float(recall_score(labels_arr, final_preds, pos_label=1, zero_division=0))
    recall_real = float(recall_score(labels_arr, final_preds, pos_label=0, zero_division=0))
    # Near-zero counts as degenerate too, not just exact zero — a threshold
    # that correctly identifies 2 REAL samples out of 75 (2.7% recall) is
    # not meaningfully different from identifying none; it's still a
    # majority-class classifier in practice, just not quite literally 0/N.
    _DEGENERATE_RECALL_FLOOR = 0.05
    is_degenerate = recall_fake < _DEGENERATE_RECALL_FLOOR or recall_real < _DEGENERATE_RECALL_FLOOR

    if is_degenerate:
        warnings.warn(
            f"Threshold {best_threshold:.6f} (metric={metric}) is DEGENERATE: it predicts only "
            f"one class for every sample (recall_real={recall_real:.3f}, recall_fake={recall_fake:.3f}). "
            f"This is not a working decision boundary — do not report metrics computed at this "
            f"threshold. See ThresholdResult.is_degenerate."
        )
    if best_threshold <= low_range[1]:
        warnings.warn(
            f"Best threshold ({best_threshold:.6f}) is at or near the search floor again. "
            f"The model's P(FAKE) outputs may be extremely compressed near 0/1 — worth "
            f"checking the raw probability distribution directly rather than trusting this "
            f"threshold blindly."
        )

    return ThresholdResult(
        threshold=best_threshold,
        metric_name=metric,
        metric_value_at_threshold=float(best_score),
        n_thresholds_tried=len(candidate_thresholds),
        is_degenerate=is_degenerate,
        recall_real_at_threshold=recall_real,
        recall_fake_at_threshold=recall_fake,
    )


def apply_threshold(probs: Sequence[float], threshold: float) -> List[int]:
    """Applies a fixed decision threshold to P(FAKE) values, returning 0/1
    predictions consistent with LABEL2IDX = {"REAL": 0, "FAKE": 1}."""
    return [1 if p >= threshold else 0 for p in probs]
