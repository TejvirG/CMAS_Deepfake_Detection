"""
Cross-Modal Attribution Score (CMAS)

CMAS(sample) = cosine_similarity(explanation_vector, ground_truth_modality_vector)

Where:
  ground_truth_modality_vector (g) encodes which modality was manipulated:
      audio manipulated -> [0, 1]
      video manipulated -> [1, 0]
      both manipulated   -> [0.5, 0.5]
  explanation_vector (e) = [visual_importance, audio_importance], produced by
  an explanation method (Integrated Gradients or cross-attention mass; see
  explainability/). e is normalized to be non-negative and sum to 1 before
  computing cosine similarity, so CMAS in practice ranges [0, 1] for
  non-degenerate explanations (both e and g lie in the non-negative orthant).

By default (config: explainability.cmas_exclude_real), REAL samples are
excluded from CMAS aggregation because they have no manipulated modality to
attribute to (g = [0, 0], for which cosine similarity is undefined).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


def normalize_importance(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Clips a raw (possibly unnormalized, possibly negative) 2-element
    importance vector to non-negative and rescales it to sum to 1. Falls back
    to an uninformative [0.5, 0.5] if the input is all-zero/degenerate.
    Public: used by explainability/attribution.py, explainability/
    integrated_gradients.py callers, and inference.py — was previously a
    "private" `_normalize`, which is misleading naming for a function that's
    actually imported across three other modules."""
    vec = np.clip(vec, a_min=0.0, a_max=None)  # importances should be non-negative
    total = vec.sum()
    if total < eps:
        return np.array([0.5, 0.5])  # degenerate/uninformative explanation
    return vec / total


# Backward-compatible alias (kept in case any external/user code imported the
# old private name directly).
_normalize = normalize_importance


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cmas_single(explanation_vector: np.ndarray, ground_truth_vector: np.ndarray, normalize_explanation: bool = True) -> float:
    """Compute CMAS for a single sample.

    explanation_vector: [visual_importance, audio_importance] (any non-negative scale)
    ground_truth_vector: [visual_gt, audio_gt] e.g. [1,0], [0,1], [0.5,0.5]
    """
    e = np.asarray(explanation_vector, dtype=np.float64)
    g = np.asarray(ground_truth_vector, dtype=np.float64)
    if normalize_explanation:
        e = _normalize(e)
    if g.sum() <= 1e-8:
        raise ValueError(
            "Ground-truth modality vector is [0, 0] (REAL sample) — CMAS is undefined for REAL "
            "samples. Filter these out before calling cmas_single (see cmas_batch(..., exclude_real=True))."
        )
    return cosine_similarity(e, g)


@dataclass
class CMASResult:
    per_sample_scores: List[float]
    mean_cmas: float
    std_cmas: float
    n_samples: int
    n_excluded_real: int


def cmas_batch(
    explanation_vectors: Iterable[np.ndarray],
    ground_truth_vectors: Iterable[np.ndarray],
    exclude_real: bool = True,
) -> CMASResult:
    """Aggregate CMAS over a batch/dataset of samples.

    explanation_vectors, ground_truth_vectors: sequences of length-2 arrays,
    aligned sample-for-sample (e.g. gathered across a DataLoader pass).
    """
    scores: List[float] = []
    n_excluded = 0
    for e, g in zip(explanation_vectors, ground_truth_vectors):
        g_arr = np.asarray(g, dtype=np.float64)
        if g_arr.sum() <= 1e-8:
            if exclude_real:
                n_excluded += 1
                continue
            else:
                scores.append(0.0)  # undefined case, scored as 0 if explicitly not excluded
                continue
        scores.append(cmas_single(e, g_arr))

    arr = np.array(scores) if scores else np.array([0.0])
    return CMASResult(
        per_sample_scores=scores,
        mean_cmas=float(arr.mean()),
        std_cmas=float(arr.std()),
        n_samples=len(scores),
        n_excluded_real=n_excluded,
    )
