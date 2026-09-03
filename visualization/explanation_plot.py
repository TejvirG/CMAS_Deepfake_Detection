"""
visualization/explanation_plot.py — small helper that renders a single
prediction's REAL/FAKE probabilities and (for multimodal FAKE predictions)
visual/audio contribution as a bar chart image, for the Gradio demo's
"explanation visualization" output. Kept separate from
visualization/visualization.py (which handles the batch/dataset-level
result plots) since this one operates on a single inference.py result dict.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_prediction_explanation(result: dict):
    """result: the dict returned by inference.predict_single(). Returns a
    matplotlib Figure (Gradio's gr.Plot output accepts this directly)."""
    has_contribution = result.get("visual_contribution_pct") is not None

    n_panels = 2 if has_contribution else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: REAL vs FAKE probability
    ax = axes[0]
    labels = ["REAL", "FAKE"]
    values = [result["real_probability"] * 100, result["fake_probability"] * 100]
    colors = ["#55A868", "#C44E52"]
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")
    threshold = result.get("decision_threshold_used", 0.5)
    ax.axhline(threshold * 100, color="gray", linestyle="--", linewidth=1, label=f"decision threshold ({threshold:.2f})")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"Prediction: {result['prediction']}")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2 (multimodal FAKE predictions only): visual vs audio contribution
    if has_contribution:
        ax2 = axes[1]
        labels2 = ["Visual", "Audio"]
        values2 = [result["visual_contribution_pct"], result["audio_contribution_pct"]]
        bars2 = ax2.bar(labels2, values2, color=["#DD8452", "#4C72B0"])
        for bar, v in zip(bars2, values2):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Contribution (%)")
        ax2.set_title("Modality contribution to FAKE prediction")

    fig.tight_layout()
    return fig
