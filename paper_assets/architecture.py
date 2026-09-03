"""
paper_assets/architecture.py — generates paper_assets/architecture.png, a
block diagram of the CMAS model architecture (visual branch, audio branch,
fusion, classifier, CMAS explainability head). Purely illustrative of the
*architecture* — contains no performance numbers.

Usage:
    python paper_assets/architecture.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def draw_box(ax, xy, w, h, text, color="#E8EEF7", fontsize=10, edgecolor="#2E4057"):
    box = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=edgecolor, facecolor=color,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def draw_arrow(ax, start, end, color="#2E4057"):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )


def main():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.3, 8)
    ax.axis("off")

    # Input
    draw_box(ax, (0.3, 6.9), 2.2, 0.7, "Input Video\n(audio + visual)", color="#F5F5F5")
    draw_box(ax, (5.0, 6.9), 2.2, 0.7, "Input Video\n(audio + visual)", color="#F5F5F5")

    # Visual branch
    draw_box(ax, (0.2, 5.6), 2.4, 0.8, "OpenCV frame\nsampling", color="#FDEBD0")
    draw_box(ax, (0.2, 4.4), 2.4, 0.8, "MediaPipe\nface detection", color="#FDEBD0")
    draw_box(ax, (0.2, 3.2), 2.4, 0.8, "EfficientNet-B0/B4\n(ImageNet pretrained)", color="#FADBD8")
    draw_box(ax, (0.2, 2.0), 2.4, 0.8, "Mean pool over\nsampled frames\n→ visual embedding", color="#FADBD8")

    # Audio branch
    draw_box(ax, (5.0, 5.6), 2.4, 0.8, "ffmpeg extraction\n16kHz mono", color="#D6EAF8")
    draw_box(ax, (5.0, 4.4), 2.4, 0.8, "Wav2Vec2-base\n(HuggingFace pretrained)", color="#D6EAF8")
    draw_box(ax, (5.0, 3.2), 2.4, 0.8, "Mean pool over\ntime → audio embedding", color="#D6EAF8")

    # Fusion
    draw_box(ax, (1.9, 0.9), 3.4, 0.9, "Cross-Attention Fusion\n(visual ↔ audio)", color="#D5F5E3")

    # Classifier
    draw_box(ax, (8.4, 3.6), 2.6, 0.8, "MLP Classifier", color="#E8DAEF")
    draw_box(ax, (8.4, 2.4), 2.6, 0.8, "REAL / FAKE", color="#E8DAEF")

    # Explainability
    draw_box(ax, (8.4, 1.2), 2.6, 0.9, "Integrated Gradients +\nAttention Attribution", color="#FCF3CF")
    draw_box(ax, (8.4, 0.0), 2.6, 0.9, "CMAS\n(cosine similarity to\nground-truth modality)", color="#FCF3CF")

    # Arrows: visual branch
    draw_arrow(ax, (1.4, 6.9), (1.4, 6.4))
    draw_arrow(ax, (1.4, 5.6), (1.4, 5.2))
    draw_arrow(ax, (1.4, 4.4), (1.4, 4.0))
    draw_arrow(ax, (1.4, 3.2), (1.4, 2.8))

    # Arrows: audio branch
    draw_arrow(ax, (6.2, 6.9), (6.2, 6.4))
    draw_arrow(ax, (6.2, 5.6), (6.2, 5.2))
    draw_arrow(ax, (6.2, 4.4), (6.2, 4.0))

    # Embeddings into fusion
    draw_arrow(ax, (1.4, 2.0), (2.8, 1.8))
    draw_arrow(ax, (6.2, 3.2), (4.4, 1.8))

    # Fusion into classifier
    draw_arrow(ax, (5.3, 1.35), (8.4, 4.0))
    draw_arrow(ax, (9.7, 3.6), (9.7, 3.2))
    draw_arrow(ax, (9.7, 2.4), (9.7, 2.1))
    draw_arrow(ax, (9.7, 1.2), (9.7, 0.9))

    ax.set_title("CMAS: Cross-Modal Attribution Score — Model Architecture", fontsize=13, fontweight="bold")

    os.makedirs("paper_assets", exist_ok=True)
    out_path = os.path.join("paper_assets", "architecture.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
