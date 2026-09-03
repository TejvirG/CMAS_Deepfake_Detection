"""
app.py — Gradio demo. Upload a video, get REAL/FAKE prediction, both class
probabilities, per-modality (visual/audio) contribution for FAKE
predictions, and a bar-chart visualization — computed by the actual trained
model (no hardcoded/fabricated outputs).

NOTE: no CMAS score is shown here. CMAS is defined against a ground-truth
manipulated-modality label that an arbitrary uploaded video doesn't have —
see inference.py's module docstring. Real CMAS (on labeled FakeAVCeleb data)
comes from evaluate.py / experiments/exp4_cmas_eval.py.

REVIEW NOTE: an earlier version of this file called predict_single(video,
checkpoint_path, cfg, device), which reloaded the checkpoint AND rebuilt the
whole model (EfficientNet + Wav2Vec2) from scratch on every single button
click — multi-second dead time per click for no reason, since the model
doesn't change between clicks. Fixed by loading the model once at import
time via inference.load_model() and reusing it across requests.

Usage:
    python app.py --checkpoint checkpoints/best_model_multimodal.pt
    python app.py --checkpoint checkpoints/best_model_multimodal.pt --threshold 0.3
    # then open the printed local URL (default http://127.0.0.1:7860)
"""
from __future__ import annotations

import argparse
import os

import gradio as gr
import yaml

from inference import load_model, predict_single
from utils.seed import get_device
from visualization.explanation_plot import render_prediction_explanation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_multimodal.pt")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument(
    "--threshold", type=float, default=None,
    help="Decision threshold on P(FAKE). Omit to use argmax (0.5). Run "
         "`evaluate.py --tune_threshold` first to find a better value for your checkpoint.",
)
parser.add_argument("--share", action="store_true")
args, _ = parser.parse_known_args()

with open(args.config) as f:
    CFG = yaml.safe_load(f)
DEVICE = get_device(CFG["device"]["auto_detect"])

# Loaded once at startup (see REVIEW NOTE above), not per-request. If the
# checkpoint doesn't exist yet (fresh clone, before running train.py), we
# defer the error to first use inside run_inference() with a clear message
# rather than crashing the whole app at import time.
MODEL = None
if os.path.exists(args.checkpoint):
    MODEL = load_model(args.checkpoint, CFG, DEVICE)


def run_inference(video_file):
    if video_file is None:
        return "No video provided.", "", "", "", "", None

    if MODEL is None:
        msg = f"No trained checkpoint found at {args.checkpoint}. Run train.py first (see README.md), then restart app.py."
        return msg, "", "", "", "", None

    result = predict_single(video_file, MODEL, CFG, DEVICE, threshold=args.threshold)

    prediction = result["prediction"]
    real_pct = f"{result['real_probability'] * 100:.1f}%"
    fake_pct = f"{result['fake_probability'] * 100:.1f}%"
    visual_pct = f"{result['visual_contribution_pct']}%" if result["visual_contribution_pct"] is not None else "N/A"
    audio_pct = f"{result['audio_contribution_pct']}%" if result["audio_contribution_pct"] is not None else "N/A"
    fig = render_prediction_explanation(result)

    return prediction, real_pct, fake_pct, visual_pct, audio_pct, fig


with gr.Blocks(title="CMAS Deepfake Detector") as demo:
    gr.Markdown(
        "# CMAS: Explainable Audio-Visual Deepfake Detection\n"
        "Upload a video clip. The model predicts REAL/FAKE, reports both class "
        "probabilities, and — for FAKE predictions from the multimodal model — which "
        "modality (visual/audio) drove the decision.\n\n"
        "*CMAS itself is a labeled-dataset evaluation metric (see evaluate.py) and is "
        "not reported here, since an arbitrary upload has no ground-truth modality label.*"
    )
    with gr.Row():
        video_input = gr.Video(label="Upload video")
    run_btn = gr.Button("Analyze", variant="primary")
    with gr.Row():
        prediction_out = gr.Textbox(label="Prediction")
        real_out = gr.Textbox(label="Real probability")
        fake_out = gr.Textbox(label="Fake probability")
    with gr.Row():
        visual_out = gr.Textbox(label="Visual contribution")
        audio_out = gr.Textbox(label="Audio contribution")
    plot_out = gr.Plot(label="Explanation visualization")

    run_btn.click(
        fn=run_inference,
        inputs=[video_input],
        outputs=[prediction_out, real_out, fake_out, visual_out, audio_out, plot_out],
    )

if __name__ == "__main__":
    demo.launch(share=args.share)
