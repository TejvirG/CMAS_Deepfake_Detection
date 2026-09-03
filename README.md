# CMAS: Cross-Modal Attribution Score for Explainable Audio-Visual Deepfake Detection

A multimodal (audio + video) deepfake detection system built around a novel explainability
metric — the **Cross-Modal Attribution Score (CMAS)** — that measures whether an explanation
method correctly identifies *which modality* was manipulated in a deepfake sample.

> ⚠️ **Honesty note on results**: This repository ships with model code, training
> scripts, and evaluation/visualization pipelines only. It does **not** ship with
> pre-computed accuracy/AUC/F1/CMAS numbers, because we do not have access to a GPU,
> the FakeAVCeleb dataset, or a multi-hour training budget in the environment that
> generated this repo. Every metric this project reports is **computed by running
> `train.py` / `evaluate.py` yourself** on real data. `results/` is created empty on
> purpose — running the pipeline populates it with `results.json`, plots, and tables
> from your actual run. See [`paper_outline.md`](paper_outline.md) for a results
> section template you fill in after training, not fabricated numbers.

---

## 1. What's in this repo

```
CMAS-Deepfake-Detection/
├── README.md
├── requirements.txt
├── verify_environment.py        # run after pip install to confirm the environment actually works
├── config.yaml                 # single source of truth for hyperparameters/paths
├── train.py                     # training loop (mixed precision, AdamW, scheduler, early stopping)
├── evaluate.py                  # computes Acc/Precision/Recall/F1/ROC-AUC on a split
├── inference.py                 # single-video inference (prediction + CMAS)
├── app.py                       # Gradio demo app
├── prepare_fakeavceleb.py       # dataset download/layout helper + manifest builder
│
├── models/
│   ├── visual_encoder.py        # EfficientNet-B0/B4 + MediaPipe face extraction
│   ├── audio_encoder.py         # Wav2Vec2-base embeddings
│   └── fusion.py                # cross-attention fusion + classifier head
│
├── dataset/
│   └── dataset.py                # FakeAVCeleb Dataset, balanced sampler, augmentation, caching
│
├── explainability/
│   ├── integrated_gradients.py   # Captum Integrated Gradients per modality
│   └── attribution.py            # modality-ablation attribution + CMAS computation
│
├── metrics/
│   └── cmas.py                   # CMAS metric (cosine similarity to ground-truth modality vector)
│
├── experiments/
│   ├── exp1_visual_only.py
│   ├── exp2_audio_only.py
│   ├── exp3_multimodal.py
│   ├── exp4_cmas_eval.py
│   └── run_all.py
│
├── visualization/
│   └── visualization.py          # ROC curves, confusion matrix, CMAS chart, attribution plots
│
├── utils/
│   ├── logging_utils.py
│   └── seed.py
│
├── paper_assets/
│   ├── architecture.py           # generates architecture.png (diagram, no fake numbers)
│   └── results_template.xlsx     # empty results table you fill in after running experiments
│
├── tests/
│   └── test_smoke.py             # fast CPU smoke tests with synthetic data (no dataset needed)
│
└── paper_outline.md
```

## 2. Quick start

### 2.1 Install

You also need the system `ffmpeg` binary on PATH (audio extraction shells out to it directly —
see the note at the top of `requirements.txt`). It's preinstalled on Colab; on a bare Ubuntu/
Debian box run `apt-get install -y ffmpeg` first, or `brew install ffmpeg` on macOS.

```bash
git clone https://github.com/TejvirG/CMAS_Deepfake_Detection.git
cd CMAS-Deepfake-Detection
pip install -r requirements.txt
```

GPU is auto-detected (`torch.cuda.is_available()`); the code runs on CPU too (slowly), which
is useful for the smoke tests in `tests/test_smoke.py`.

### 2.2 Verify the environment

Before running anything else, confirm every dependency actually works (not just "imports
without error" — this also runs real forward passes through the visual and audio encoders,
checks for the system `ffmpeg` binary, and confirms MediaPipe's face-detector model can be
fetched):

```bash
python verify_environment.py
```

`[FAIL]` lines must be fixed before proceeding; `[WARN]` lines (e.g. no GPU, or a
network-restricted download) are informational and won't block the pipeline, which is designed
to degrade gracefully rather than crash — see `models/visual_encoder.py`'s fallback-to-center-crop
behavior for an example.

### 2.3 Run the smoke test first (no dataset required)

This verifies every module imports and runs end-to-end on synthetic tensors/videos — including
several regression tests for bugs found during code review (see comments in
`tests/test_smoke.py`) — before you spend time on the real dataset. A couple of tests that need
to download pretrained Wav2Vec2 weights from HuggingFace are auto-skipped (not failed) in a
network-restricted environment; they'll run normally on Colab or any machine with internet
access:

```bash
python -m pytest tests/test_smoke.py -v
```

### 2.4 Get FakeAVCeleb

FakeAVCeleb is distributed under a research-use agreement — you must request access from the
dataset authors (see https://sites.google.com/view/fakeavcelebdataset). Once you have the raw
data:

```bash
python prepare_fakeavceleb.py \
    --raw_dir /path/to/FakeAVCeleb \
    --out_dir data/fakeavceleb \
    --val_ratio 0.15 --test_ratio 0.15
```

This builds `data/fakeavceleb/{train,val,test}_manifest.csv` with columns:
`video_path,label,manipulated_modality` (`label` ∈ {REAL, FAKE}; `manipulated_modality` ∈
{none, audio, video, both}), which is what `dataset/dataset.py` consumes. If your directory
layout differs from the official release, edit the parsing logic at the top of
`prepare_fakeavceleb.py` — it is intentionally kept simple and commented.

### 2.5 Configure

Edit `config.yaml` (paths, batch size, epochs, learning rate, etc.). Nothing is hardcoded in
the scripts — they all read from this file (override any key with `--config path.yaml` or
CLI flags, see `train.py --help`).

### 2.6 Train

```bash
python train.py --config config.yaml
```

Produces `checkpoints/best_model_multimodal.pt` (filename includes the mode: `best_model_{mode}.pt`, e.g. `best_model_visual_only.pt` for `--mode visual_only`), `logs/train.log`, and TensorBoard logs under `logs/tb/`.

### 2.7 Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_model_multimodal.pt --split test --config config.yaml
```

Writes `results/results.json` and `results/metrics_report.txt` with **real** Accuracy,
Precision, Recall, F1, and ROC-AUC computed on your test split.

### 2.8 Run the four experiments + CMAS table

```bash
python experiments/run_all.py --config config.yaml
```

This trains/evaluates the visual-only, audio-only, and multimodal models (Experiments 1–3),
then runs the CMAS explainability evaluation (Experiment 4) and writes
`results/experiment_comparison.csv` and `results/cmas_table.csv`.

### 2.9 Visualize

```bash
python visualization/visualization.py --results_dir results/
```

Saves `results/roc_curve.png`, `results/confusion_matrix.png`, `results/cmas_comparison.png`,
`results/attribution_examples.png`.

### 2.10 Inference on a single video

```bash
python inference.py --video path/to/clip.mp4 --checkpoint checkpoints/best_model_multimodal.pt
```

### 2.11 Demo app

```bash
python app.py
```

Launches a Gradio UI at `http://127.0.0.1:7860` where you upload a video and get prediction,
confidence, visual/audio contribution %, and CMAS score.

## 3. Google Colab — step-by-step fresh setup

This is the exact sequence for a brand-new Colab session (nothing pre-installed beyond Colab's
base image). Each step is its own cell.

**Step 1 — Turn on a GPU runtime** (do this before running any cells):
`Runtime → Change runtime type → Hardware accelerator → GPU → Save`.

**Step 2 — Clone and enter the repo:**
```python
!git clone https://github.com/TejvirG/CMAS_Deepfake_Detection.git
%cd CMAS-Deepfake-Detection
```

**Step 3 — Install dependencies:**
```python
!pip install -r requirements.txt -q
```
This should complete in well under a minute on Colab — `torch`/`torchvision` are already
preinstalled and satisfy the floors in `requirements.txt`, so pip won't touch them (see the
comments at the top of `requirements.txt` for why that matters: a naive `pip install torch`
pulls several GB of CUDA packages and risks replacing Colab's already-correct GPU build).
Everything else installs fresh.

**Step 4 — Restart the runtime.**
`Runtime → Restart session`. This is required: a couple of the packages `pip` just
installed/upgraded (notably anything that touches `numpy` or `protobuf`'s loaded C extensions)
won't take effect in the current Python process until it restarts. Skipping this step is the
most common cause of `pip install` "succeeding" but imports still failing right after.

**Step 5 — Verify the environment before touching real data:**
```python
%cd CMAS-Deepfake-Detection
!python verify_environment.py
```
Read the output. `[OK]` and `[WARN]` lines are both fine to proceed past — `[WARN]` covers
things like "no GPU detected" or "couldn't reach a model-download URL," which don't block the
pipeline (it degrades gracefully; see `models/visual_encoder.py`). Any `[FAIL]` line must be
fixed before continuing; the script tells you exactly what failed and how to fix it. Exit code
is 0 unless there's a real `[FAIL]`.

**Step 6 — Run the smoke tests:**
```python
!python -m pytest tests/test_smoke.py -v
```
All tests should pass on Colab (unlike a network-restricted CI environment, Colab can reach
HuggingFace, so none of the 23 tests should skip here). If something fails here, stop — resolve
it before running anything on the real dataset.

From here, continue with Section 2 above (getting FakeAVCeleb, building manifests, training) —
Colab's `train.py` call is identical to the local one:
```python
!python train.py --config config.yaml
```

`train.py` automatically detects `torch.cuda.is_available()` and uses mixed-precision training
(`torch.amp`, falling back to the older `torch.cuda.amp` namespace automatically on torch <2.3)
when a GPU is present.

**If your Colab session disconnects and you reconnect later:** you'll need to re-run Steps 2–4
(the runtime is a fresh VM), but your `data/`, `cache/`, `checkpoints/`, and `results/` are only
preserved if they live on mounted Google Drive rather than Colab's local disk — see Section 2's
note on this.

## 4. The CMAS metric

For a sample with ground-truth manipulated-modality vector `g` (one-hot for single-modality
fakes, `[0.5, 0.5]` for both-modalities fakes) and an explanation-derived importance vector
`e = [visual_importance, audio_importance]` (each in `[0, 1]`, normalized to sum to 1):

```
CMAS(sample) = cosine_similarity(e, g)
```

Implementation: `metrics/cmas.py`. `e` is produced two ways (both included, selectable via
config): (1) Integrated Gradients attribution magnitude per modality
(`explainability/integrated_gradients.py`), and (2) modality-ablation importance
(`explainability/attribution.py` / `CMASDeepfakeDetector.modality_ablation_importance()`) — the
drop in the predicted-class logit when each modality's embedding is zeroed out. CMAS ranges
from -1 to 1; 1 means the explanation points entirely at the correct manipulated modality.

> **Review note on method (2):** an earlier version of this repo computed modality importance
> from the fusion module's raw cross-attention weights. Because each modality is pooled to a
> single embedding before fusion, that attention is a softmax over exactly one key, which
> `torch` (correctly) always evaluates to `1.0` regardless of the input — so the resulting
> "importance" was silently constant (`[0.5, 0.5]` for every sample) and carried no signal. This
> was caught during review, verified empirically (see `tests/test_smoke.py`), and replaced with
> the ablation-based method above, which is architecture-agnostic and provably input-dependent.
> See `models/fusion.py`'s `CrossAttentionFusion` docstring and `paper_outline.md` §Limitations
> for the full writeup — this is exactly the kind of thing worth stating plainly in a paper
> rather than hiding.

## 5. Known limitations

- Real videos and "both-modality" fakes don't have a single manipulated modality to attribute
  to; CMAS is only meaningful for single-modality FAKE samples (see `paper_outline.md`
  §Limitations for the full discussion) — `metrics/cmas.py` excludes REAL samples by default.
- EfficientNet-B4 and full-length audio processing are memory-heavy; `config.yaml` defaults to
  EfficientNet-B0 + 3-second audio windows for Colab T4 compatibility. Switch to B4 in
  `config.yaml` if you have more VRAM.
- Modality-ablation importance (see §4 above) measures how much the *classifier* relies on each
  modality's fused representation — it is not a spatial/temporal localization method (it won't
  tell you which frames or which audio segment look fake), and it re-runs only the small
  fusion+classifier head, not a re-derivation from raw pixels/waveform.
- `dataset/dataset.py`'s `FaceFrameExtractor` (OpenCV + MediaPipe) and `AudioEncoder` are
  constructed once per `Dataset`, in the main process, before `DataLoader` forks worker
  subprocesses. MediaPipe's Tasks API is backed by a TFLite delegate (XNNPACK) that opens its
  own internal threads — forking a process with live threads is a well-known deadlock risk
  (child processes can inherit copies of mutexes that never get released, since the threads that
  would release them don't exist in the child). This was reproduced in practice on a Colab GPU
  runtime: `config.yaml`'s `num_workers` therefore now defaults to `0` (single-process data
  loading, no forking, no deadlock risk) rather than a default that can silently hang forever at
  `epoch 1/N: 0%` with no error message. If you want the throughput benefit of parallel data
  loading and are confident your environment doesn't hit this (e.g. you've disabled MediaPipe
  entirely, or you're on a platform using `spawn` instead of `fork` for multiprocessing), you can
  raise `num_workers` yourself — just know that's an explicit trade you're opting into, not the
  tested default.
- `FaceFrameExtractor` downloads MediaPipe's face-detector model asset (~230KB) on first use and
  caches it at `~/.cache/cmas_deepfake/mediapipe_models/`. On Colab this re-downloads every
  fresh session (that cache directory isn't on the persisted disk unless you're on mounted
  Drive), which is harmless but means the very first video processed in each session is a touch
  slower than the rest. If the download fails (blocked network, offline container), face
  cropping silently falls back to a full-frame center crop rather than raising — check
  `verify_environment.py`'s output if you want to know which mode you're actually running in
  before training.
- `experiments/run_all.py` trains three separate models end-to-end (visual-only, audio-only,
  multimodal) plus the CMAS evaluation — on Colab-class GPUs with the default `config.yaml`
  epoch counts this is realistically a multi-hour run, not a quick script. Pass `--epochs` to
  override for a faster/smoke run, or run one `experiments/expN_*.py` script at a time.
- No results are bundled. Train it, then fill in `paper_assets/results_template.xlsx` and
  `paper_outline.md` with your numbers.

## 6. License

Code released under the MIT License (see `LICENSE`). FakeAVCeleb itself has its own license —
respect the terms of the dataset's research-use agreement.
