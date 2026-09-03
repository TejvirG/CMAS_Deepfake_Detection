# CMAS: Cross-Modal Attribution Score for Explainable Audio-Visual Deepfake Detection

*Paper outline / skeleton. Sections marked `[FILL IN AFTER RUNNING EXPERIMENTS]` must be
completed with real numbers produced by `train.py`, `evaluate.py`, and `experiments/run_all.py`
— do not fill them with invented values.*

---

## Abstract

*(150–250 words)*

Deepfake detectors typically output a binary REAL/FAKE decision without indicating **which**
modality — the visual stream, the audio stream, or both — was manipulated, and without
verifying that their explanations actually point at the correct evidence. We introduce the
**Cross-Modal Attribution Score (CMAS)**, a metric that measures the alignment between an
explanation method's predicted modality importance and the ground-truth manipulated modality,
computed as the cosine similarity between a two-dimensional explanation vector
`[visual_importance, audio_importance]` and a ground-truth modality vector (`[1,0]` for
visual-only fakes, `[0,1]` for audio-only fakes, `[0.5,0.5]` for both). We pair CMAS with a
multimodal deepfake detector combining an EfficientNet visual encoder, a Wav2Vec2 audio
encoder, and a cross-attention fusion module, evaluated on FakeAVCeleb.
`[FILL IN AFTER RUNNING EXPERIMENTS: one-sentence headline result, e.g. "Our multimodal
detector achieves X% accuracy / Y AUC, and CMAS evaluation shows explanation method Z aligns
most closely with ground-truth manipulated modality (mean CMAS = W)."]`

## 1. Introduction

- Motivation: deepfakes increasingly manipulate audio and video independently (voice cloning,
  face-swap, lip-sync re-dubbing); detectors need to both **classify** and **explain**.
- Gap: existing explainability work for deepfake detection (Grad-CAM style saliency, attention
  visualization) is largely qualitative — there is no standard *quantitative* measure of
  whether an explanation attributes a fake to the correct modality.
- Contribution list:
  1. CMAS — a lightweight, dataset-label-driven metric for cross-modal explanation fidelity.
  2. A cross-attention audio-visual fusion architecture with two attribution mechanisms
     (Integrated Gradients, modality-ablation importance) evaluated against CMAS.
  3. An open-source, reproducible pipeline (this repository) with caching, balanced sampling,
     and ablations (visual-only / audio-only / multimodal).

## 2. Related Work

- **Deepfake detection**: single-modality (image/video) CNN detectors; audio anti-spoofing
  models; multimodal fusion approaches for audio-visual deepfakes (cite FakeAVCeleb paper,
  audio-visual sync-based detectors, cross-modal consistency methods).
- **Explainable AI for deepfakes**: saliency maps (Grad-CAM, Integrated Gradients) applied to
  single-modality detectors; attention visualization in transformer-based detectors.
- **Cross-modal attribution / evaluation metrics**: metrics that connect model explanations to
  ground truth (pointing game, localization metrics in vision) — CMAS extends this idea to the
  *modality* level rather than the *spatial* level.
- **Positioning**: to our knowledge, no prior work defines a quantitative ground-truth-driven
  metric for whether a multimodal deepfake explanation identifies the correct manipulated
  modality; CMAS fills this gap.

## 3. Methodology

### 3.1 Problem formulation

Binary classification REAL vs FAKE from a video with paired audio+visual streams, plus (for
FAKE samples) the manipulated-modality label used only for evaluation, not as a training
signal.

### 3.2 Model architecture

- **Visual branch**: OpenCV frame sampling → MediaPipe face detection/cropping → EfficientNet-
  B0/B4 (ImageNet-pretrained) → mean-pooled frame embeddings.
- **Audio branch**: ffmpeg extraction (16kHz mono) → Wav2Vec2-base (HuggingFace-pretrained) →
  mean-pooled hidden states.
- **Fusion**: bidirectional cross-attention between the projected visual and audio embeddings,
  concatenated and passed to an MLP classifier. An attention-pooling fallback is also
  implemented for lower-compute settings.
- **Training**: mixed-precision AdamW with discriminative learning rates (lower LR for
  pretrained backbones), cosine LR schedule with warmup, gradual unfreezing (backbones frozen
  for the first `N` epochs), class-weighted loss, early stopping on validation F1.

*See `models/visual_encoder.py`, `models/audio_encoder.py`, `models/fusion.py`, `train.py`.*

### 3.3 CMAS formulation

For a FAKE sample with ground-truth manipulated-modality vector `g ∈ {[1,0], [0,1], [0.5,0.5]}`
and an explanation vector `e = [visual_importance, audio_importance]` (normalized to the
non-negative simplex), we define:

```
CMAS(sample) = cosine_similarity(e, g) = (e·g) / (‖e‖ ‖g‖)
```

`e` is produced two ways:
1. **Modality-ablation**: the drop in the predicted-class logit when each modality's embedding
   is zeroed out before the fusion+classifier head (cheap — only the small fusion+classifier
   is re-run, not the encoders — and always available for the multimodal model).
   *Design note, stated plainly for the reader rather than glossed over:* our first
   implementation of this attribution instead read the fusion module's raw cross-attention
   weights. Because each modality is pooled to a single embedding before fusion, the
   cross-attention softmax is taken over exactly one key, which is mathematically forced to
   output `1.0` regardless of the query — so that signal was constant across every sample and
   uninformative. A second attempt substituted the L2 norm of the post-attention
   representation, which turned out to be equally constant because that representation passes
   through LayerNorm (which normalizes per-sample vector norms to a near-fixed scale). Both
   failures were caught empirically (identical output across deliberately varied random inputs)
   before being reported anywhere, and the ablation-based method above — which reads the actual
   classifier decision function rather than an intermediate representation — was verified to be
   genuinely input-dependent (see `tests/test_smoke.py`) before being adopted. We include this
   account because a metric this central to the paper's contribution should be described
   accurately, including the dead ends.
2. **Integrated Gradients**: Captum IG attribution of the predicted-class logit back to the
   visual and audio embedding vectors, reduced to a scalar per modality via sum of absolute
   attribution (gradient-based, more expensive, arguably more faithful to model behavior).

CMAS is undefined for REAL samples (`g = [0,0]`) and excluded from aggregate statistics by
default. CMAS ∈ [0, 1] given both `e`, `g` are non-negative; 1 indicates the explanation points
entirely at the truly manipulated modality, 0 indicates it points entirely at the wrong one
(e.g. an audio-only fake explained as fully visual).

*See `metrics/cmas.py`, `explainability/integrated_gradients.py`,
`explainability/attribution.py`.*

## 4. Experiments

- **Dataset**: FakeAVCeleb, identity-disjoint train/val/test split (`prepare_fakeavceleb.py`),
  balanced sampling to counter class imbalance, feature caching for repeated epochs.
- **Experiment 1 — Visual-only ablation**: EfficientNet branch + classifier only.
- **Experiment 2 — Audio-only ablation**: Wav2Vec2 branch + classifier only.
- **Experiment 3 — Multimodal**: full cross-attention fusion model.
- **Experiment 4 — CMAS evaluation**: for the multimodal model, compute CMAS on the test split
  using each attribution method (ablation, Integrated Gradients, both averaged), reporting
  mean/std CMAS per method.
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC for classification (Experiments 1–3);
  mean/std CMAS per attribution method for explainability (Experiment 4).

`[Reproduce with: python experiments/run_all.py --config config.yaml]`

## 5. Results

`[FILL IN AFTER RUNNING EXPERIMENTS — do not fabricate. Populate from results/experiment_comparison.csv and results/cmas_table.csv]`

### 5.1 Classification performance

| Method | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Visual-only | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| Audio-only | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| Multimodal (CMAS) | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

*(source: `results/experiment_comparison.csv`, generated by `experiments/run_all.py`)*

### 5.2 CMAS by attribution method

| Method | Mean CMAS | Std CMAS | N samples |
|---|---|---|---|
| Ablation-based | `TBD` | `TBD` | `TBD` |
| Integrated Gradients | `TBD` | `TBD` | `TBD` |
| Both (averaged) | `TBD` | `TBD` | `TBD` |

*(source: `results/cmas_table.csv`, generated by `experiments/exp4_cmas_eval.py`)*

### 5.3 Qualitative examples

`[Insert 3-5 example predictions with visual/audio contribution % and CMAS score from
results/results.json → cmas_per_sample, plus the corresponding attribution plot from
visualization/visualization.py (results/attribution_examples.png)]`

## 6. Discussion

`[FILL IN: interpret the results once available — e.g. does the multimodal model outperform
single-modality ablations? does one attribution method align with ground truth better than the
other, and does that make sense given how each is computed?]`

## 7. Limitations

- CMAS is only meaningful for samples with a single well-defined manipulated modality; for
  REAL samples and ambiguous/partial manipulations it is either undefined or only a rough
  proxy.
- CMAS evaluates modality-level attribution, not spatial or temporal localization within a
  modality (e.g. it does not tell you *which frames* or *which audio segment* were faked).
- The ablation-based attribution method (§3.3) only measures the fusion+classifier head's
  reliance on each modality's *pooled* embedding; because each modality is a single vector
  entering fusion (rather than, say, per-frame or per-audio-chunk tokens), it cannot say which
  part of that modality mattered. A multi-token cross-attention fusion (per-frame visual
  tokens, per-chunk audio tokens) would let attention weights themselves be non-degenerate and
  is a natural extension for finer-grained, genuinely attention-based attribution — noted here
  as future work rather than attempted in this version, to avoid a bigger architectural change
  than the required "average frame embeddings" / single audio embedding specification called for.
- Integrated Gradients attributions here are computed at the embedding level (post-encoder)
  rather than the raw pixel/waveform level, trading fidelity to raw input for tractability and
  a direct per-modality scalar.
- Results depend on FakeAVCeleb's specific manipulation methods (face-swap/reenactment for
  video, voice-cloning/lip-sync for audio); generalization to unseen manipulation techniques is
  untested here.
- Compute constraints (Colab-class GPU) limit frame count, audio duration, and IG step count;
  these are configurable in `config.yaml` and larger settings may change results.

## 8. Conclusion

`[FILL IN AFTER RESULTS: 3-5 sentence summary of contribution + headline numbers + one
takeaway for future work, e.g. extending CMAS to more than two modalities or to spatial/temporal
localization.]`

## References

`[Add citations: FakeAVCeleb dataset paper, EfficientNet, Wav2Vec2, Integrated Gradients
(Sundararajan et al.), Captum, relevant audio-visual deepfake detection and XAI-for-deepfakes
prior work found during your literature review.]`
