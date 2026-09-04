# CMAS: Cross-Modal Attribution Score for Explainable Audio-Visual Deepfake Detection

CMAS is an audio-visual deepfake detection project that I built to study both deepfake classification and the explanations produced by multimodal models.

The system uses video and audio together to classify media as real or fake. Along with the detector, I introduce the **Cross-Modal Attribution Score (CMAS)**, a metric for evaluating whether a model's explanation points to the modality that was actually manipulated.

The main idea is simple. A model can correctly classify a clip as fake without necessarily relying on the right evidence. If only the audio was manipulated, for example, I would expect the model to rely more on the audio than the video. CMAS provides a way to measure this type of explanation alignment at the modality level.

The project was evaluated using the FakeAVCeleb dataset.

## Paper

The complete paper is available here:

[**CMAS.pdf**](Result/CMAS.pdf)

**Title:**
CMAS: A Cross-Modal Attribution Score for Modality-Level Explanation Evaluation in Audio-Visual Deepfake Detection

The paper contains the complete methodology, experimental setup, results, explainability analysis, and limitations.

## What I built

The project includes:

* Audio-visual deepfake detection
* Visual-only and audio-only baselines
* A multimodal model using both modalities
* EfficientNet-B0 for visual features
* Wav2Vec2-base for audio features
* Cross-modal fusion
* Integrated Gradients for attribution
* Modality ablation for attribution
* The CMAS metric
* Classification and explainability evaluation
* A Gradio demo for individual video inference

## Results

The final multimodal model was evaluated on a test split containing 3,103 clips.

| Model       | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ----------- | -------: | --------: | -----: | -----: | ------: |
| Visual-only |   97.23% |    99.93% | 97.23% | 98.56% |  98.36% |
| Audio-only  |   84.56% |    99.04% | 85.01% | 91.49% |  83.60% |
| Multimodal  |   99.55% |    99.97% | 99.57% | 99.77% |  99.93% |

The CMAS evaluation produced the following results:

| Attribution Method   | Mean CMAS | Standard Deviation |
| -------------------- | --------: | -----------------: |
| Modality Ablation    |    0.9997 |             0.0052 |
| Integrated Gradients |    0.9466 |             0.0340 |
| Both                 |    0.9859 |             0.0067 |

The CMAS analysis was performed on 125 FAKE samples with defined manipulated-modality labels.

The complete results and analysis are available in [CMAS.pdf](Result/CMAS.pdf).

## Dataset

I used the **FakeAVCeleb** dataset for the experiments.

The full dataset contains 21,544 clips:

| Category             | Number of clips |
| -------------------- | --------------: |
| Real                 |             500 |
| Video-only fake      |           9,709 |
| Audio-only fake      |             500 |
| Audio and video fake |          10,835 |
| Total                |          21,544 |

The dataset is not included in this repository. Access needs to be requested from the dataset authors, and the dataset's research-use terms should be followed.

After obtaining the dataset, the repository can be used to create the required manifests:

```bash
python prepare_fakeavceleb.py \
    --raw_dir /path/to/FakeAVCeleb \
    --out_dir data/fakeavceleb \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

This creates:

```text
data/fakeavceleb/
├── train_manifest.csv
├── val_manifest.csv
└── test_manifest.csv
```

Each manifest contains:

```text
video_path
label
manipulated_modality
```

The `label` can be:

```text
REAL
FAKE
```

The `manipulated_modality` can be:

```text
none
audio
video
both
```

## Model architecture

The overall pipeline is:

```text
                    Input Video
                        |
                        v
                 Frame Sampling
                        |
                        v
                  Face Detection
                        |
                        v
                  EfficientNet-B0
                        |
                        v
                 Visual Features
                        |
                        |
                        |        Input Audio
                        |             |
                        |             v
                        |       Audio Extraction
                        |             |
                        |             v
                        |        Wav2Vec2-base
                        |             |
                        |             v
                        |        Audio Features
                        |             |
                        +------+------+
                               |
                               v
                       Multimodal Fusion
                               |
                               v
                           Classifier
                               |
                               v
                         REAL / FAKE
```

### Visual branch

I use EfficientNet-B0 as the default visual encoder.

A fixed number of frames are sampled from each video. Face detection is performed using MediaPipe, and the processed frames are passed through EfficientNet.

The frame-level features are combined to produce a visual representation.

The repository also contains support for EfficientNet-B4 for systems with more available GPU memory.

### Audio branch

The audio branch uses Wav2Vec2-base.

Audio is extracted using FFmpeg and converted to mono 16 kHz audio before being passed to Wav2Vec2.

The resulting representations are pooled to produce a fixed-size audio representation.

### Fusion

The visual and audio representations are combined using a cross-modal fusion module.

The fused representation is then passed to a classifier that predicts whether the input is REAL or FAKE.

I also evaluate the visual and audio branches separately to compare the contribution of each modality.

## CMAS

### Cross-Modal Attribution Score

The main research contribution of this project is CMAS.

The idea is to represent the model's explanation using two values:

```text
[visual_importance, audio_importance]
```

The known manipulated modality is represented in the same two-dimensional space.

For a video-only fake:

```text
[1, 0]
```

For an audio-only fake:

```text
[0, 1]
```

For a fake where both modalities are manipulated:

```text
[0.5, 0.5]
```

Real samples are not included in CMAS because they do not have a manipulated modality.

CMAS is calculated using cosine similarity:

```text
CMAS(sample) = cosine_similarity(e, g)
```

where `e` is the explanation-derived attribution vector and `g` is the ground-truth modality vector.

A higher CMAS value means that the model's attribution is more closely aligned with the known manipulated modality.

The implementation is in:

```text
metrics/cmas.py
```

## Explanation methods

I use two approaches to estimate modality contribution.

### Integrated Gradients

Integrated Gradients is used to estimate the contribution of the model's inputs to its prediction.

The resulting attribution values are aggregated at the modality level and used to calculate CMAS.

The implementation is in:

```text
explainability/integrated_gradients.py
```

### Modality ablation

For modality ablation, I compare the model's output with both modalities available against the output after removing one modality.

The change in the predicted-class logit is used as an estimate of that modality's contribution.

The implementation is in:

```text
explainability/attribution.py
```

## Attention weights

There is an important detail about the current fusion implementation.

Each modality is pooled into a single representation before the cross-attention step. This means that the attention operation has only one key/value.

Because of this, the softmax attention weight is always 1.0.

I therefore do not use the raw cross-attention weights as modality-level explanations. The explanation analysis in this project uses modality ablation and Integrated Gradients instead.

This limitation is also discussed in the paper.

## Repository structure

```text
CMAS-Deepfake-Detection/
│
├── README.md
├── requirements.txt
├── verify_environment.py
├── config.yaml
├── train.py
├── evaluate.py
├── inference.py
├── app.py
├── prepare_fakeavceleb.py
│
├── models/
│   ├── visual_encoder.py
│   ├── audio_encoder.py
│   └── fusion.py
│
├── dataset/
│   └── dataset.py
│
├── explainability/
│   ├── integrated_gradients.py
│   └── attribution.py
│
├── metrics/
│   └── cmas.py
│
├── experiments/
│   ├── exp1_visual_only.py
│   ├── exp2_audio_only.py
│   ├── exp3_multimodal.py
│   ├── exp4_cmas_eval.py
│   └── run_all.py
│
├── visualization/
│   └── visualization.py
│
├── utils/
│   ├── logging_utils.py
│   └── seed.py
│
├── paper_assets/
│   ├── architecture.py
│   └── results_template.xlsx
│
├── tests/
│   └── test_smoke.py
│
├── Result/
│   └── CMAS.pdf
│
└── paper_outline.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/TejvirG/CMAS_Deepfake_Detection.git
cd CMAS-Deepfake-Detection
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The project also requires FFmpeg.

On Ubuntu or Debian:

```bash
sudo apt-get install -y ffmpeg
```

On macOS:

```bash
brew install ffmpeg
```

A GPU is recommended for training and explainability experiments. The project can also run on CPU, but training will take considerably longer.

## Verify the environment

Before starting a full experiment, run:

```bash
python verify_environment.py
```

This checks the main dependencies and components used by the project, including PyTorch, EfficientNet, Wav2Vec2, FFmpeg, MediaPipe, and GPU availability.

Warnings do not necessarily indicate a problem. Any `[FAIL]` message should be checked before starting training.

## Run the tests

The repository contains smoke tests that use synthetic data and do not require FakeAVCeleb.

```bash
python -m pytest tests/test_smoke.py -v
```

These tests are useful for checking the main components before starting the longer training process.

Some tests that require pretrained Wav2Vec2 weights may be skipped when the environment does not have internet access.

## Prepare the dataset

After obtaining FakeAVCeleb, create the dataset manifests:

```bash
python prepare_fakeavceleb.py \
    --raw_dir /path/to/FakeAVCeleb \
    --out_dir data/fakeavceleb \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

Before training, check the generated manifests to make sure the file paths and labels are correct.

## Configure the project

Most experiment settings are stored in:

```text
config.yaml
```

The configuration includes settings such as:

* Dataset paths
* Batch size
* Learning rate
* Number of epochs
* Sampling settings
* Random seed
* Attribution method
* Integrated Gradients steps
* Data-loader settings

The training and experiment scripts read these settings from the configuration file.

## Train the model

The main training script is:

```bash
python train.py --config config.yaml
```

The trained checkpoints are saved under:

```text
checkpoints/
```

Training logs are stored under:

```text
logs/
```

TensorBoard logs are stored under:

```text
logs/tb/
```

## Evaluate the model

After training, run:

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model_multimodal.pt \
    --split test \
    --config config.yaml
```

The evaluation pipeline generates the classification metrics for the selected test split.

The final reported results are included in:

```text
Result/CMAS.pdf
```

## Run the experiments

The project contains four experiment scripts.

### Experiment 1: Visual-only

```bash
python experiments/exp1_visual_only.py
```

This evaluates the model using only the visual branch.

### Experiment 2: Audio-only

```bash
python experiments/exp2_audio_only.py
```

This evaluates the model using only the audio branch.

### Experiment 3: Multimodal

```bash
python experiments/exp3_multimodal.py
```

This evaluates the complete audio-visual model.

### Experiment 4: CMAS evaluation

```bash
python experiments/exp4_cmas_eval.py
```

This evaluates modality attribution using CMAS.

The experiments can also be run together:

```bash
python experiments/run_all.py --config config.yaml
```

Running all experiments can take several hours on a typical Colab GPU. Running them separately can be more practical when testing or debugging the project.

## Visualization

The visualization script can be run with:

```bash
python visualization/visualization.py --results_dir results/
```

It generates figures for the classification and attribution experiments, including:

```text
results/roc_curve.png
results/confusion_matrix.png
results/cmas_comparison.png
results/attribution_examples.png
```

## Single-video inference

A trained multimodal model can be used on an individual video:

```bash
python inference.py \
    --video path/to/clip.mp4 \
    --checkpoint checkpoints/best_model_multimodal.pt
```

The inference output includes the predicted class and modality contribution information.

## Gradio demo

The project also includes a simple Gradio interface.

Run:

```bash
python app.py
```

The application will be available locally at:

```text
http://127.0.0.1:7860
```

You can upload a video and view the model's prediction, confidence, visual and audio contribution values, and CMAS score where applicable.

## Google Colab

The project can be run on Google Colab with a GPU.

First, enable a GPU through:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Clone the repository:

```python
!git clone https://github.com/TejvirG/CMAS_Deepfake_Detection.git
%cd CMAS-Deepfake-Detection
```

Install the dependencies:

```python
!pip install -r requirements.txt -q
```

After installation, restart the runtime.

Then run:

```python
!python verify_environment.py
```

Run the tests:

```python
!python -m pytest tests/test_smoke.py -v
```

After preparing the FakeAVCeleb dataset, training can be started with:

```python
!python train.py --config config.yaml
```

The training code automatically checks for CUDA and uses the GPU when available.

For longer experiments, it is better to keep the dataset, checkpoints, and generated files on Google Drive because files stored only on the temporary Colab runtime can be lost when the session ends.

## Reproducibility

I use a fixed random seed in the default configuration.

For comparable results, the following should be kept consistent:

* FakeAVCeleb version
* Dataset split
* Random seed
* Model configuration
* Sampling strategy
* Training settings
* Attribution settings
* CMAS evaluation settings

The repository does not include the FakeAVCeleb dataset or trained model weights.

## Limitations

There are several limitations to the current project.

### Single dataset

The experiments are based on FakeAVCeleb only. The reported results therefore do not establish how well the model generalizes to other deepfake datasets.

### Class imbalance

FakeAVCeleb contains substantially more fake samples than real samples. The relatively small number of real samples is important when interpreting the classification results.

### CMAS evaluation size

Integrated Gradients is computationally expensive, so the CMAS evaluation was performed on a smaller subset of the test data.

The reported CMAS results are based on 125 FAKE samples with defined manipulated-modality labels.

### Modality-level explanation

CMAS works at the modality level. It measures alignment between visual and audio attribution, but it does not identify the exact frame, facial region, or audio segment responsible for a prediction.

### Dataset-specific shortcuts

Deepfake datasets can contain artifacts or shortcuts that make detection easier than it would be on naturally occurring manipulated media. The current implementation does not completely control for these effects.

### Attention interpretation

The current fusion architecture reduces each modality to a single representation before cross-attention. Because of this, the raw attention weights are not suitable for direct modality-level attribution.

### Computational requirements

Running the visual-only, audio-only, and multimodal experiments followed by the CMAS evaluation can take several hours on a typical Colab GPU.

## Future work

Some directions I would like to explore further are:

* Evaluating CMAS on additional audio-visual deepfake datasets
* Testing cross-dataset generalization
* Increasing the number of real samples where possible
* Adding temporal sampling and temporal augmentation
* Investigating dataset-specific shortcuts
* Comparing CMAS across different multimodal architectures
* Studying spatial and temporal explanations
* Evaluating additional attribution methods
* Extending CMAS beyond modality-level attribution

## Citation

If you use CMAS or this implementation in your work, please cite the associated paper.

```bibtex
@article{grewal2026cmas,
  title={CMAS: A Cross-Modal Attribution Score for Modality-Level Explanation Evaluation in Audio-Visual Deepfake Detection},
  author={Grewal, Tejvir Singh},
  year={2026},
  note={Preprint}
}
```

The final paper is available at [`Result/CMAS.pdf`](Result/CMAS.pdf).

## License

The code in this repository is released under the MIT License. See `LICENSE` for the full license text.

FakeAVCeleb is distributed separately under its own research-use terms. The dataset is not included in this repository, and its terms should be followed when obtaining and using the dataset.

