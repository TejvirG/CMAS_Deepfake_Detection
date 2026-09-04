CMAS: Cross-Modal Attribution Score for Audio-Visual Deepfake Detection

A multimodal audio-visual deepfake detection system with a focus on evaluating whether model explanations identify the modality that was actually manipulated.

The project introduces the Cross-Modal Attribution Score (CMAS), a simple modality-level metric for comparing an explanation's visual and audio contributions with the known manipulated modality of a deepfake sample.

The project uses FakeAVCeleb, which contains real videos as well as video-only, audio-only, and audio-video manipulated samples.

Project Overview

Multimodal deepfake detectors can combine information from both video and audio to decide whether a sample is real or fake. However, getting the final prediction right does not necessarily mean that the model relied on the correct modality.

For example, if only the audio has been manipulated, an explanation should ideally indicate that audio contributed more strongly to the decision.

This project explores that problem through CMAS.

The main components are:

Visual feature extraction using EfficientNet-B0
Audio feature extraction using Wav2Vec2-base
Audio-visual feature fusion using cross-attention
Deepfake classification
Integrated Gradients attribution
Modality ablation attribution
CMAS for modality-level explanation evaluation
Evaluation on FakeAVCeleb

The implementation and experiments are intended to support the accompanying research paper.

Paper

The current version of the paper is available in:

Result/CMAS.pdf

Title:
CMAS: A Cross-Modal Attribution Score for Modality-Level Explanation Evaluation in Audio-Visual Deepfake Detection

The paper is currently a preprint/research manuscript under submission. It has not been presented here as a published paper or accepted publication.

If the paper is accepted or published, this section can be updated with the official publication information.

Method

The system has two main input streams.

Video

Video frames are sampled from each clip and processed using:

OpenCV for video handling
MediaPipe for face detection
EfficientNet-B0 for visual feature extraction

The frame-level features are aggregated into a visual representation.

Audio

Audio is extracted from the video and converted to a 16 kHz mono waveform.

The audio stream uses:

FFmpeg for audio extraction
Wav2Vec2-base for audio representation
Mean pooling for the final audio embedding
Multimodal Fusion

The visual and audio representations are combined using a cross-attention based fusion module.

The resulting multimodal representation is passed to a classifier that predicts whether the input is REAL or FAKE.

CMAS

The main idea of CMAS is to evaluate whether an explanation assigns importance to the modality that was actually manipulated.

For each manipulated sample, the ground-truth modality is represented as a two-element vector:

Video-only  -> [1, 0]
Audio-only  -> [0, 1]
Both        -> [0.5, 0.5]

An explanation method produces a corresponding attribution vector:

e = [video contribution, audio contribution]

The attribution vector is normalized before comparison.

CMAS is then calculated using cosine similarity between the ground-truth modality vector and the explanation vector.

A higher score means that the explanation is more aligned with the known manipulated modality.

The implementation is located in:

metrics/cmas.py
Explanation Methods

The current experiments use two attribution approaches.

Modality Ablation

One modality is removed at a time and the change in the model's output is measured.

This gives an estimate of how much the prediction depends on the visual and audio inputs.

Integrated Gradients

Integrated Gradients is applied to estimate the contribution of the input representations to the model prediction.

The number of integration steps can be configured in config.yaml.

Important Implementation Detail

The current fusion module produces one pooled visual embedding and one pooled audio embedding before cross-attention.

Because each attention operation therefore has only one key, the resulting attention weights are always 1.0.

For this reason, the raw attention weights are not treated as a meaningful explanation method in the reported CMAS analysis.

The project instead uses modality ablation and Integrated Gradients for attribution.

Dataset

The experiments use FakeAVCeleb.

The dataset contains:

Real samples
Video-only manipulated samples
Audio-only manipulated samples
Audio-video manipulated samples

The dataset is not included in this repository.

You need to obtain the dataset separately and follow its applicable terms of use.

The repository includes code for preparing the dataset structure expected by the training and evaluation pipeline.

Reported Results

The following results correspond to the final experiment configuration described in the accompanying paper.

Multimodal Model

Test set size: 3,103 clips

Metric	Score
Accuracy	99.55%
Precision	99.97%
Recall	99.57%
F1	99.77%
ROC-AUC	99.93%

Confusion matrix:

                Predicted
                REAL   FAKE

Actual REAL       74      1
Actual FAKE       13   3015
Visual-Only Model
Metric	Score
Accuracy	97.23%
Precision	99.93%
Recall	97.23%
F1	98.56%
ROC-AUC	98.36%
Audio-Only Model
Metric	Score
Accuracy	84.56%
Precision	99.04%
Recall	85.01%
F1	91.49%
ROC-AUC	83.60%
CMAS

The reported CMAS analysis used 125 FAKE samples with defined manipulated-modality labels from the evaluation subset.

Attribution Method	Mean CMAS	Standard Deviation
Modality Ablation	0.9997	0.0052
Integrated Gradients	0.9466	0.0340
Ablation + Integrated Gradients	0.9859	0.0067

These results are reported for the experiments described in the paper and should not be interpreted as evidence of generalization beyond the evaluated dataset and experimental setup.

Repository Structure
CMAS-Deepfake-Detection/
│
├── README.md
├── requirements.txt
├── config.yaml
├── verify_environment.py
│
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
│   └── attribution.py
│
├── metrics/
│   └── cmas.py
│
├── experiments/
│   ├── exp1_visual_only.py
│   ├── exp2_audio_only.py
│   ├── exp3_multimodal.py
│   └── exp4_cmas_eval.py
│
├── visualization/
│
├── utils/
│
├── paper_assets/
│
├── tests/
│
├── paper_outline/
│
└── Result/
    └── CMAS.pdf
Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CMAS-Deepfake-Detection

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Activate it on Linux or macOS:

source venv/bin/activate

Install the dependencies:

pip install -r requirements.txt
Environment Check

Run:

python verify_environment.py

This checks whether the main Python dependencies and required environment components are available.

Dataset Preparation

The dataset is not included in this repository.

After obtaining FakeAVCeleb, prepare the dataset using:

python prepare_fakeavceleb.py

The exact dataset paths should be configured according to the structure expected by the project.

Configuration

The main configuration is stored in:

config.yaml

Important settings include:

Random seed
Batch size
Dataset paths
Sampling configuration
Integrated Gradients steps
Attribution settings
CMAS configuration

Review the configuration before starting training.

Training

The main training script is:

python train.py

The training configuration is read from config.yaml.

Evaluation

Run the main evaluation with:

python evaluate.py

The evaluation scripts report classification metrics and generate the outputs used for analysis.

Experiments

The repository contains separate experiment scripts for the different model configurations.

Visual-only
python experiments/exp1_visual_only.py
Audio-only
python experiments/exp2_audio_only.py
Multimodal
python experiments/exp3_multimodal.py
CMAS Evaluation
python experiments/exp4_cmas_eval.py

The CMAS evaluation has a configurable sample limit because attribution methods such as Integrated Gradients are computationally more expensive than standard inference.

Inference

The inference script can be used to run the trained model on an individual sample:

python inference.py

Refer to the script configuration and arguments for the expected input format.

Web Application

A simple application is included in:

app.py

It can be started with the appropriate Python environment and the dependencies installed from requirements.txt.

Reproducing the Experiments

A typical workflow is:

1. Install dependencies
2. Verify the environment
3. Obtain FakeAVCeleb
4. Prepare the dataset
5. Configure config.yaml
6. Train the required models
7. Run evaluation
8. Run the individual modality experiments
9. Run CMAS evaluation
10. Generate visualizations

The reported numbers in the paper correspond to a particular trained-model state, dataset split, configuration, and evaluation procedure. Re-running the repository may produce different results if the data split, checkpoints, environment, or configuration differs.

Limitations

There are several limitations to the current implementation.

Dataset

The reported experiments use FakeAVCeleb only. Results on this dataset do not establish performance on other deepfake datasets or real-world media.

Dataset Split

The final reported test split contains 3,103 clips, including a relatively small number of REAL samples. The dataset split and manifest were regenerated during development, so earlier intermediate experiments used a different test-set size.

Model Architecture

The experiments use a single main visual encoder, audio encoder, and fusion architecture. Comparisons against a wider range of multimodal architectures were outside the current scope.

CMAS

CMAS evaluates modality-level attribution using a two-dimensional visual/audio representation. It does not provide a detailed spatial or temporal explanation of which face region, frame, word, or audio segment influenced the model.

Attention

The current fusion architecture pools each modality before cross-attention. As a result, the attention weights themselves do not provide useful modality-level attribution and are not used as the primary explanation method.

Attribution Evaluation

The reported CMAS analysis uses a limited evaluation subset because Integrated Gradients is computationally expensive.

Dataset-Specific Effects

The project does not claim that the reported near-perfect classification performance demonstrates robustness to all types of synthetic media. Dataset-specific artifacts and shortcuts can affect deepfake detection results, so broader evaluation would be required before making stronger generalization claims.

Project Status

This repository contains the current implementation, experiments, and research manuscript for the project.

The paper is currently being submitted for peer review.

There is currently no claim of acceptance or publication.

Citation

At present, there is no published citation to provide.

If the paper receives an official publication or preprint identifier, this section will be updated with the corresponding citation.
