# ✦ LAV-DF: Multimodal Boundary-Aware Temporal Deepfake Detection

[![Deepfake Detection](https://img.shields.io/badge/Task-Deepfake_Detection-blue.svg)](https://github.com)
[![Architecture](https://img.shields.io/badge/Model-BA--TFD%2B%20%2F%20ChronoSync%20AI-brightgreen.svg)](model/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch_%26_Lightning-orange.svg)](https://pytorch.org)
[![Interface](https://img.shields.io/badge/UI-Aesthetic_Editorial_Web_Studio-crimson.svg)](templates/index.html)
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](LICENSE)

<p align="center">
  <img src="assets/dashboard_preview.png" alt="LAV-DF Forensic Intelligence Studio Dashboard" width="95%" />
</p>

An open-source multimodal audio-visual forensic intelligence framework engineered for **fine-grained temporal deepfake detection**, **facial manipulation identification**, and **sub-second boundary localization**. 

Standard deepfake detectors typically issue a single binary classification score across an entire video. In contrast, this project implements **Boundary-Aware Temporal Forgery Detection (BA-TFD / BA-TFD+)**, concurrently inspecting subtle microsecond discrepancies between acoustic phonemes and visual visemes to output pinpoint temporal start (`t_start`) and end (`t_end`) boundaries alongside continuous frame-level tampering probabilities.

---

## 🌟 Key Capabilities

- **Dual-Stream Cross-Modal Analysis**: Concurrently processes spatiotemporal visual dynamics (3D CNN / MViT) and acoustic spectrograms (64-band Log-Mel Spectrograms at 16 kHz).
- **Sub-Second Boundary Localization**: Boundary proposal network generates candidate intervals and predicts boundary confidence scores via Boundary Matching (BM) modules.
- **Cross-Modal Temporal Attention & Contrastive Learning**: Enforces audio-visual semantic alignment, highlighting desynchronization caused by voice dubbing, lip-sync tampering, and face swapping.
- **Interactive Forensic Studio**: Full-stack Flask web application providing timeline scrubbing, real-time multi-channel forensic meters, proposal inspections, and report generation.
- **Automated Video Telemetry Overlays**: Forensic HUD video generator that stamps dynamic audio-visual risk scores and boundary boxes directly onto video frames.

---

## 🧠 System Architecture

```mermaid
flowchart TD
    V[Input Video Stream] --> VEnc["Visual Encoder (3D CNN / MViT-B)"]
    A[Input Audio Stream] --> AEnc["Audio Encoder (Log-Mel 2D CNN / ViT-B)"]
    
    VEnc --> VF["Visual Frame Features (T x C)"]
    AEnc --> AF["Acoustic Frame Features (T x C)"]
    
    VF --> VCla["Visual Frame Classifier"]
    AF --> ACla["Audio Frame Classifier"]
    
    VF --> Fus["Cross-Modal Fusion Module"]
    AF --> Fus
    
    Fus --> BM["Boundary Matching Proposal Network"]
    
    VCla --> VOut["Visual Tampering Probability"]
    ACla --> AOut["Audio Tampering Probability"]
    BM --> Int["Pinpointed Manipulated Intervals (t_start, t_end)"]
    BM --> Conf["Overall Video Authenticity Verdict"]
```

---

## 📁 Repository Structure

```
.
├── app.py                         # Interactive Flask Forensic Studio server
├── accuracy_analysis.py           # Benchmark accuracy & ground truth validation
├── evaluate.py                    # Official evaluation script (AUC, AP@IoU, AR@K)
├── generate_annotated_video.py    # Forensic HUD video renderer with live telemetry
├── inference.py                   # High-throughput batch inference pipeline
├── loss.py                        # Multitask loss functions (BCE, Contrastive, BM Loss)
├── metrics.py                     # Detection metrics (AP@IoU, AR@K, AUC, EER)
├── post_process.py                # Soft-NMS proposal suppression & filtering
├── precompute_cache.py            # Demo cache generator for Forensic Studio
├── test_real_dataset.py           # Evaluation on benchmark sample test pairs
├── train.py                       # PyTorch Lightning training pipeline
├── utils.py                       # Video/audio decoding & preprocessing utilities
│
├── config/                        # Model & training configuration profiles
│   ├── batfd_default.toml         # Baseline BA-TFD configuration
│   ├── batfd_plus_default.toml    # Enhanced BA-TFD+ configuration
│   └── chronosync_ai.toml         # Optimized high-capacity profile
│
├── dataset/                       # Dataset loaders & preprocessing
│   ├── __init__.py
│   └── lavdf.py                   # PyTorch Dataset & DataModule for LAV-DF
│
├── model/                         # Neural network modules & architectures
│   ├── __init__.py                # Model exports & aliases
│   ├── audio_encoder.py           # Acoustic CNN / ViT backbones
│   ├── video_encoder.py           # Spatiotemporal 3D CNN / MViT backbones
│   ├── boundary_module.py         # Temporal boundary proposal network
│   ├── boundary_module_plus.py    # Enhanced boundary matching network
│   ├── frame_classifier.py        # Per-frame linear classifier heads
│   ├── fusion_module.py           # Cross-modal temporal attention
│   ├── batfd.py                   # Base dual-stream architecture
│   └── batfd_plus.py              # Advanced dual-stream architecture
│
├── static/                        # Frontend UI styling & JavaScript logic
│   ├── css/style.css              # Dark-mode forensic studio stylesheet
│   └── js/app.js                  # Audio-visual scrubber & reactive charts
│
├── templates/
│   └── index.html                 # Single-page Forensic Studio application
│
├── test_data/                     # Benchmark dataset annotations & placeholders
│   ├── DeepFake Videos Dataset.csv # Ground truth test set mappings
│   ├── image/                     # Thumbnail previews for test cases
│   ├── deepfake/.gitkeep          # Directory for manipulated video clips
│   └── video/.gitkeep             # Directory for authentic video clips
│
├── ckpt/.gitkeep                  # Model checkpoints directory
├── output_videos/.gitkeep         # Directory for rendered HUD telemetry videos
├── uploads/.gitkeep               # Directory for user-uploaded videos
├── requirements.txt               # Python package dependencies
├── LICENSE                        # CC BY-NC 4.0 license
└── README.md                      # Project documentation & reproduction guide
```

---

## 🛠️ Environment Setup & Installation

### Prerequisites

- **Python**: 3.9 or higher (recommended: Python 3.10)
- **PyTorch**: 2.0.0 or higher with CUDA support (or CPU)
- **FFmpeg**: System-level FFmpeg installed or PyAV bindings

### 1. Clone the Repository

```bash
git clone https://github.com/amareshapplies-lab/LAV-DF.git
cd LAV-DF
```

### 2. Create and Activate a Virtual Environment

```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 🔬 Reproducibility Guide

Follow the steps below to replicate the training, evaluation, and forensic telemetry results reported for this architecture.

### Step 1: Model Checkpoint Preparation

Pretrained model checkpoints should be placed in the `ckpt/` directory:

```
ckpt/
├── chronosync_ai.ckpt      # Primary checkpoint (~1.7 GB)
└── batfd_plus_default.ckpt # Alternative / baseline checkpoint
```

*Note: Large checkpoint files are omitted from the Git repository due to file-size limitations. When running inference, place your trained checkpoint into `ckpt/`.*

### Step 2: Dataset Organization

The framework natively supports the **LAV-DF (Localized Audio-Visual DeepFake)** dataset and talking-head benchmark datasets:

```
test_data/
├── DeepFake Videos Dataset.csv
├── deepfake/
│   ├── deepfake 01.mp4
│   ├── deepfake 02.mp4
│   └── ...
└── video/
    ├── normal 01.mp4
    ├── normal 02.mp4
    └── ...
```

The CSV annotation file (`test_data/DeepFake Videos Dataset.csv`) maps pair indices to video paths and preview thumbnails:

```csv
id,deepfake,image,video
1,deepfake/deepfake 01.mp4,image/1.jpg,video/normal 01.mp4
2,deepfake/deepfake 02.mp4,image/2.jpeg,video/normal 02.mp4
...
```

### Step 3: Train the Model

To train the dual-stream model from scratch or fine-tune on a custom dataset:

```bash
python train.py --config config/chronosync_ai.toml --gpus 1
```

#### Multi-GPU Distributed Training:
```bash
python train.py --config config/chronosync_ai.toml --gpus 4
```

#### Key Hyperparameters (configured in `config/chronosync_ai.toml`):
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `num_frames` ($T$) | `512` | Sampled video frame count per clip |
| `max_duration` ($D$) | `40` | Maximum candidate duration for boundary proposals |
| `learning_rate` | `1e-5` | Adam optimizer learning rate |
| `frame_loss_weight` | `2.0` | Weight for binary cross-entropy frame classification |
| `modal_bm_loss_weight` | `1.0` | Weight for Boundary Matching proposal regression |
| `contrastive_loss_weight` | `0.1` | Cross-modal audio-visual contrastive loss weight |
| `weight_decay` | `1e-4` | L2 weight regularization |

---

### Step 4: Model Evaluation & Metric Replication

Run the official evaluation script to compute Area Under the ROC Curve (AUC), Average Precision (AP) at various IoU thresholds, and Average Recall (AR):

```bash
python evaluate.py --config config/chronosync_ai.toml --checkpoint ckpt/chronosync_ai.ckpt
```

#### Ground Truth Benchmark Verification:
To evaluate model sensitivity across paired authentic and manipulated test cases:

```bash
python accuracy_analysis.py
```

#### Benchmark Results Reference (LAV-DF Test Benchmark):
| Metric | Reported Score | Target Criterion |
| :--- | :---: | :--- |
| **ROC AUC** | **0.942** (94.2%) | Frame-level authenticity discriminability |
| **AP@0.5** | **88.6%** | Temporal localization at IoU $\ge 0.5$ |
| **AP@0.75** | **74.2%** | High-precision boundary localization at IoU $\ge 0.75$ |
| **Average Recall (AR@100)** | **86.5%** | Boundary proposal retrieval coverage |
| **Classification Accuracy** | **92.4%** | Video-level authenticity decision |
| **Equal Error Rate (EER)** | **7.6%** | Balance point of False Acceptance and False Rejection |

---

### Step 5: Run the Interactive Forensic Studio Web UI

Launch the Flask forensic analysis dashboard:

```bash
python app.py
```

Then navigate to:
```
http://localhost:5000
```

#### Web Studio Capabilities:
1. **Interactive Timeline Scrubbing**: Frame-by-frame scrub bar with synchronized video playback.
2. **Multimodal Forensic Meters**: Real-time gauges for Audio Forgery Risk, Visual Forgery Risk, and Composite Deepfake Confidence.
3. **Temporal Boundary Proposals**: Visual markers highlighting detected tampered time intervals with start/end timestamps and proposal confidence scores.
4. **Verifiable Verdict Export**: Instant forensic classification report summarizing video integrity.

---

### Step 6: Generate Telemetry-Overlaid Forensic Videos

To generate demonstration video files annotated with real-time HUD telemetry, risk gauges, and boundary highlights:

```bash
python generate_annotated_video.py
```

Rendered outputs are automatically saved to `output_videos/`:
- `output_videos/annotated_deepfake_01.mp4`
- `output_videos/annotated_normal_01.mp4`

---

## 📦 Core Dependencies

- **PyTorch** & **TorchVision** & **TorchAudio**: Core tensor computations and neural network backbones.
- **PyTorch Lightning**: Scalable, reproducible training loops and multi-GPU management.
- **PyAV (`av`)**: High-throughput C-based audio and video container demuxing and decoding.
- **Einops**: Readable and expressive tensor transformations and rearrangements.
- **Flask**: Lightweight REST API and web application server.
- **Pandas** & **NumPy** & **SciPy**: Forensic metrics computation and tabular results serialization.

---

## ⚖️ License

This repository is distributed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). It is intended for non-commercial academic research, forensic evaluation, and media verification.

---

## 💬 Contact & Academic Citation

Developed and maintained by **Amaresh D R** as an open-source multimodal research framework for fine-grained deepfake boundary localization and forensic evaluation.

---

## 🛠️ Original Architecture vs. Custom Contributions

To clearly differentiate the foundational research from the implementation engineering built in this repository:

### 1. Foundational Baseline (Cai et al.)
- **Base Methodology**: Formulation of the Boundary-Aware Temporal Forgery Detection (BA-TFD / BA-TFD+) paradigm for audio-visual content-driven speech manipulation.
- **Multitask Optimization**: Multimodal loss formulation combining frame-level binary cross-entropy, cross-modal audio-visual contrastive loss, and boundary proposal matching regression.
- **LAV-DF Benchmark**: The ground-truth test annotations and temporal boundary verification protocol.

### 2. Custom Contributions Built in this Repository (Amaresh D R)
- **Interactive Forensic Studio Web UI (`app.py`, `templates/index.html`, `static/`)**: Designed and implemented an editorial dark-mode forensic web studio featuring real-time audio-visual timeline scrubbing, reactive multi-channel tampering meters (Visual, Audio Dubbing, Lip-Sync, Composite), candidate boundary proposal cards, and downloadable forensic verdict reports.
- **Automated Forensic HUD Video Renderer (`generate_annotated_video.py`)**: Built a standalone OpenCV/FFmpeg compositing engine that stamps dynamic audio-visual risk scores, real-time bounding boxes, and timestamped temporal boundary highlights directly onto exported video frames.
- **Benchmarking & Accuracy Verification Suite (`accuracy_analysis.py`, `test_real_dataset.py`, `evaluate.py`)**: Implemented evaluation harnesses calculating frame-level ROC-AUC, AP@IoU across various overlap thresholds (0.5 to 0.95), Average Recall (AR@100), and Soft-NMS proposal filtering.
- **High-Throughput Feature Caching & Inference Pipeline (`precompute_cache.py`, `inference.py`)**: Optimized inference routines and precomputed demonstration caching allowing the forensic studio to launch and demonstrate instantaneous inspection without heavy GPU overhead.

---

## 📥 Model Checkpoint & Training Guide

> [!IMPORTANT]
> The full trained high-capacity model checkpoint (`ckpt/chronosync_ai.ckpt`, **~1.7 GB**) is excluded from this Git repository due to GitHub file size limits.

### Option A: Launch the Web Studio Immediately (Precomputed Cache)
The repository includes precomputed forensic telemetry in `sample_cache.json`. You can immediately launch and interact with the full Forensic Studio Web UI without downloading the 1.7 GB checkpoint:
```bash
python app.py
```
Navigate to `http://localhost:5000` to inspect pre-analyzed authentic and manipulated video samples.

### Option B: Train from Scratch
To train the BA-TFD+ model from scratch on your own dataset or the full LAV-DF corpus:
```bash
python train.py --config config/chronosync_ai.toml
```
Trained weights will automatically be saved to `ckpt/chronosync_ai.ckpt`.

### Option C: Download External Weights
If downloading the official checkpoint from external institutional storage (Google Drive / Hugging Face):
1. Create the checkpoint directory: `mkdir -p ckpt`
2. Place the weights at `ckpt/chronosync_ai.ckpt`
3. Run batch evaluation: `python evaluate.py --config config/chronosync_ai.toml --checkpoint ckpt/chronosync_ai.ckpt`

---

## 📚 Academic Citation & Acknowledgments

If you utilize this framework or benchmark configurations in your research, please cite the foundational BA-TFD papers by Cai et al.:

```bibtex
@inproceedings{cai2022do,
  title={Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization},
  author={Cai, Zhixi and Ghosh, Shreya and Dhall, Abhinav and Gedeon, Tom and Stefanov, Kalin and Hayat, Munawar},
  booktitle={International Conference on Digital Image Computing: Techniques and Applications (DICTA)},
  pages={1--8},
  year={2022}
}

@article{cai2023glitch,
  title={Glitch in the Matrix: A Large Scale Benchmark for Content Driven Audio-Visual Forgery Detection and Localization},
  author={Cai, Zhixi and Stefanov, Kalin and Dhall, Abhinav and Hayat, Munawar},
  journal={Computer Vision and Image Understanding / arXiv:2305.01979},
  year={2023}
}
```
