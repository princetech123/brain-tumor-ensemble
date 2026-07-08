# Multi-Model Ensemble Deep Learning Framework for Brain Tumor Detection and Classification Using MRI Images

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, publication-quality medical AI system designed for brain tumor classification and localization from MRI scans. The platform utilizes a multi-model ensemble combining Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) via a dynamic Multi-Head Self-Attention feature fusion layer, backed by Explainable AI (XAI) overlays.

---

## 🌟 Key Features

- **Clinical Preprocessing Pipeline:** Integrated OpenCV workflow (Gaussian Blur noise reduction, CLAHE contrast equalization, and contour-based Otsu Skull Stripping).
- **Heterogeneous Model Backbones:** Feature extractors leveraging ResNet50, EfficientNet-B3, and Vision Transformer (ViT-B/16) pre-trained architectures.
- **Attention-based Feature Fusion:** Custom fusion network that projects multi-scale features to a uniform space, applies a Multi-Head Self-Attention layer to weigh the backbones dynamically, and classifies via dense layers.
- **Explainable AI (XAI):** Visual diagnostic justifications via Grad-CAM (for CNNs) and class-token dot-product Attention Rollout maps (for ViT) overlayed with thermal JET colormaps.
- **Dashboard Interface:** Premium dark-themed, glassmorphic Flask web app providing interactive dropzones, side-by-side diagnostic visuals, and diagnostic consensus progress bars.
- **Performance Analytics:** Evaluation logs tracking diagnostic metrics (Accuracy, sensitivity, specificity, AUC) and system performance (latency and memory).

---

## 📁 Repository Layout

```text
brain_tumor_ensemble/
├── config/
│   └── config.yaml               # Hyperparameters, paths, and preprocessing settings
├── data/
│   └── sample_mri/               # Folder housing training splits (generated via script)
├── src/
│   ├── preprocessing.py          # Clinical preprocessor & PyTorch Dataset
│   ├── models/
│   │   ├── backbones.py          # ResNet50, EfficientNetB3, ViT loaders
│   │   ├── fusion.py             # Attention Fusion network
│   │   └── explainability.py     # Grad-CAM and ViT attention maps
│   ├── training.py               # Two-stage training and test execution
│   ├── evaluation.py             # Performance curves & metric calculators
│   └── web/
│       ├── app.py                # Flask server
│       ├── templates/            # UI layout (index, result, analytics, about)
│       └── static/               # Style sheets, interactive JS scripts, overlays
├── tests/
│   ├── test_preprocessing.py     # Unit tests for preprocessing
│   ├── test_models.py            # Unit tests for backbone tensors
│   └── test_fusion.py            # Unit tests for fusion attention
├── requirements.txt              # Package dependencies
├── Dockerfile                    # Docker container config
├── docker-compose.yml            # Container volume composition
├── README.md                     # Framework overview
├── api_documentation.md          # REST API docs
├── installation_guide.md         # Deployment and usage guide
└── generate_synthetic_data.py    # Synthetic MRI generator for local execution
```

---

## 🚀 Quick Start Guide

### 1. Set Up Environment
```bash
# Clone the repository
git clone https://github.com/your-username/brain_tumor_ensemble.git
cd brain_tumor_ensemble

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
Before training, generate a synthetic set of brain MRI scans (contains normal slices and slices with implanted lesions):
```bash
python3 generate_synthetic_data.py
```

### 3. Run Unit Tests
Ensure all components are operating correctly:
```bash
python3 -m unittest discover tests/
```

### 4. Train the Ensemble
Fine-tune the backbones and train the attention fusion head:
```bash
python3 src/training.py
```
This script saves trained checkpoints to `checkpoints/` and writes charts and comparative metrics (`logs/metrics.json`) to the static assets directory.

### 5. Launch the Web Application
```bash
python3 src/web/app.py
```
Open [http://localhost:5001](http://localhost:5001) in your browser.
