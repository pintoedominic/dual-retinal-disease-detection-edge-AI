# Dual Retinal Disease Detection - Edge AI

**Edge-deployed dual retinal disease detection system (ROP + DR) on NVIDIA Jetson Orin NX - 93%+ accuracy, real-time inference, no cloud dependency.**

> Master of Engineering Dissertation - Auckland University of Technology (AUT), 2025  
> Author: Dominic Pintoe | Student ID: 23198505  
> Supervisor: Prof. Hamid Gholamhosseini

---

## Overview

This repository contains the full implementation of a two-stage deep learning system for automated detection of **Retinopathy of Prematurity (ROP)** and **Paediatric Diabetic Retinopathy (DR)** on the **NVIDIA Jetson Orin NX** edge computing platform.

The core innovation is a **novel two-stage YOLO11 classification pipeline** that resolves the intractable class imbalance problem present in paediatric retinal datasets - achieving **93.68% end-to-end accuracy** where conventional direct four-class approaches repeatedly failed (< 30% accuracy).

---

## Key Results

| Metric | Value |
|---|---|
| Stage A Accuracy (No ROP vs ROP) | **94.01%** |
| Stage B Accuracy (Severity Grading) | **98.68%** |
| End-to-End 4-Class Accuracy | **93.68%** |
| Jetson Orin NX — ROP Pipeline FPS | **~26.1 FPS** |
| Jetson Orin NX — Combined ROP+DR FPS | **7–9 FPS** |
| YOLO11 Model Size | **10.5 MB (Stage A) / 19.9 MB (Stage B)** |
| ResNet50 Comparison Accuracy | 95.17% (3.1× slower, 5.1× larger) |
| Transfer Learning Impact (Severity) | −5.73% (negative transfer confirmed) |

---

## Architecture: Two-Stage Pipeline

```
Input Image
    │
    ▼
┌─────────────────────┐
│  Stage A — YOLO11s  │   Binary: No ROP vs ROP
│  Accuracy: 94.01%   │
└─────────────────────┘
    │              │
  No ROP          ROP
    │              │
    ▼              ▼
Final: No ROP  ┌─────────────────────┐
               │  Stage B — YOLO11m  │   Severity: Mild / Moderate / Severe
               │  Accuracy: 98.68%   │
               └─────────────────────┘
```

**Mathematical decomposition:**

```
P(class | image) = P(ROP_present | image) × P(severity | image, ROP_present)
```

This factorisation allows each stage to be optimised independently, eliminating the majority-class dominance that causes conventional four-class models to fail on imbalanced medical datasets.

---

## Dataset

**Czech ROP Dataset** — 6,004 high-resolution retinal images from 375 premature infants.

| Class | Count | Percentage |
|---|---|---|
| No ROP (DG 0–3) | 3,743 | 62.3% |
| Mild ROP (DG 4–6) | 138 | 2.3% |
| Moderate ROP (DG 7–10) | 818 | 13.6% |
| Severe ROP (DG 11–13) | 1,311 | 21.8% |

**APTOS 2019 Diabetic Retinopathy Dataset** — 3,662 adult retinal images used for transfer learning experiments.

> **Note:** Datasets are not included in this repository due to licensing restrictions. See [Dataset Setup](#dataset-setup) for access instructions.

---

## Repository Structure

```
dual-retinal-disease-detection-edge-AI/
│
├── scripts/
│   ├── two_stage_inference.py          # End-to-end 4-class evaluation
│   ├── transfer_learning_pipeline.py   # APTOS → ROP transfer learning (4 stages)
│   ├── resnet_two_stage.py             # ResNet50 comparison baseline
│   ├── jetson_app.py                   # Complete ROP+DR Jetson application
│   ├── benchmark.py                    # Integrated system benchmarking
│   ├── jetson_deploy_prep.py           # ONNX export & Jetson preparation
│   ├── export_models.py                # DR model ONNX export
│   └── aptos_setup.py                  # APTOS dataset verification
│
├── configs/
│   ├── training_config.yaml            # Training hyperparameters
│   └── jetson_config.yaml              # Jetson deployment configuration
│
├── docs/
│   ├── SETUP.md                        # Detailed setup instructions
│   ├── JETSON_DEPLOYMENT.md            # Jetson deployment guide
│   └── RESULTS.md                      # Full experimental results
│
├── tests/
│   └── test_inference.py               # Basic inference validation tests
│
├── models/                             # Model weights (not tracked — see docs)
├── datasets/                           # Dataset directory (not tracked)
├── results/                            # Experiment outputs
│
├── requirements.txt
├── requirements_jetson.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/pintoedominic/dual-retinal-disease-detection-edge-AI.git
cd dual-retinal-disease-detection-edge-AI
```

### 2. Install Dependencies

**PC / Development environment:**
```bash
pip install -r requirements.txt
```

**NVIDIA Jetson Orin NX:**
```bash
pip3 install -r requirements_jetson.txt
```

### 3. Dataset Setup

Download the Czech ROP dataset and APTOS 2019 dataset, then configure paths in `configs/training_config.yaml`.

### 4. Train the Two-Stage Model

Training runs sequentially — Stage A trains first, then Stage B uses a larger backbone for the harder severity task.

```bash
# Stage A: Binary screening (No ROP vs ROP)
# Stage B: Severity grading (Mild / Moderate / Severe)
python scripts/two_stage_inference.py
```

### 5. Transfer Learning Experiment

```bash
python scripts/transfer_learning_pipeline.py
```

This runs the full 4-stage pipeline: APTOS pretraining → Mixed training → ROP binary fine-tuning → ROP severity specialisation.

### 6. Run on Jetson Orin NX

```bash
# Export models to ONNX first (on development PC)
python scripts/jetson_deploy_prep.py

# Transfer ONNX models to Jetson, then run:
python3 scripts/jetson_app.py \
  --rop-binary ~/models/rop_binary.onnx \
  --rop-severity ~/models/rop_severity.onnx \
  --dr-binary ~/models/dr_binary.onnx \
  --dr-severity ~/models/dr_severity.onnx \
  --mode both
```

---

## Experimental Results Summary

### Two-Stage YOLO11 — Classification Report

**Stage A (Binary):**
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| ROP | 0.9099 | 0.9339 | 0.9217 |
| No ROP | 0.9592 | 0.9439 | 0.9515 |
| **Overall** | — | — | **94.01%** |

**Stage B (Severity):**
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Mild | 1.0000 | 1.0000 | 1.0000 |
| Moderate | 1.0000 | 0.9674 | 0.9834 |
| Severe | 0.9760 | 1.0000 | 0.9879 |
| **Overall** | — | — | **98.68%** |

**End-to-End 4-Class:**
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| No ROP | 0.9592 | 0.9439 | 0.9515 |
| Mild | 0.7368 | 1.0000 | 0.8485 |
| Moderate | 0.9012 | 0.8902 | 0.8957 |
| Severe | 0.9248 | 0.9389 | 0.9318 |
| **Overall** | — | — | **93.68%** |

### YOLO11 vs ResNet50 Comparison

| Metric | YOLO11 | ResNet50 |
|---|---|---|
| End-to-End Accuracy | 93.68% | **95.17%** |
| Model Size (Stage A) | **10.5 MB** | 94.0 MB |
| Inference Speed (PC) | **12.5 ms** | ~38.8 ms |
| Speed Advantage | **3.1× faster** | — |
| Size Advantage | **5.1× smaller** | — |

### Transfer Learning Results

| Stage | No Transfer | With Transfer | Δ |
|---|---|---|---|
| Binary (Stage A) | 94.01% | 94.01% | ±0.00% |
| Severity (Stage B) | 98.68% | 92.95% | **−5.73%** |

> Transfer learning from adult DR (APTOS) to paediatric ROP **degraded** severity classification by 5.73%, confirming significant domain gap between adult diabetic retinopathy and paediatric ROP pathology.

### Edge Deployment — Jetson Orin NX (ONNX, CPU Provider)

| Pipeline | Avg Latency | FPS |
|---|---|---|
| ROP only | ~37.7 ms | ~26.5 FPS |
| DR only | ~109.2 ms | ~9.2 FPS |
| ROP + DR combined | ~171.9 ms (avg) | 6–9 FPS |

### Optimization Pipeline (PC baseline: 6.2 FPS)

| Format | FPS | Speedup | Accuracy Loss |
|---|---|---|---|
| PyTorch (baseline) | 6.2 | 1.0× | — |
| ONNX | 8.4 | 1.35× | 0.00% |
| TensorRT FP16 | 15.3 | 2.47× | 0.03% |
| TensorRT INT8 | 18.7 | 3.02× | 0.26% |

---

## Model Sizes

| Model | Size |
|---|---|
| ROP Binary (YOLO11s) | 10.5 MB |
| ROP Severity (YOLO11m) | 19.9 MB |
| DR Binary | 19.9 MB |
| DR Severity | 19.9 MB |
| **Total System** | **70.3 MB** |

---

## Dataset Setup

### Czech ROP Dataset
Access through the original dataset authors. Configure the path in `configs/training_config.yaml`:
```yaml
rop_dataset_path: "datasets/czech_rop/"
```

### APTOS 2019 DR Dataset
Available on [Kaggle — APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data).
```yaml
aptos_dataset_path: "datasets/aptos2019/"
```

---

## Hardware Requirements

### Development / Training
- GPU: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- RAM: 16GB+
- Python 3.8+

### Edge Deployment
- NVIDIA Jetson Orin NX 16GB
- JetPack 5.x
- ONNX Runtime 1.19.2 (CPU provider)
- Camera: CSI camera (nvarguscamerasrc) or USB camera

---

## Key Findings

1. **Two-stage decomposition solves class imbalance** — Direct four-class classification achieved < 30% accuracy due to extreme class imbalance (62.3% No ROP). The two-stage approach reached 93.68% end-to-end accuracy.

2. **Negative transfer learning** — Pre-training on adult DR (APTOS) degraded ROP severity classification by 5.73%, demonstrating that retinal morphology differences between adult DR and paediatric ROP are significant enough to cause negative domain transfer.

3. **YOLO11 is deployment-optimal** — Despite ResNet50 achieving marginally higher accuracy (95.17% vs 93.68%), YOLO11 is 3.1× faster and 5.1× smaller, making it decisively better for edge deployment.

4. **Real-time edge deployment is viable** — The Jetson Orin NX achieved 26 FPS for single-disease screening using ONNX runtime, confirming feasibility of edge-deployed clinical screening tools.

5. **Camera quality is a clinical bottleneck** — The CSI camera used in testing could not provide medical-grade retinal image quality, highlighting the gap between research validation and real clinical implementation.

---

## Citation

If you use this work, please cite:

```
Pintoe, D. (2025). Deep Learning Based Early Detection of Paediatric Retinopathy of 
Prematurity and Diabetic Retinopathy on Nvidia Jetson Orin NX. 
Master of Engineering Dissertation, Auckland University of Technology.
```

---

## License

This project is submitted as academic work for Auckland University of Technology. Code is made available for research and educational purposes.

---

## Acknowledgements

- Supervisor: Prof. Hamid Gholamhosseini, Auckland University of Technology
- Czech ROP Dataset contributors
- APTOS 2019 Dataset — Kaggle competition organizers
- Ultralytics YOLO11 framework
- NVIDIA Jetson platform
