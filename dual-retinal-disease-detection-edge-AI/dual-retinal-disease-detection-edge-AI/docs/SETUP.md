# Setup Guide

## Development Environment (PC / Workstation)

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (RTX series recommended)
- CUDA 11.8+ and cuDNN 8+

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/dual-retinal-disease-detection-edge-AI.git
cd dual-retinal-disease-detection-edge-AI
pip install -r requirements.txt
```

### Dataset Preparation

#### Czech ROP Dataset
1. Obtain the Czech ROP dataset from the original authors (Štěpánka Rusňáková et al.)
2. Place images in `datasets/czech_rop/images_stack_without_captions/`
3. Image filenames follow the pattern: `XXX_M/F_GAxx_BWxxxx_PAxx_DGxx_PFx_Dx_Sxx_x.jpg`
   - `DG` code determines the diagnosis grade (0–13)

#### APTOS 2019 Diabetic Retinopathy Dataset
1. Download from Kaggle: https://www.kaggle.com/c/aptos2019-blindness-detection
2. Extract to `datasets/aptos2019/archive/`
3. Required files: `train_images/`, `train_1.csv`, `valid.csv`

### Running Training

```bash
# Verify APTOS dataset structure
python scripts/aptos_setup.py

# Run transfer learning pipeline (4 stages)
python scripts/transfer_learning_pipeline.py

# Run end-to-end 4-class evaluation
python scripts/two_stage_inference.py

# Run ResNet50 comparison baseline
python scripts/resnet_two_stage.py
```

### Exporting Models for Jetson

```bash
# Export YOLO11 models to ONNX format
python scripts/jetson_deploy_prep.py

# Export DR transfer learning models to ONNX
python scripts/export_models.py

# Run integrated benchmark
python scripts/benchmark.py
```

## Label Mapping

The Czech ROP dataset uses DG (Diagnosis Grade) codes in filenames:

| DG Code | Class | Stage |
|---|---|---|
| DG 0–3 | No ROP | Stage A: No ROP |
| DG 4–6 | Mild ROP | Stage B: Mild |
| DG 7–10 | Moderate ROP | Stage B: Moderate |
| DG 11–13 | Severe ROP | Stage B: Severe |

## Training Output

Models are saved to `runs/classify/{run_name}/weights/best.pt` by default (Ultralytics standard).
