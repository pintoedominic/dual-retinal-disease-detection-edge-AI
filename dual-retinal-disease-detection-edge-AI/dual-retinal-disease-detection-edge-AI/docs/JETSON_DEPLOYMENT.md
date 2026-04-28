# Jetson Orin NX Deployment Guide

## Hardware Used

- **Device:** NVIDIA Jetson Orin NX 16GB
- **OS:** Ubuntu (JetPack 5.x)
- **Camera:** CSI camera (nvarguscamerasrc)
- **ONNX Runtime:** 1.19.2 (CPU provider — `AzureExecutionProvider`, `CPUExecutionProvider`)

> **Note:** GPU provider for ONNX Runtime is not available via pip for Jetson aarch64. CPU provider was used for all Jetson inference measurements.

## Deployment Steps

### 1. Install ONNX Runtime on Jetson

```bash
pip3 install onnxruntime
# onnxruntime-gpu is not available for aarch64 via pip
# For GPU acceleration, build from source or use TensorRT directly
```

### 2. Transfer ONNX Models to Jetson

```bash
# On development PC — export models first:
python scripts/jetson_deploy_prep.py

# Transfer to Jetson via SCP:
scp runs/classify/ROP_stageA_bin/weights/best.onnx jetson@<IP>:~/models/rop_binary.onnx
scp runs/classify/ROP_stageB_severity/weights/best.onnx jetson@<IP>:~/models/rop_severity.onnx
scp runs/classify/ROP_binary_transfer/weights/best.onnx jetson@<IP>:~/models/dr_binary.onnx
scp runs/classify/ROP_severity_transfer/weights/best.onnx jetson@<IP>:~/models/dr_severity.onnx
```

### 3. Verify Deployment

```bash
# On Jetson — verify model loading and dummy inference:
ls -lh ~/models/

# Expected model sizes:
# rop_binary.onnx    ~20 MB
# rop_severity.onnx  ~40 MB
# dr_binary.onnx     ~40 MB
# dr_severity.onnx   ~40 MB
```

### 4. Run the Application

```bash
# Combined ROP + DR detection (both pipelines):
python3 scripts/jetson_app.py \
  --rop-binary ~/models/rop_binary.onnx \
  --rop-severity ~/models/rop_severity.onnx \
  --dr-binary ~/models/dr_binary.onnx \
  --dr-severity ~/models/dr_severity.onnx \
  --mode both

# ROP only (faster — ~26 FPS):
python3 scripts/jetson_app.py \
  --rop-binary ~/models/rop_binary.onnx \
  --rop-severity ~/models/rop_severity.onnx \
  --dr-binary ~/models/dr_binary.onnx \
  --dr-severity ~/models/dr_severity.onnx \
  --mode rop
```

### Detection Modes
- `rop` — ROP detection only (~26 FPS on Jetson)
- `dr` — DR detection only (~9 FPS on Jetson)
- `both` — Combined ROP + DR (6–9 FPS on Jetson)
- `auto` — Selects pipeline based on patient context (age, type)

### Camera Setup

The CSI camera is automatically detected. The application tries configurations in this order:
1. CSI camera via `nvarguscamerasrc` (Jetson native — recommended)
2. USB camera via `/dev/video0`
3. V4L2 pipeline

### Performance Measured on Jetson Orin NX

| Pipeline | Avg FPS | Avg Latency |
|---|---|---|
| ROP only | ~26.1 FPS | ~37.7 ms |
| DR only | ~9.2 FPS | ~109.2 ms |
| Both combined | ~6–9 FPS | ~172 ms |

### Known Limitations

1. **Camera image quality:** The CSI camera used during testing does not provide medical-grade retinal image resolution. A dedicated fundus camera or RetCam would be required for clinical deployment.
2. **ONNX GPU provider:** Not available via pip for Jetson aarch64. TensorRT integration would provide additional speedup for production use.
3. **Continuous operation:** Extended operation may cause thermal throttling, which was not characterised in this study.
