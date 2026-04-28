# Experimental Results

Full experimental results for the Two-Stage YOLO11 ROP/DR Detection System.

---

## Two-Stage YOLO11 — Primary Results

### Stage A: Binary Classification (No ROP vs ROP)

**Test Set: 601 images**

```
Stage A Test Accuracy: 94.01%

Confusion Matrix (rows=true, cols=pred):
           No ROP   ROP
No ROP  [  212     15 ]
ROP     [   21    353 ]

Classification Report:
              precision    recall  f1-score   support
ROP             0.9099    0.9339    0.9217       227
no_ROP          0.9592    0.9439    0.9515       374
accuracy                            0.9401       601
macro avg       0.9346    0.9389    0.9366       601
weighted avg    0.9406    0.9401    0.9402       601
```

### Stage B: Severity Classification (Mild / Moderate / Severe)

**Test Set: 227 ROP-positive images**

```
Stage B Test Accuracy: 98.68%

Confusion Matrix (rows=true, cols=pred):
           Mild   Mod   Sev
Mild     [  13     0     0 ]
Moderate [   0    89     3 ]
Severe   [   0     0   122 ]

Classification Report:
              precision    recall  f1-score   support
mild            1.0000    1.0000    1.0000        13
moderate        1.0000    0.9674    0.9834        92
severe          0.9760    1.0000    0.9879       122
accuracy                            0.9868       227
macro avg       0.9920    0.9891    0.9904       227
weighted avg    0.9871    0.9868    0.9868       227
```

### End-to-End 4-Class Evaluation

```
End-to-end 4-class Test Accuracy: 93.68%

Confusion Matrix (rows=true, cols=pred):
           no_ROP  mild  moderate  severe
no_ROP   [  353     4       8       9  ]
mild     [    0    14       0       0  ]
moderate [    7     1      73       1  ]
severe   [    8     0       0     123  ]

Classification Report:
              precision    recall  f1-score   support
no_ROP          0.9592    0.9439    0.9515       374
mild            0.7368    1.0000    0.8485        14
moderate        0.9012    0.8902    0.8957        82
severe          0.9248    0.9389    0.9318       131
accuracy                            0.9368       601
macro avg       0.8805    0.9433    0.9069       601
weighted avg    0.9386    0.9368    0.9372       601
```

---

## ResNet50 Comparison

### Stage A: Binary Classification

```
Stage A ResNet Accuracy: 95.17%

Confusion Matrix:
           No ROP   ROP
No ROP  [  208     19 ]
ROP     [   10    364 ]

Classification Report:
              precision    recall  f1-score   support
ROP             0.9541    0.9163    0.9348       227
no_ROP          0.9504    0.9733    0.9617       374
accuracy                            0.9517       601
```

### Stage B: Severity Classification

```
Stage B ResNet Accuracy: 99.12%

Confusion Matrix:
           Mild   Mod   Sev
Mild     [  13     0     0 ]
Moderate [   1    91     0 ]
Severe   [   0     1   121 ]

Classification Report:
              precision    recall  f1-score   support
mild            0.9286    1.0000    0.9630        13
moderate        0.9891    0.9891    0.9891        92
severe          1.0000    0.9918    0.9959       122
accuracy                            0.9912       227
```

### End-to-End ResNet

```
Two-Stage ResNet End-to-End Accuracy: 95.17%

Confusion Matrix:
           no_ROP  mild  moderate  severe
no_ROP   [  364     2       0       8  ]
mild     [    0    14       0       0  ]
moderate [    5     0      77       0  ]
severe   [   14     0       0     117  ]
```

---

## Transfer Learning Results (APTOS → ROP)

```
Binary (Stage A):
  Original (no transfer):   94.01%
  With transfer learning:   94.01%
  Improvement:              +0.00%

Severity (Stage B):
  Original (no transfer):   98.68%
  With transfer learning:   92.95%
  Improvement:              -5.73%  ← Negative transfer
```

### Transfer Severity Confusion Matrix

```
           Mild   Mod   Sev
Mild     [  13     0     0 ]
Moderate [   0    85     7 ]
Severe   [   0     9   113 ]
```

---

## PC Inference Benchmark

**System: Development PC with NVIDIA GPU**

```
ROP Pipeline:    12.5 ms  (79.8 FPS)
DR Pipeline:     13.3 ms  (75.3 FPS)
Both Pipelines:  26.8 ms  (37.3 FPS)

P95 Latencies:
  ROP:   13.7 ms
  DR:    15.1 ms
  Both:  32.5 ms

Total Model Size: 70.3 MB
  ROP Binary:   10.5 MB
  ROP Severity: 19.9 MB
  DR Binary:    19.9 MB
  DR Severity:  19.9 MB
```

---

## Jetson Orin NX Benchmark

**ONNX Runtime 1.19.2, CPU provider (AzureExecutionProvider, CPUExecutionProvider)**

### Dummy Inference (no camera)

```
ROP Binary ONNX:    17.0 ms
ROP Severity ONNX:  35.0 ms
DR Binary ONNX:     37.7 ms
DR Severity ONNX:   42.9 ms
```

### Live Camera Performance (CSI, 1280×720)

```
Final ROP Performance (289 frames):
  Average FPS:     26.1
  Average latency: 37.7 ms
  P95 latency:     38.5 ms

Final DR Performance (82 frames):
  Average FPS:     9.4
  Average latency: 109.2 ms
  P95 latency:     108.1 ms

Final BOTH Performance (667 frames):
  Average FPS:     6.0
  Average latency: 171.9 ms
  P95 latency:     353.9 ms
```

---

## Optimization Pipeline

| Format | FPS | Speedup vs Baseline | Accuracy Loss |
|---|---|---|---|
| PyTorch FP32 (baseline) | 6.2 | 1.00× | — |
| ONNX | 8.4 | 1.35× | 0.00% |
| TensorRT FP16 | 15.3 | 2.47× | 0.03% |
| TensorRT INT8 | 18.7 | 3.02× | 0.26% |
