# export_dr_transfer_models.py
# Export DR transfer learning models to ONNX format

import os
import time
import numpy as np
from ultralytics import YOLO
import torch

print("ðŸš€ Exporting DR Transfer Learning Models to ONNX")
print("=" * 60)

# Model paths (your transfer learning models)
DR_BINARY_WEIGHTS = r"C:\Dominic\Spyder Files\runs\classify\ROP_binary_transfer\weights\best.pt"
DR_SEVERITY_WEIGHTS = r"C:\Dominic\Spyder Files\runs\classify\ROP_severity_transfer\weights\best.pt"

# Test image for benchmarking (use any test image)
TEST_IMAGE = r"C:\Dominic\Datasets Project\ROP_TWO_STAGE\STAGE_A_BIN\test\ROP\013_F_GA39_BW3390_PA42_DG11_PF0_D1_S02_14.jpg"

# Configuration
IMGSZ = 224
DEVICE = 0 if torch.cuda.is_available() else "cpu"

def verify_models_exist():
    """Verify that transfer learning models exist"""
    print("ðŸ” Verifying DR transfer learning models...")
    
    models_exist = True
    
    if os.path.exists(DR_BINARY_WEIGHTS):
        size_mb = os.path.getsize(DR_BINARY_WEIGHTS) / (1024*1024)
        print(f"âœ… DR Binary model found: {DR_BINARY_WEIGHTS} ({size_mb:.1f} MB)")
    else:
        print(f"âŒ DR Binary model not found: {DR_BINARY_WEIGHTS}")
        models_exist = False
    
    if os.path.exists(DR_SEVERITY_WEIGHTS):
        size_mb = os.path.getsize(DR_SEVERITY_WEIGHTS) / (1024*1024)
        print(f"âœ… DR Severity model found: {DR_SEVERITY_WEIGHTS} ({size_mb:.1f} MB)")
    else:
        print(f"âŒ DR Severity model not found: {DR_SEVERITY_WEIGHTS}")
        models_exist = False
    
    return models_exist

def benchmark_dr_models():
    """Benchmark DR transfer learning models before export"""
    print("\nðŸ“Š Benchmarking DR Transfer Learning Models")
    print("-" * 50)
    
    # Load models
    print("ðŸ“¦ Loading DR models...")
    dr_binary = YOLO(DR_BINARY_WEIGHTS)
    dr_severity = YOLO(DR_SEVERITY_WEIGHTS)
    
    print(f"âœ… DR Binary loaded: {dr_binary.names}")
    print(f"âœ… DR Severity loaded: {dr_severity.names}")
    
    # Warm up
    print("ðŸ”¥ Warming up models...")
    for _ in range(5):
        dr_binary(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
        dr_severity(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
    
    # Benchmark inference speed
    print("â±ï¸  Benchmarking inference speed (100 iterations)...")
    
    # DR Binary benchmark
    times_binary = []
    for i in range(100):
        start = time.time()
        result = dr_binary(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
        end = time.time()
        times_binary.append((end - start) * 1000)
    
    # DR Severity benchmark
    times_severity = []
    for i in range(100):
        start = time.time()
        result = dr_severity(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
        end = time.time()
        times_severity.append((end - start) * 1000)
    
    # End-to-end DR pipeline benchmark
    times_dr_pipeline = []
    for i in range(100):
        start = time.time()
        
        # DR Binary stage
        binary_result = dr_binary(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
        pred_idx = int(binary_result[0].probs.top1)
        pred_name = dr_binary.names[pred_idx]
        
        # DR Severity stage (if pathology detected)
        if pred_name == "pathology":
            severity_result = dr_severity(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
        
        end = time.time()
        times_dr_pipeline.append((end - start) * 1000)
    
    # Calculate results
    avg_binary = np.mean(times_binary)
    avg_severity = np.mean(times_severity)
    avg_pipeline = np.mean(times_dr_pipeline)
    
    print(f"\nðŸ“Š DR Transfer Learning Performance:")
    print(f"DR Binary: {avg_binary:.1f} ms ({1000/avg_binary:.1f} FPS)")
    print(f"DR Severity: {avg_severity:.1f} ms ({1000/avg_severity:.1f} FPS)")
    print(f"DR Pipeline: {avg_pipeline:.1f} ms ({1000/avg_pipeline:.1f} FPS)")
    print(f"P95 latencies: Binary={np.percentile(times_binary, 95):.1f}ms, Severity={np.percentile(times_severity, 95):.1f}ms")
    
    return {
        'dr_binary_ms': avg_binary,
        'dr_severity_ms': avg_severity,
        'dr_pipeline_ms': avg_pipeline,
        'dr_binary_fps': 1000/avg_binary,
        'dr_severity_fps': 1000/avg_severity,
        'dr_pipeline_fps': 1000/avg_pipeline
    }

def export_dr_models():
    """Export DR models to ONNX format"""
    print("\nðŸ”§ Exporting DR Models to ONNX")
    print("-" * 50)
    
    # Load models
    dr_binary = YOLO(DR_BINARY_WEIGHTS)
    dr_severity = YOLO(DR_SEVERITY_WEIGHTS)
    
    # Export DR Binary to ONNX
    print("ðŸ“¦ Exporting DR Binary model to ONNX...")
    try:
        onnx_binary_path = dr_binary.export(
            format='onnx',
            imgsz=IMGSZ,
            dynamic=False,
            half=False,
            int8=False,
            device=DEVICE
        )
        print(f"âœ… DR Binary ONNX: {onnx_binary_path}")
    except Exception as e:
        print(f"âŒ DR Binary export failed: {e}")
        return None, None
    
    # Export DR Severity to ONNX
    print("ðŸ“¦ Exporting DR Severity model to ONNX...")
    try:
        onnx_severity_path = dr_severity.export(
            format='onnx',
            imgsz=IMGSZ,
            dynamic=False,
            half=False,
            int8=False,
            device=DEVICE
        )
        print(f"âœ… DR Severity ONNX: {onnx_severity_path}")
    except Exception as e:
        print(f"âŒ DR Severity export failed: {e}")
        return onnx_binary_path if 'onnx_binary_path' in locals() else None, None
    
    return onnx_binary_path, onnx_severity_path

def benchmark_dr_onnx(onnx_binary_path, onnx_severity_path):
    """Benchmark DR ONNX models"""
    print("\nâ±ï¸  Benchmarking DR ONNX Models")
    print("-" * 50)
    
    try:
        import onnxruntime as ort
        
        # Create ONNX sessions
        session_binary = ort.InferenceSession(onnx_binary_path)
        session_severity = ort.InferenceSession(onnx_severity_path)
        
        # Prepare input
        import cv2
        img = cv2.imread(TEST_IMAGE)
        img = cv2.resize(img, (IMGSZ, IMGSZ))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # Benchmark ONNX DR Binary
        times_onnx_binary = []
        for i in range(100):
            start = time.time()
            outputs = session_binary.run(None, {session_binary.get_inputs()[0].name: img})
            end = time.time()
            times_onnx_binary.append((end - start) * 1000)
        
        # Benchmark ONNX DR Severity
        times_onnx_severity = []
        for i in range(100):
            start = time.time()
            outputs = session_severity.run(None, {session_severity.get_inputs()[0].name: img})
            end = time.time()
            times_onnx_severity.append((end - start) * 1000)
        
        avg_onnx_binary = np.mean(times_onnx_binary)
        avg_onnx_severity = np.mean(times_onnx_severity)
        
        print(f"ðŸ“Š DR ONNX Results:")
        print(f"DR Binary ONNX: {avg_onnx_binary:.1f} ms ({1000/avg_onnx_binary:.1f} FPS)")
        print(f"DR Severity ONNX: {avg_onnx_severity:.1f} ms ({1000/avg_onnx_severity:.1f} FPS)")
        
        return avg_onnx_binary, avg_onnx_severity
        
    except ImportError:
        print("âš ï¸  ONNX Runtime not available. Install with: pip install onnxruntime")
        return None, None
    except Exception as e:
        print(f"âŒ DR ONNX benchmarking failed: {e}")
        return None, None

def main():
    """Execute complete DR model export pipeline"""
    print("ðŸš€ Starting DR Transfer Learning Model Export")
    
    # Step 1: Verify models exist
    if not verify_models_exist():
        print("âŒ Required models not found. Please check paths.")
        return False
    
    # Step 2: Benchmark original models
    dr_performance = benchmark_dr_models()
    
    # Step 3: Export to ONNX
    onnx_binary, onnx_severity = export_dr_models()
    if not onnx_binary or not onnx_severity:
        print("âŒ Export failed. Cannot continue.")
        return False
    
    # Step 4: Benchmark ONNX models
    onnx_binary_time, onnx_severity_time = benchmark_dr_onnx(onnx_binary, onnx_severity)
    
    # Step 5: Summary
    print(f"\nâœ… DR Model Export Complete!")
    print(f"ðŸ“ Files ready for Jetson deployment:")
    print(f"   DR Binary: {onnx_binary}")
    print(f"   DR Severity: {onnx_severity}")
    
    print(f"\nðŸ“Š DR Performance Summary:")
    print(f"PyTorch Pipeline: {dr_performance['dr_pipeline_ms']:.1f} ms ({dr_performance['dr_pipeline_fps']:.1f} FPS)")
    if onnx_binary_time and onnx_severity_time:
        onnx_pipeline_time = onnx_binary_time + onnx_severity_time
        print(f"ONNX Pipeline: ~{onnx_pipeline_time:.1f} ms (~{1000/onnx_pipeline_time:.1f} FPS)")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\nðŸŽ‰ Ready to proceed with integrated system deployment!")
    else:
        print("\nâŒ Export failed. Please check error messages.")