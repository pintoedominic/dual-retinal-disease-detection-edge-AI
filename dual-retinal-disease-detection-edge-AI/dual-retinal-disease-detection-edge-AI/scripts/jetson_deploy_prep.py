# jetson_deployment_preparation.py
# Prepare YOLO11 models for Jetson Orin NX deployment

import torch
from ultralytics import YOLO
import time
import os
import psutil
import numpy as np

print("ðŸš€ Jetson Orin NX Deployment Preparation")
print("=" * 50)

# Model paths (your best performing original models)
STAGE_A_WEIGHTS = r"C:\Dominic\Spyder Files\runs\classify\ROP_stageA_bin\weights\best.pt"
STAGE_B_WEIGHTS = r"C:\Dominic\Spyder Files\runs\classify\ROP_stageB_severity\weights\best.pt"

# Test image for benchmarking
TEST_IMAGE = r"C:\Dominic\Datasets Project\ROP_TWO_STAGE\STAGE_A_BIN\test\ROP\006_F_GA40_BW3200_PA44_DG11_PF0_D1_S02_5.jpg"  # Use any test image

# Configuration
IMGSZ = 224
DEVICE = 0 if torch.cuda.is_available() else "cpu"

def benchmark_original_models():
    """Benchmark original YOLO11 models before optimization"""
    print("\nðŸ“Š Step 1: Benchmarking Original Models")
    print("-" * 40)
    
    # Load models
    modelA = YOLO(STAGE_A_WEIGHTS)
    modelB = YOLO(STAGE_B_WEIGHTS)
    
    print(f"âœ… Loaded Stage A: {STAGE_A_WEIGHTS}")
    print(f"âœ… Loaded Stage B: {STAGE_B_WEIGHTS}")
    
    # Model info
    print(f"\nðŸ“‹ Model Information:")
    print(f"Stage A model size: {os.path.getsize(STAGE_A_WEIGHTS) / (1024*1024):.1f} MB")
    print(f"Stage B model size: {os.path.getsize(STAGE_B_WEIGHTS) / (1024*1024):.1f} MB")
    
    # Warm up
    print(f"\nðŸ”¥ Warming up models...")
    for _ in range(5):
        modelA(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
        modelB(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
    
    # Benchmark inference speed
    print(f"\nâ±ï¸  Benchmarking inference speed (100 iterations)...")
    
    # Stage A benchmark
    times_A = []
    for i in range(100):
        start = time.time()
        result_A = modelA(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
        end = time.time()
        times_A.append((end - start) * 1000)  # Convert to ms
    
    # Stage B benchmark
    times_B = []
    for i in range(100):
        start = time.time()
        result_B = modelB(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
        end = time.time()
        times_B.append((end - start) * 1000)  # Convert to ms
    
    # End-to-end benchmark (both stages)
    times_combined = []
    for i in range(100):
        start = time.time()
        
        # Stage A
        result_A = modelA(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
        predA_idx = int(result_A[0].probs.top1)
        predA_name = modelA.names[predA_idx]
        
        # Stage B (if ROP detected)
        if predA_name == "ROP":
            result_B = modelB(TEST_IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
        
        end = time.time()
        times_combined.append((end - start) * 1000)  # Convert to ms
    
    # Results
    avg_A = np.mean(times_A)
    avg_B = np.mean(times_B)
    avg_combined = np.mean(times_combined)
    
    print(f"\nðŸ“Š Inference Results:")
    print(f"Stage A average: {avg_A:.1f} ms ({1000/avg_A:.1f} FPS)")
    print(f"Stage B average: {avg_B:.1f} ms ({1000/avg_B:.1f} FPS)")
    print(f"End-to-end average: {avg_combined:.1f} ms ({1000/avg_combined:.1f} FPS)")
    print(f"P95 latencies: A={np.percentile(times_A, 95):.1f}ms, B={np.percentile(times_B, 95):.1f}ms")
    
    return {
        'stage_a_ms': avg_A,
        'stage_b_ms': avg_B,
        'end_to_end_ms': avg_combined,
        'stage_a_fps': 1000/avg_A,
        'stage_b_fps': 1000/avg_B,
        'end_to_end_fps': 1000/avg_combined
    }

def export_for_jetson():
    """Export models in formats suitable for Jetson deployment"""
    print("\nðŸ”§ Step 2: Exporting Models for Jetson")
    print("-" * 40)
    
    # Load models
    modelA = YOLO(STAGE_A_WEIGHTS)
    modelB = YOLO(STAGE_B_WEIGHTS)
    
    # Export Stage A to ONNX
    print("ðŸ“¦ Exporting Stage A to ONNX...")
    onnx_path_A = modelA.export(
        format='onnx',
        imgsz=IMGSZ,
        dynamic=False,
        half=False,  # Start with FP32, then try FP16
        int8=False,
        device=DEVICE
    )
    print(f"âœ… Stage A ONNX: {onnx_path_A}")
    
    # Export Stage B to ONNX
    print("ðŸ“¦ Exporting Stage B to ONNX...")
    onnx_path_B = modelB.export(
        format='onnx',
        imgsz=IMGSZ,
        dynamic=False,
        half=False,
        int8=False,
        device=DEVICE
    )
    print(f"âœ… Stage B ONNX: {onnx_path_B}")
    
    # Also export to TorchScript for comparison
    print("ðŸ“¦ Exporting to TorchScript...")
    try:
        torchscript_A = modelA.export(format='torchscript', imgsz=IMGSZ)
        torchscript_B = modelB.export(format='torchscript', imgsz=IMGSZ)
        print(f"âœ… TorchScript exports complete")
    except Exception as e:
        print(f"âš ï¸  TorchScript export failed: {e}")
    
    return onnx_path_A, onnx_path_B

def benchmark_onnx_models(onnx_path_A, onnx_path_B):
    """Benchmark ONNX models for comparison"""
    print("\nâ±ï¸  Step 3: Benchmarking ONNX Models")
    print("-" * 40)
    
    try:
        import onnxruntime as ort
        
        # Create ONNX sessions
        session_A = ort.InferenceSession(onnx_path_A)
        session_B = ort.InferenceSession(onnx_path_B)
        
        # Prepare input
        import cv2
        img = cv2.imread(TEST_IMAGE)
        img = cv2.resize(img, (IMGSZ, IMGSZ))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)   # Add batch dimension
        
        # Benchmark ONNX Stage A
        times_onnx_A = []
        for i in range(100):
            start = time.time()
            outputs = session_A.run(None, {session_A.get_inputs()[0].name: img})
            end = time.time()
            times_onnx_A.append((end - start) * 1000)
        
        # Benchmark ONNX Stage B
        times_onnx_B = []
        for i in range(100):
            start = time.time()
            outputs = session_B.run(None, {session_B.get_inputs()[0].name: img})
            end = time.time()
            times_onnx_B.append((end - start) * 1000)
        
        avg_onnx_A = np.mean(times_onnx_A)
        avg_onnx_B = np.mean(times_onnx_B)
        
        print(f"ðŸ“Š ONNX Results:")
        print(f"Stage A ONNX: {avg_onnx_A:.1f} ms ({1000/avg_onnx_A:.1f} FPS)")
        print(f"Stage B ONNX: {avg_onnx_B:.1f} ms ({1000/avg_onnx_B:.1f} FPS)")
        
        return avg_onnx_A, avg_onnx_B
        
    except ImportError:
        print("âš ï¸  ONNX Runtime not available. Install with: pip install onnxruntime")
        return None, None
    except Exception as e:
        print(f"âŒ ONNX benchmarking failed: {e}")
        return None, None

def estimate_jetson_performance(pc_results):
    """Estimate Jetson Orin NX performance based on PC results"""
    print("\nðŸ”® Step 4: Jetson Performance Estimation")
    print("-" * 40)
    
    # Rough performance scaling factors based on hardware differences
    # These are estimates - actual performance may vary
    jetson_scaling_factor = 0.3  # Jetson is roughly 30% of PC GPU performance
    
    print(f"ðŸ“Š Estimated Jetson Orin NX Performance:")
    print(f"(Using {jetson_scaling_factor:.1%} scaling factor)")
    
    estimated_A = pc_results['stage_a_ms'] / jetson_scaling_factor
    estimated_B = pc_results['stage_b_ms'] / jetson_scaling_factor
    estimated_combined = pc_results['end_to_end_ms'] / jetson_scaling_factor
    
    print(f"Stage A: ~{estimated_A:.1f} ms (~{1000/estimated_A:.1f} FPS)")
    print(f"Stage B: ~{estimated_B:.1f} ms (~{1000/estimated_B:.1f} FPS)")
    print(f"End-to-end: ~{estimated_combined:.1f} ms (~{1000/estimated_combined:.1f} FPS)")
    
    # Check if meets real-time requirements
    real_time_threshold = 33.33  # 30 FPS
    meets_realtime = estimated_combined < real_time_threshold
    
    print(f"\nðŸŽ¯ Real-time Performance Assessment:")
    print(f"Target: <33.33ms (30 FPS) for real-time")
    print(f"Estimated: {estimated_combined:.1f} ms")
    print(f"Real-time capable: {'âœ… YES' if meets_realtime else 'âŒ NO'}")
    
    return {
        'estimated_stage_a_ms': estimated_A,
        'estimated_stage_b_ms': estimated_B,
        'estimated_end_to_end_ms': estimated_combined,
        'meets_realtime': meets_realtime
    }

def generate_deployment_summary():
    """Generate summary for dissertation"""
    print("\nðŸ“‹ Step 5: Deployment Summary")
    print("-" * 40)
    
    # System info
    print(f"ðŸ’» Current System:")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU only'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
    
    print(f"\nðŸŽ¯ Deployment Readiness:")
    print(f"âœ… Models exported to ONNX format")
    print(f"âœ… Performance benchmarked")
    print(f"âœ… Jetson deployment estimates calculated")
    print(f"âœ… Ready for physical Jetson testing")

def main():
    """Execute complete deployment preparation"""
    print("ðŸš€ Starting Jetson Deployment Preparation")
    
    # Step 1: Benchmark original models
    pc_results = benchmark_original_models()
    
    # Step 2: Export models
    onnx_A, onnx_B = export_for_jetson()
    
    # Step 3: Benchmark ONNX
    onnx_A_time, onnx_B_time = benchmark_onnx_models(onnx_A, onnx_B)
    
    # Step 4: Estimate Jetson performance
    jetson_estimates = estimate_jetson_performance(pc_results)
    
    # Step 5: Generate summary
    generate_deployment_summary()
    
    print(f"\nâœ… Deployment preparation complete!")
    print(f"ðŸ“ ONNX models ready for Jetson transfer:")
    print(f"   Stage A: {onnx_A}")
    print(f"   Stage B: {onnx_B}")
    
    return pc_results, jetson_estimates

if __name__ == "__main__":
    results = main()
