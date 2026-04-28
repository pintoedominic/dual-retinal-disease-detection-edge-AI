# integrated_rop_dr_benchmark.py
# Comprehensive benchmarking of integrated ROP + DR system

import os
import time
import numpy as np
from ultralytics import YOLO
import torch

print("ðŸš€ Integrated ROP + DR System Benchmarking")
print("=" * 60)

# All model paths
ROP_BINARY_WEIGHTS = r"C:\Dominic\Spyder Files\runs\classify\ROP_stageA_bin\weights\best.pt"
ROP_SEVERITY_WEIGHTS = r"C:\Dominic\Spyder Files\runs\classify\ROP_stageB_severity\weights\best.pt"
DR_BINARY_WEIGHTS = r"C:\Dominic\Spyder Files\runs\classify\ROP_binary_transfer\weights\best.pt"
DR_SEVERITY_WEIGHTS = r"C:\Dominic\Spyder Files\runs\classify\ROP_severity_transfer\weights\best.pt"

# Test image
TEST_IMAGE = r"C:\Dominic\Datasets Project\ROP_TWO_STAGE\STAGE_A_BIN\test\ROP\052_M_GA26_BW820_PA75_DG9_PF0_D1_S08_9.jpg"

# Configuration
IMGSZ = 224
DEVICE = 0 if torch.cuda.is_available() else "cpu"

class IntegratedROPDRSystem:
    """Complete ROP + DR detection system"""
    
    def __init__(self):
        print("ðŸ“¦ Loading all models...")
        
        # Load ROP models
        self.rop_binary = YOLO(ROP_BINARY_WEIGHTS)
        self.rop_severity = YOLO(ROP_SEVERITY_WEIGHTS)
        print(f"âœ… ROP models loaded")
        
        # Load DR models
        self.dr_binary = YOLO(DR_BINARY_WEIGHTS)
        self.dr_severity = YOLO(DR_SEVERITY_WEIGHTS)
        print(f"âœ… DR models loaded")
        
        print(f"ðŸŽ¯ Integrated system ready!")
    
    def detect_rop(self, image):
        """ROP detection pipeline"""
        start_time = time.time()
        
        # Stage A: No ROP vs ROP
        binary_result = self.rop_binary(image, imgsz=IMGSZ, device=DEVICE, verbose=False)
        pred_idx = int(binary_result[0].probs.top1)
        pred_name = self.rop_binary.names[pred_idx]
        
        if pred_name == "no_ROP":
            final_result = "No ROP"
            confidence = float(binary_result[0].probs.top1conf)
        else:
            # Stage B: ROP severity
            severity_result = self.rop_severity(image, imgsz=IMGSZ, device=DEVICE, verbose=False)
            severity_idx = int(severity_result[0].probs.top1)
            severity_name = self.rop_severity.names[severity_idx]
            final_result = f"ROP - {severity_name}"
            confidence = float(severity_result[0].probs.top1conf)
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'result': final_result,
            'confidence': confidence,
            'inference_time_ms': inference_time,
            'pipeline': 'ROP'
        }
    
    def detect_dr(self, image):
        """DR detection pipeline"""
        start_time = time.time()
        
        # Stage A: No pathology vs pathology
        binary_result = self.dr_binary(image, imgsz=IMGSZ, device=DEVICE, verbose=False)
        pred_idx = int(binary_result[0].probs.top1)
        pred_name = self.dr_binary.names[pred_idx]
        
        if pred_name == "no_pathology":
            final_result = "No DR"
            confidence = float(binary_result[0].probs.top1conf)
        else:
            # Stage B: DR severity
            severity_result = self.dr_severity(image, imgsz=IMGSZ, device=DEVICE, verbose=False)
            severity_idx = int(severity_result[0].probs.top1)
            severity_name = self.dr_severity.names[severity_idx]
            final_result = f"DR - {severity_name}"
            confidence = float(severity_result[0].probs.top1conf)
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'result': final_result,
            'confidence': confidence,
            'inference_time_ms': inference_time,
            'pipeline': 'DR'
        }
    
    def detect_both(self, image):
        """Run both ROP and DR detection"""
        start_time = time.time()
        
        rop_result = self.detect_rop(image)
        dr_result = self.detect_dr(image)
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'rop_result': rop_result,
            'dr_result': dr_result,
            'total_time_ms': total_time
        }
    
    def get_model_sizes(self):
        """Get total model sizes"""
        sizes = {}
        
        models = {
            'rop_binary': ROP_BINARY_WEIGHTS,
            'rop_severity': ROP_SEVERITY_WEIGHTS,
            'dr_binary': DR_BINARY_WEIGHTS,
            'dr_severity': DR_SEVERITY_WEIGHTS
        }
        
        total_size = 0
        for name, path in models.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024*1024)
                sizes[name] = size_mb
                total_size += size_mb
            else:
                sizes[name] = 0
        
        sizes['total'] = total_size
        return sizes

def benchmark_integrated_system():
    """Comprehensive benchmarking of integrated system"""
    print("\nðŸ“Š Comprehensive System Benchmarking")
    print("=" * 60)
    
    # Initialize system
    system = IntegratedROPDRSystem()
    
    # Get model sizes
    sizes = system.get_model_sizes()
    print(f"\nðŸ“‹ Model Sizes:")
    print(f"ROP Binary: {sizes['rop_binary']:.1f} MB")
    print(f"ROP Severity: {sizes['rop_severity']:.1f} MB")
    print(f"DR Binary: {sizes['dr_binary']:.1f} MB")
    print(f"DR Severity: {sizes['dr_severity']:.1f} MB")
    print(f"Total System: {sizes['total']:.1f} MB")
    
    # Warm up
    print(f"\nðŸ”¥ Warming up integrated system...")
    for _ in range(5):
        system.detect_rop(TEST_IMAGE)
        system.detect_dr(TEST_IMAGE)
    
    # Benchmark individual pipelines
    print(f"\nâ±ï¸  Benchmarking individual pipelines (100 iterations each)...")
    
    # ROP pipeline benchmark
    rop_times = []
    for i in range(100):
        result = system.detect_rop(TEST_IMAGE)
        rop_times.append(result['inference_time_ms'])
    
    # DR pipeline benchmark
    dr_times = []
    for i in range(100):
        result = system.detect_dr(TEST_IMAGE)
        dr_times.append(result['inference_time_ms'])
    
    # Both pipelines benchmark
    both_times = []
    for i in range(100):
        result = system.detect_both(TEST_IMAGE)
        both_times.append(result['total_time_ms'])
    
    # Calculate statistics
    rop_avg = np.mean(rop_times)
    dr_avg = np.mean(dr_times)
    both_avg = np.mean(both_times)
    
    print(f"\nðŸ“Š Performance Results:")
    print(f"ROP Pipeline: {rop_avg:.1f} ms ({1000/rop_avg:.1f} FPS)")
    print(f"DR Pipeline: {dr_avg:.1f} ms ({1000/dr_avg:.1f} FPS)")
    print(f"Both Pipelines: {both_avg:.1f} ms ({1000/both_avg:.1f} FPS)")
    
    print(f"\nP95 Latencies:")
    print(f"ROP: {np.percentile(rop_times, 95):.1f} ms")
    print(f"DR: {np.percentile(dr_times, 95):.1f} ms")
    print(f"Both: {np.percentile(both_times, 95):.1f} ms")
    
    return {
        'rop_avg_ms': rop_avg,
        'dr_avg_ms': dr_avg,
        'both_avg_ms': both_avg,
        'rop_fps': 1000/rop_avg,
        'dr_fps': 1000/dr_avg,
        'both_fps': 1000/both_avg,
        'total_model_size_mb': sizes['total']
    }

def estimate_jetson_performance(pc_results):
    """Estimate Jetson performance for integrated system"""
    print(f"\nðŸ”® Jetson Orin NX Performance Estimation")
    print("=" * 60)
    
    # Performance scaling factor (conservative estimate)
    scaling_factor = 0.3
    
    # Estimate individual pipeline performance
    rop_jetson = pc_results['rop_avg_ms'] / scaling_factor
    dr_jetson = pc_results['dr_avg_ms'] / scaling_factor
    both_jetson = pc_results['both_avg_ms'] / scaling_factor
    
    print(f"ðŸ“Š Estimated Jetson Performance:")
    print(f"(Using {scaling_factor:.1%} scaling factor)")
    print(f"ROP Pipeline: ~{rop_jetson:.1f} ms (~{1000/rop_jetson:.1f} FPS)")
    print(f"DR Pipeline: ~{dr_jetson:.1f} ms (~{1000/dr_jetson:.1f} FPS)")
    print(f"Both Pipelines: ~{both_jetson:.1f} ms (~{1000/both_jetson:.1f} FPS)")
    
    # Clinical usage scenarios
    print(f"\nðŸ¥ Clinical Usage Scenarios:")
    print(f"Single Disease Screening: ~{max(rop_jetson, dr_jetson):.1f} ms")
    print(f"Comprehensive Screening: ~{both_jetson:.1f} ms")
    print(f"Memory Requirements: ~{pc_results['total_model_size_mb']:.1f} MB models + ~4GB inference")
    
    # Real-time assessment
    real_time_threshold = 33.33  # 30 FPS
    
    print(f"\nðŸŽ¯ Real-time Assessment:")
    print(f"Target: <{real_time_threshold:.1f} ms (30 FPS)")
    print(f"ROP: {'âœ… YES' if rop_jetson < real_time_threshold else 'âš ï¸  BORDERLINE'}")
    print(f"DR: {'âœ… YES' if dr_jetson < real_time_threshold else 'âš ï¸  BORDERLINE'}")
    print(f"Both: {'âœ… YES' if both_jetson < real_time_threshold else 'âŒ NO'}")
    
    return {
        'rop_jetson_ms': rop_jetson,
        'dr_jetson_ms': dr_jetson,
        'both_jetson_ms': both_jetson,
        'meets_realtime_individual': max(rop_jetson, dr_jetson) < real_time_threshold,
        'meets_realtime_both': both_jetson < real_time_threshold
    }

def export_all_models_to_onnx():
    """Export all models to ONNX if not already done"""
    print(f"\nðŸ”§ Exporting All Models to ONNX")
    print("=" * 60)
    
    models_to_export = [
        (ROP_BINARY_WEIGHTS, "ROP Binary"),
        (ROP_SEVERITY_WEIGHTS, "ROP Severity"),
        (DR_BINARY_WEIGHTS, "DR Binary"),
        (DR_SEVERITY_WEIGHTS, "DR Severity")
    ]
    
    onnx_paths = []
    
    for model_path, model_name in models_to_export:
        print(f"ðŸ“¦ Exporting {model_name}...")
        try:
            model = YOLO(model_path)
            onnx_path = model.export(
                format='onnx',
                imgsz=IMGSZ,
                dynamic=False,
                half=False,
                int8=False,
                device=DEVICE
            )
            print(f"âœ… {model_name} ONNX: {onnx_path}")
            onnx_paths.append(onnx_path)
        except Exception as e:
            print(f"âŒ {model_name} export failed: {e}")
            onnx_paths.append(None)
    
    return onnx_paths

def main():
    """Execute complete integrated system analysis"""
    print("ðŸš€ Starting Integrated ROP + DR System Analysis")
    
    # Step 1: Benchmark integrated system
    pc_results = benchmark_integrated_system()
    
    # Step 2: Estimate Jetson performance
    jetson_estimates = estimate_jetson_performance(pc_results)
    
    # Step 3: Export all models to ONNX
    onnx_paths = export_all_models_to_onnx()
    
    # Step 4: Final summary
    print(f"\nâœ… Integrated System Analysis Complete!")
    print(f"=" * 60)
    
    print(f"\nðŸ“Š PC Performance Summary:")
    print(f"ROP Pipeline: {pc_results['rop_avg_ms']:.1f} ms ({pc_results['rop_fps']:.1f} FPS)")
    print(f"DR Pipeline: {pc_results['dr_avg_ms']:.1f} ms ({pc_results['dr_fps']:.1f} FPS)")
    print(f"Complete System: {pc_results['both_avg_ms']:.1f} ms ({pc_results['both_fps']:.1f} FPS)")
    print(f"Total Model Size: {pc_results['total_model_size_mb']:.1f} MB")
    
    print(f"\nðŸ”® Jetson Estimates:")
    print(f"ROP: ~{jetson_estimates['rop_jetson_ms']:.1f} ms")
    print(f"DR: ~{jetson_estimates['dr_jetson_ms']:.1f} ms")
    print(f"Both: ~{jetson_estimates['both_jetson_ms']:.1f} ms")
    
    print(f"\nðŸ“ ONNX Files Ready:")
    for i, (_, name) in enumerate([(ROP_BINARY_WEIGHTS, "ROP Binary"), 
                                   (ROP_SEVERITY_WEIGHTS, "ROP Severity"),
                                   (DR_BINARY_WEIGHTS, "DR Binary"), 
                                   (DR_SEVERITY_WEIGHTS, "DR Severity")]):
        if i < len(onnx_paths) and onnx_paths[i]:
            print(f"   {name}: {onnx_paths[i]}")
        else:
            print(f"   {name}: Export failed")
    
    return pc_results, jetson_estimates, onnx_paths

if __name__ == "__main__":
    results = main()
    print(f"\nðŸŽ‰ Ready for complete Jetson deployment!")