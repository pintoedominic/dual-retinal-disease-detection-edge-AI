# complete_rop_dr_jetson_app.py
# Complete ROP + DR detection application for Jetson Orin NX

import cv2
import numpy as np
import time
import argparse
import os
from pathlib import Path
import logging
import json

# Try importing optimized inference engines
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("âŒ ONNX Runtime not available")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("âŒ Ultralytics YOLO not available")

class PerformanceMonitor:
    """Monitor system performance during inference"""
    
    def __init__(self):
        self.inference_times = {'rop': [], 'dr': [], 'both': []}
        self.start_time = None
        self.frame_count = 0
        
    def start_inference(self):
        self.start_time = time.time()
        
    def end_inference(self, pipeline_type='both'):
        if self.start_time:
            inference_time = time.time() - self.start_time
            self.inference_times[pipeline_type].append(inference_time)
            self.frame_count += 1
            return inference_time
        return 0
    
    def get_fps(self, pipeline_type='both'):
        if len(self.inference_times[pipeline_type]) > 0:
            avg_time = np.mean(self.inference_times[pipeline_type][-30:])  # Last 30 frames
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0
    
    def get_stats(self, pipeline_type='both'):
        if len(self.inference_times[pipeline_type]) == 0:
            return {"fps": 0, "avg_latency_ms": 0, "p95_latency_ms": 0}
        
        times_ms = np.array(self.inference_times[pipeline_type]) * 1000
        return {
            "fps": self.get_fps(pipeline_type),
            "avg_latency_ms": np.mean(times_ms),
            "p95_latency_ms": np.percentile(times_ms, 95),
            "frame_count": len(self.inference_times[pipeline_type])
        }

class IntegratedROPDRDetector:
    """Complete ROP + DR detection system for Jetson"""
    
    def __init__(self, rop_binary_path, rop_severity_path, dr_binary_path, dr_severity_path, 
                 device='gpu', img_size=224):
        self.img_size = img_size
        self.device = device
        
        print(f"ðŸ“¦ Loading integrated ROP + DR detection system...")
        
        # Load ROP models
        self.rop_binary_model = self._load_model(rop_binary_path, "ROP Binary")
        self.rop_severity_model = self._load_model(rop_severity_path, "ROP Severity")
        
        # Load DR models
        self.dr_binary_model = self._load_model(dr_binary_path, "DR Binary")
        self.dr_severity_model = self._load_model(dr_severity_path, "DR Severity")
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        
        print(f"âœ… Integrated system initialized")
        print(f"   Image size: {img_size}x{img_size}")
        print(f"   Device: {device}")
    
    def _load_model(self, model_path, stage_name):
        """Load model with fallback priority: ONNX -> YOLO"""
        
        model_path = Path(model_path)
        
        # Try ONNX first (.onnx files)
        if ONNX_AVAILABLE:
            onnx_path = model_path.with_suffix('.onnx')
            if onnx_path.exists():
                print(f"ðŸ“¦ Loading {stage_name} ONNX model: {onnx_path}")
                return self._load_onnx_model(str(onnx_path))
        
        # Try YOLO (.pt files)
        if YOLO_AVAILABLE:
            if model_path.suffix == '.pt' and model_path.exists():
                print(f"ðŸ“¦ Loading {stage_name} YOLO model: {model_path}")
                return YOLO(str(model_path))
        
        raise ValueError(f"âŒ Could not load {stage_name} model from {model_path}")
    
    def _load_onnx_model(self, onnx_path):
        """Load ONNX model with GPU acceleration"""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self.device == 'cpu':
            providers = ['CPUExecutionProvider']
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        return session
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        # Resize to model input size
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_onnx(self, model, image):
        """Run inference on ONNX model"""
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: image})
        return outputs[0]
    
    def predict_yolo(self, model, original_image):
        """Run inference on YOLO model"""
        results = model(original_image, imgsz=self.img_size, verbose=False)
        return results[0]
    
    def detect_rop(self, image):
        """
        ROP detection pipeline
        
        Returns:
            dict: {
                'prediction': str,
                'confidence': float,
                'stage_a_result': str,
                'stage_b_result': str
            }
        """
        try:
            # Stage A: Binary classification (No ROP vs ROP)
            if isinstance(self.rop_binary_model, ort.InferenceSession):
                # ONNX inference
                processed_img = self.preprocess_image(image)
                stage_a_output = self.predict_onnx(self.rop_binary_model, processed_img)
                stage_a_pred_idx = np.argmax(stage_a_output[0])
                stage_a_confidence = np.max(stage_a_output[0])
                stage_a_result = "ROP" if stage_a_pred_idx == 1 else "no_ROP"
            else:
                # YOLO inference
                stage_a_result_obj = self.predict_yolo(self.rop_binary_model, image)
                stage_a_pred_idx = int(stage_a_result_obj.probs.top1)
                stage_a_confidence = float(stage_a_result_obj.probs.top1conf)
                stage_a_result = self.rop_binary_model.names[stage_a_pred_idx]
            
            # Stage B: Severity classification (only if ROP detected)
            stage_b_result = None
            stage_b_confidence = 0.0
            
            if stage_a_result == "ROP":
                if isinstance(self.rop_severity_model, ort.InferenceSession):
                    # ONNX inference
                    if 'processed_img' not in locals():
                        processed_img = self.preprocess_image(image)
                    stage_b_output = self.predict_onnx(self.rop_severity_model, processed_img)
                    stage_b_pred_idx = np.argmax(stage_b_output[0])
                    stage_b_confidence = np.max(stage_b_output[0])
                    severity_classes = ["mild", "moderate", "severe"]
                    stage_b_result = severity_classes[stage_b_pred_idx]
                else:
                    # YOLO inference
                    stage_b_result_obj = self.predict_yolo(self.rop_severity_model, image)
                    stage_b_pred_idx = int(stage_b_result_obj.probs.top1)
                    stage_b_confidence = float(stage_b_result_obj.probs.top1conf)
                    stage_b_result = self.rop_severity_model.names[stage_b_pred_idx]
            
            # Determine final prediction
            if stage_a_result == "no_ROP":
                final_prediction = "No ROP"
                final_confidence = stage_a_confidence
            else:
                final_prediction = f"ROP - {stage_b_result}" if stage_b_result else "ROP"
                final_confidence = min(stage_a_confidence, stage_b_confidence) if stage_b_result else stage_a_confidence
            
            return {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'stage_a_result': stage_a_result,
                'stage_b_result': stage_b_result,
                'pipeline': 'ROP'
            }
            
        except Exception as e:
            print(f"âŒ ROP inference error: {e}")
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'stage_a_result': 'Error',
                'stage_b_result': None,
                'pipeline': 'ROP'
            }
    
    def detect_dr(self, image):
        """
        DR detection pipeline
        
        Returns:
            dict: {
                'prediction': str,
                'confidence': float,
                'stage_a_result': str,
                'stage_b_result': str
            }
        """
        try:
            # Stage A: Binary classification (No pathology vs pathology)
            if isinstance(self.dr_binary_model, ort.InferenceSession):
                # ONNX inference
                processed_img = self.preprocess_image(image)
                stage_a_output = self.predict_onnx(self.dr_binary_model, processed_img)
                stage_a_pred_idx = np.argmax(stage_a_output[0])
                stage_a_confidence = np.max(stage_a_output[0])
                stage_a_result = "pathology" if stage_a_pred_idx == 1 else "no_pathology"
            else:
                # YOLO inference
                stage_a_result_obj = self.predict_yolo(self.dr_binary_model, image)
                stage_a_pred_idx = int(stage_a_result_obj.probs.top1)
                stage_a_confidence = float(stage_a_result_obj.probs.top1conf)
                stage_a_result = self.dr_binary_model.names[stage_a_pred_idx]
            
            # Stage B: Severity classification (only if pathology detected)
            stage_b_result = None
            stage_b_confidence = 0.0
            
            if stage_a_result == "pathology":
                if isinstance(self.dr_severity_model, ort.InferenceSession):
                    # ONNX inference
                    if 'processed_img' not in locals():
                        processed_img = self.preprocess_image(image)
                    stage_b_output = self.predict_onnx(self.dr_severity_model, processed_img)
                    stage_b_pred_idx = np.argmax(stage_b_output[0])
                    stage_b_confidence = np.max(stage_b_output[0])
                    severity_classes = ["mild", "moderate", "severe"]
                    stage_b_result = severity_classes[stage_b_pred_idx]
                else:
                    # YOLO inference
                    stage_b_result_obj = self.predict_yolo(self.dr_severity_model, image)
                    stage_b_pred_idx = int(stage_b_result_obj.probs.top1)
                    stage_b_confidence = float(stage_b_result_obj.probs.top1conf)
                    stage_b_result = self.dr_severity_model.names[stage_b_pred_idx]
            
            # Determine final prediction
            if stage_a_result == "no_pathology":
                final_prediction = "No DR"
                final_confidence = stage_a_confidence
            else:
                final_prediction = f"DR - {stage_b_result}" if stage_b_result else "DR"
                final_confidence = min(stage_a_confidence, stage_b_confidence) if stage_b_result else stage_a_confidence
            
            return {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'stage_a_result': stage_a_result,
                'stage_b_result': stage_b_result,
                'pipeline': 'DR'
            }
            
        except Exception as e:
            print(f"âŒ DR inference error: {e}")
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'stage_a_result': 'Error',
                'stage_b_result': None,
                'pipeline': 'DR'
            }
    
    def detect_integrated(self, image, patient_context=None, mode='both'):
        """
        Integrated detection with multiple modes
        
        Args:
            image: Input image
            patient_context: Dict with patient info (age, type, etc.)
            mode: 'rop', 'dr', 'both', 'auto'
        
        Returns:
            dict with detection results
        """
        self.perf_monitor.start_inference()
        
        results = {}
        
        try:
            if mode == 'rop':
                results['rop'] = self.detect_rop(image)
                inference_time = self.perf_monitor.end_inference('rop')
                
            elif mode == 'dr':
                results['dr'] = self.detect_dr(image)
                inference_time = self.perf_monitor.end_inference('dr')
                
            elif mode == 'both':
                results['rop'] = self.detect_rop(image)
                results['dr'] = self.detect_dr(image)
                inference_time = self.perf_monitor.end_inference('both')
                
            elif mode == 'auto' and patient_context:
                # Auto mode based on patient context
                age = patient_context.get('age', 0)
                patient_type = patient_context.get('type', 'unknown')
                
                if age < 18 or patient_type == 'preterm':
                    results['rop'] = self.detect_rop(image)
                    results['primary'] = 'rop'
                    inference_time = self.perf_monitor.end_inference('rop')
                elif patient_context.get('diabetes', False) or age > 40:
                    results['dr'] = self.detect_dr(image)
                    results['primary'] = 'dr'
                    inference_time = self.perf_monitor.end_inference('dr')
                else:
                    # Unknown context - run both
                    results['rop'] = self.detect_rop(image)
                    results['dr'] = self.detect_dr(image)
                    results['primary'] = 'both'
                    inference_time = self.perf_monitor.end_inference('both')
            else:
                # Default to both
                results['rop'] = self.detect_rop(image)
                results['dr'] = self.detect_dr(image)
                inference_time = self.perf_monitor.end_inference('both')
            
            results['inference_time_ms'] = inference_time * 1000
            results['mode'] = mode
            
            return results
            
        except Exception as e:
            print(f"âŒ Integrated detection error: {e}")
            self.perf_monitor.end_inference('both')
            return {
                'error': str(e),
                'inference_time_ms': 0,
                'mode': mode
            }

def setup_camera(camera_id=0, width=1920, height=1080, fps=30):
    """Setup camera with optimal settings for Jetson"""
    
    camera_configs = [
        # USB camera
        camera_id,
        # CSI camera (Jetson native)
        f"nvarguscamerasrc sensor-id={camera_id} ! video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! nvvidconv ! video/x-raw, format=BGC ! appsink drop=1",
        # V4L2 camera
        f"v4l2src device=/dev/video{camera_id} ! video/x-raw, width={width}, height={height}, framerate={fps}/1 ! videoconvert ! appsink"
    ]
    
    camera = None
    for config in camera_configs:
        try:
            print(f"ðŸŽ¥ Trying camera config: {config}")
            camera = cv2.VideoCapture(config)
            
            if camera.isOpened():
                ret, frame = camera.read()
                if ret and frame is not None:
                    print(f"âœ… Camera initialized successfully")
                    print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    
                    if isinstance(config, int):
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        camera.set(cv2.CAP_PROP_FPS, fps)
                        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    return camera
                else:
                    camera.release()
                    camera = None
            else:
                camera = None
                
        except Exception as e:
            print(f"âŒ Camera config failed: {e}")
            if camera:
                camera.release()
            camera = None
    
    raise RuntimeError("âŒ Could not initialize any camera")

def draw_integrated_results(frame, results, perf_stats):
    """Draw integrated detection results on frame"""
    
    # Colors for different predictions
    colors = {
        'No ROP': (0, 255, 0),
        'No DR': (0, 255, 0),
        'ROP - mild': (255, 255, 0),
        'ROP - moderate': (255, 165, 0),
        'ROP - severe': (255, 0, 0),
        'DR - mild': (255, 255, 0),
        'DR - moderate': (255, 165, 0),
        'DR - severe': (255, 0, 0),
        'Error': (128, 128, 128)
    }
    
    y_offset = 30
    
    # Draw ROP results if available
    if 'rop' in results:
        rop_pred = results['rop']['prediction']
        rop_conf = results['rop']['confidence']
        color = colors.get(rop_pred, (255, 255, 255))
        
        cv2.putText(frame, f"ROP: {rop_pred}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Conf: {rop_conf:.2f}", 
                    (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 60
    
    # Draw DR results if available
    if 'dr' in results:
        dr_pred = results['dr']['prediction']
        dr_conf = results['dr']['confidence']
        color = colors.get(dr_pred, (255, 255, 255))
        
        cv2.putText(frame, f"DR: {dr_pred}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Conf: {dr_conf:.2f}", 
                    (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 60
    
    # Draw mode
    mode = results.get('mode', 'unknown')
    cv2.putText(frame, f"Mode: {mode}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw performance info
    mode_for_stats = 'both' if 'rop' in results and 'dr' in results else ('rop' if 'rop' in results else 'dr')
    fps_text = f"FPS: {perf_stats.get('fps', 0):.1f}"
    cv2.putText(frame, fps_text, 
                (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    latency_text = f"Latency: {perf_stats.get('avg_latency_ms', 0):.1f}ms"
    cv2.putText(frame, latency_text, 
                (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Draw timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, 
                (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description='Complete ROP + DR Detection on Jetson')
    parser.add_argument('--rop-binary', required=True, help='Path to ROP binary model')
    parser.add_argument('--rop-severity', required=True, help='Path to ROP severity model')
    parser.add_argument('--dr-binary', required=True, help='Path to DR binary model')
    parser.add_argument('--dr-severity', required=True, help='Path to DR severity model')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--img-size', type=int, default=224, help='Model input size')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu', help='Inference device')
    parser.add_argument('--mode', choices=['rop', 'dr', 'both', 'auto'], default='both', 
                       help='Detection mode')
    parser.add_argument('--save-video', help='Save output video to file')
    parser.add_argument('--no-display', action='store_true', help='Run without display')
    parser.add_argument('--patient-age', type=int, help='Patient age for auto mode')
    parser.add_argument('--patient-type', choices=['preterm', 'adult', 'unknown'], 
                       default='unknown', help='Patient type for auto mode')
    
    args = parser.parse_args()
    
    print("ðŸš€ Starting Complete ROP + DR Detection System")
    print("=" * 60)
    
    try:
        # Initialize integrated detector
        print("ðŸ“¦ Loading integrated detection system...")
        detector = IntegratedROPDRDetector(
            args.rop_binary, args.rop_severity,
            args.dr_binary, args.dr_severity,
            args.device, args.img_size
        )
        
        # Setup camera
        print("ðŸŽ¥ Setting up camera...")
        camera = setup_camera(args.camera)
        
        # Setup video writer if requested
        video_writer = None
        if args.save_video:
            ret, frame = camera.read()
            if ret:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (w, h))
                print(f"ðŸ“¹ Saving video to: {args.save_video}")
        
        # Patient context for auto mode
        patient_context = {
            'age': args.patient_age,
            'type': args.patient_type,
            'diabetes': args.patient_type == 'adult'
        }
        
        print(f"â–¶ï¸  Starting detection in '{args.mode}' mode...")
        print("Press 'q' to quit, 's' to save screenshot, 'm' to cycle modes")
        
        frame_count = 0
        current_mode = args.mode
        modes = ['rop', 'dr', 'both', 'auto']
        
        while True:
            ret, frame = camera.read()
            if not ret:
                print("âŒ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Run integrated detection
            results = detector.detect_integrated(frame, patient_context, current_mode)
            
            # Get performance stats
            mode_for_stats = 'both' if 'rop' in results and 'dr' in results else ('rop' if 'rop' in results else 'dr')
            perf_stats = detector.perf_monitor.get_stats(mode_for_stats)
            
            # Draw results on frame
            output_frame = draw_integrated_results(frame.copy(), results, perf_stats)
            
            # Save frame to video if requested
            if video_writer:
                video_writer.write(output_frame)
            
            # Display frame (if not headless)
            if not args.no_display:
                cv2.imshow('Complete ROP + DR Detection', output_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_name = f"rop_dr_detection_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_name, output_frame)
                    print(f"ðŸ“¸ Screenshot saved: {screenshot_name}")
                elif key == ord('m'):
                    # Cycle through modes
                    current_idx = modes.index(current_mode)
                    current_mode = modes[(current_idx + 1) % len(modes)]
                    print(f"ðŸ”„ Switched to mode: {current_mode}")
            
            # Print periodic stats
            if frame_count % 100 == 0:
                print(f"ðŸ“Š Frame {frame_count}:")
                print(f"   Mode: {current_mode}")
                print(f"   Performance: {perf_stats}")
                if 'rop' in results:
                    print(f"   ROP: {results['rop']['prediction']} ({results['rop']['confidence']:.2f})")
                if 'dr' in results:
                    print(f"   DR: {results['dr']['prediction']} ({results['dr']['confidence']:.2f})")
    
    except KeyboardInterrupt:
        print("\nâ¹ï¸  Interrupted by user")
    except Exception as e:
        print(f"âŒ Application error: {e}")
    finally:
        # Cleanup
        print("ðŸ§¹ Cleaning up...")
        if 'camera' in locals() and camera:
            camera.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        if 'detector' in locals():
            for mode in ['rop', 'dr', 'both']:
                final_stats = detector.perf_monitor.get_stats(mode)
                if final_stats['frame_count'] > 0:
                    print(f"\nðŸ“Š Final {mode.upper()} Performance:")
                    print(f"   Frames processed: {final_stats['frame_count']}")
                    print(f"   Average FPS: {final_stats['fps']:.1f}")
                    print(f"   Average latency: {final_stats['avg_latency_ms']:.1f}ms")
                    print(f"   P95 latency: {final_stats['p95_latency_ms']:.1f}ms")

if __name__ == "__main__":
    main()
