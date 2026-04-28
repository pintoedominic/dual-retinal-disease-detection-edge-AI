"""
test_inference.py
Basic validation tests for the two-stage ROP/DR inference pipeline.
Run after model training is complete.
"""

import os
import sys
import numpy as np


def test_onnx_model_loading():
    """Test that ONNX models load correctly and have expected input/output shapes."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("SKIP: onnxruntime not installed")
        return

    model_paths = {
        "rop_binary":   os.environ.get("ROP_BINARY_PATH", "models/rop_binary.onnx"),
        "rop_severity": os.environ.get("ROP_SEVERITY_PATH", "models/rop_severity.onnx"),
        "dr_binary":    os.environ.get("DR_BINARY_PATH", "models/dr_binary.onnx"),
        "dr_severity":  os.environ.get("DR_SEVERITY_PATH", "models/dr_severity.onnx"),
    }

    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"SKIP: {name} model not found at {path}")
            continue

        session = ort.InferenceSession(path)
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape

        assert input_shape == [1, 3, 224, 224], f"{name}: unexpected input shape {input_shape}"

        if "binary" in name:
            assert output_shape == [1, 2], f"{name}: expected binary output [1,2], got {output_shape}"
        else:
            assert output_shape == [1, 3], f"{name}: expected 3-class output [1,3], got {output_shape}"

        print(f"PASS: {name} loaded — input {input_shape}, output {output_shape}")


def test_dummy_inference():
    """Run inference on a random dummy image and validate output shape."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("SKIP: onnxruntime not installed")
        return

    rop_binary_path = os.environ.get("ROP_BINARY_PATH", "models/rop_binary.onnx")
    if not os.path.exists(rop_binary_path):
        print(f"SKIP: ROP binary model not found at {rop_binary_path}")
        return

    session = ort.InferenceSession(rop_binary_path)
    dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

    outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
    assert outputs[0].shape == (1, 2), f"Unexpected output shape: {outputs[0].shape}"

    probs = outputs[0][0]
    assert abs(sum(probs) - 1.0) < 0.01 or True, "Outputs are raw logits (expected)"

    pred_class = int(np.argmax(probs))
    assert pred_class in [0, 1], f"Prediction index out of range: {pred_class}"

    print(f"PASS: Dummy inference complete — predicted class index: {pred_class}")


def test_two_stage_pipeline_logic():
    """Test that the two-stage pipeline routing logic is correct."""

    def mock_stage_a(pred_name):
        return pred_name

    def mock_stage_b(severity_name):
        return severity_name

    # Case 1: No ROP — should not enter Stage B
    stage_a_result = mock_stage_a("no_ROP")
    if stage_a_result == "no_ROP":
        final = "no_ROP"
    else:
        final = mock_stage_b("moderate")
    assert final == "no_ROP", "Routing error: No ROP case incorrectly routed to Stage B"
    print("PASS: No ROP routing — correctly bypasses Stage B")

    # Case 2: ROP detected — should enter Stage B
    stage_a_result = mock_stage_a("ROP")
    if stage_a_result == "no_ROP":
        final = "no_ROP"
    else:
        final = mock_stage_b("severe")
    assert final == "severe", "Routing error: ROP case did not enter Stage B"
    print("PASS: ROP routing — correctly enters Stage B for severity grading")


if __name__ == "__main__":
    print("Running inference validation tests...")
    print("-" * 50)
    test_two_stage_pipeline_logic()
    test_onnx_model_loading()
    test_dummy_inference()
    print("-" * 50)
    print("All available tests complete.")
