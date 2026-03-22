# export_onnx.py
"""Phase 3: Export QAT model to ONNX and validate."""
import argparse
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.ao.quantization as tq

sys.path.append("./sagemaker_training")
from models import create_student


CLASS_NAMES = sorted(["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"])


def export_qat_to_onnx(student, output_path, opset=13):
    """Export QAT model with fake-quant nodes to ONNX."""
    student.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        student,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output_path}")


def validate_onnx(onnx_path, student, num_samples=10):
    """Compare ONNX output against PyTorch model on random inputs."""
    student.eval()
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    max_diff = 0.0
    mismatches = 0
    for _ in range(num_samples):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            pt_out = student(x).numpy()
        ort_out = session.run(None, {input_name: x.numpy()})[0]

        diff = np.max(np.abs(pt_out - ort_out))
        max_diff = max(max_diff, diff)
        if np.argmax(pt_out) != np.argmax(ort_out):
            mismatches += 1

    print(f"Max output difference: {max_diff:.2e}")
    print(f"Prediction mismatches: {mismatches}/{num_samples}")
    return max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-arch", required=True,
                        choices=["mobilenetv3_small", "mobilenetv4_conv_s"])
    parser.add_argument("--qat-model", required=True, help="Path to model_qat.pth")
    parser.add_argument("--output", default="model_distilled.onnx")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    student = create_student(args.student_arch, num_classes=len(CLASS_NAMES))

    # Prepare QAT structure then load weights
    student.train()
    student.qconfig = tq.get_default_qat_qconfig("x86")
    tq.prepare_qat(student, inplace=True)
    state_dict = torch.load(args.qat_model, map_location="cpu")
    student.load_state_dict(state_dict)

    # Primary path: export QAT model with fake-quant nodes
    import os
    try:
        export_qat_to_onnx(student, args.output, args.opset)
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        validate_onnx(args.output, student)
    except Exception as e:
        print(f"QAT ONNX export failed: {e}")
        print("Falling back to FP32 export + ONNX Runtime quantize_static...")
        # Fallback: convert QAT to quantized, export as FP32, then quantize via ORT
        student_fp32 = tq.convert(student, inplace=False)
        fp32_path = args.output.replace(".onnx", "_fp32.onnx")
        export_qat_to_onnx(student_fp32, fp32_path, args.opset)

        from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod
        from onnxruntime.quantization import quant_pre_process
        preproc_path = args.output.replace(".onnx", "_preproc.onnx")
        quant_pre_process(fp32_path, preproc_path)
        # NOTE: Provide a CalibrationDataReader here for real calibration
        print(f"Pre-processed model saved. Run calibrate_onnx.py on {preproc_path} to complete.")

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    if size_mb > 5.0:
        print(f"WARNING: Model size {size_mb:.2f} MB exceeds 5MB target")
    else:
        print(f"Model size {size_mb:.2f} MB is within target")


if __name__ == "__main__":
    main()
