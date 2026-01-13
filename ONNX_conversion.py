import torch
import torch.onnx
from torchvision import models
import torch.nn as nn
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader
import numpy as np
import time
import sys


dataloader_path = './sagemaker_training'
sys.path.append(dataloader_path)

from model_efficient_net import WildLifeEfficientNet as ENB3

def convert_to_onnx(model_path, onnx_path, input_size=(4, 3, 224, 224)):
    """Convert PyTorch .pth to ONNX"""
    print("Loading PyTorch model...")
    model = ENB3(num_classes=6)
    
    # Load weights (handles different checkpoint formats)
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    
    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,  # Use latest for best performance
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        do_constant_folding=True,  # Optimize constants
        verbose=False
    )
    print(f"✅ ONNX model saved to: {onnx_path}")