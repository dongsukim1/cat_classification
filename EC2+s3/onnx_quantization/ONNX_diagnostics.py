# evaluate_onnx.py
import torch
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import os
import sys

# Add your model path
sys.path.append('./sagemaker_training')
from sagemaker_training.model_efficient_net import WildLifeEfficientNet

def load_torch_model(pth_path, num_classes=6):
    model = WildLifeEfficientNet(num_classes=num_classes)
    state_dict = torch.load(pth_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # Add batch dim

def main():
    pth_path = "./EC2+s3/training/run4_expanded_empty_ENB3/model.pth"         
    onnx_path = "model.onnx"        # Your exported ONNX
    test_image = "./data/s3+expanded_empty/bobcat/5a0b00c7-23d2-11e8-a6a3-ec086b02610b.jpg"        

    img = Image.open(test_image)
    print(img.size)  # Was it resized to 224 or 300?

    # Load models
    torch_model = load_torch_model(pth_path)
    ort_session = ort.InferenceSession(onnx_path)

    # Preprocess
    input_tensor = preprocess_image(test_image)
    input_np = input_tensor.numpy()

    # Run inference
    with torch.no_grad():
        torch_out = torch_model(input_tensor)
    onnx_out = ort_session.run(None, {"input": input_np})[0]

    # Compare
    diff = np.max(np.abs(torch_out.numpy() - onnx_out))
    print(f"Max output difference: {diff:.2e}")
    print("✅ ONNX matches PyTorch!" if diff < 1e-4 else "❌ Mismatch detected!")

    # Predictions
    torch_pred = torch.argmax(torch_out, dim=1).item()
    onnx_pred = np.argmax(onnx_out, axis=1)[0]
    print(f"PyTorch prediction: {torch_pred}, ONNX prediction: {onnx_pred}")

if __name__ == "__main__":
    main()