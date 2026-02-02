# calibrate.py
import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType
)

class CalibrationReader(CalibrationDataReader):
    def __init__(self, base_dir, class_names, max_images_per_class=100, batch_size=1):
        """
        Args:
            base_dir (str): e.g. "./data/s3+expanded_empty"
            class_names (list): e.g. ["mountain_lion", "bobcat", ...]
            max_images_per_class (int): try to get this many per class (use fewer if not available)
            batch_size (int): usually 1 for calibration
        """
        self.batch_size = batch_size
        self.image_paths = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for class_name in class_names:
            folder = os.path.join(base_dir, class_name)
            if not os.path.isdir(folder):
                print(f"⚠️ Warning: Class folder not found: {folder}")
                continue

            all_images = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            if not all_images:
                print(f"⚠️ Warning: No images in {folder}")
                continue

            # Take min(max_images_per_class, available)
            num_to_take = min(max_images_per_class, len(all_images))
            sampled = random.sample(all_images, num_to_take)
            self.image_paths.extend(sampled)
            print(f"✅ Loaded {num_to_take} images from {class_name}")

        random.shuffle(self.image_paths)
        self.idx = 0
        print(f"📊 Total calibration images: {len(self.image_paths)}")

    def get_next(self):
        if self.idx >= len(self.image_paths):
            return None
        batch = []
        for _ in range(self.batch_size):
            if self.idx < len(self.image_paths):
                path = self.image_paths[self.idx]
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = self.transform(img)
                    batch.append(tensor.numpy())
                    self.idx += 1
                except Exception as e:
                    print(f"Skipping corrupted image {path}: {e}")
                    continue
        if not batch:
            return None
        return {"input": np.stack(batch, axis=0)}

def main():
    # 🔧 Configuration
    base_data_dir = "./data/s3+expanded_empty"  # ← matches your format
    class_names = [
        "mountain_lion",
        "bobcat",
        "coyote",
        "fox",
        "deer",
        "empty"
    ]
    max_per_class = 100  # Will use fewer if not available

    input_onnx = "model.onnx"
    output_onnx = "model_quant.onnx"

    print("🔍 Building calibration dataset from class folders...")
    calib_reader = CalibrationReader(
        base_dir=base_data_dir,
        class_names=class_names,
        max_images_per_class=max_per_class,
        batch_size=1
    )

    if len(calib_reader.image_paths) == 0:
        raise ValueError("No valid images found for calibration!")

    print("\n Running static quantization...")
    quantize_static(
        model_input=input_onnx,
        model_output=output_onnx,
        calibration_data_reader=calib_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=['Conv', 'MatMul'],
        use_external_data_format=False,
        nodes_to_exclude=[
        "classifier.1",          # final Linear layer
        "features.6.0.block.2.fc2"  # the layer with bias warning
        ]
    )
    print(f"\n Quantized model saved to: {output_onnx}")

if __name__ == "__main__":
    main()