# calibrate.py
import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import argparse
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType
)

from sagemaker_training.wildlife_dataloader_sm import load_bbox_data_sm


def crop_to_bbox(image, bbox_info):
    if not bbox_info:
        return image
    bbox = bbox_info[0].get("bbox", None)
    if bbox is None:
        return image
    x, y, width, height = bbox
    img_width, img_height = image.size
    x = max(0, min(x, img_width))
    y = max(0, min(y, img_height))
    width = max(1, min(width, img_width - x))
    height = max(1, min(height, img_height - y))
    left, top, right, bottom = int(x), int(y), int(x + width), int(y + height)
    try:
        return image.crop((left, top, right, bottom))
    except Exception as exc:
        print(f"Warning: Failed to crop image with bbox {bbox}: {exc}")
        return image

class CalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        base_dir,
        class_names,
        max_images_per_class=100,
        batch_size=1,
        bbox_dict=None,
        use_bboxes=False,
    ):
        """
        Args:
            base_dir (str): e.g. "./data/s3+expanded_empty"
            class_names (list): e.g. ["mountain_lion", "bobcat", ...]
            max_images_per_class (int): try to get this many per class (use fewer if not available)
            batch_size (int): usually 1 for calibration
        """
        self.batch_size = batch_size
        self.image_paths = []
        self.bbox_dict = bbox_dict or {}
        self.use_bboxes = use_bboxes
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
                    if self.use_bboxes:
                        image_id = Path(path).stem
                        img = crop_to_bbox(img, self.bbox_dict.get(image_id))
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
    parser = argparse.ArgumentParser(description="Calibrate and quantize an ONNX model.")
    parser.add_argument("--data-dir", default="./data/s3+expanded_empty")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"],
    )
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--input-onnx", default="model.onnx")
    parser.add_argument("--output-onnx", default="model_quant.onnx")
    parser.add_argument("--labels-file", default=None)
    parser.add_argument("--use-bboxes", action="store_true")
    args = parser.parse_args()

    bbox_dict = {}
    if args.use_bboxes:
        if not args.labels_file:
            raise SystemExit("--labels-file is required when --use-bboxes is set.")
        bbox_dict = load_bbox_data_sm(args.labels_file)

    print("🔍 Building calibration dataset from class folders...")
    calib_reader = CalibrationReader(
        base_dir=args.data_dir,
        class_names=args.classes,
        max_images_per_class=args.max_per_class,
        batch_size=1,
        bbox_dict=bbox_dict,
        use_bboxes=args.use_bboxes,
    )

    if len(calib_reader.image_paths) == 0:
        raise ValueError("No valid images found for calibration!")

    print("\n Running static quantization...")
    quantize_static(
        model_input=args.input_onnx,
        model_output=args.output_onnx,
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
    print(f"\n Quantized model saved to: {args.output_onnx}")

if __name__ == "__main__":
    main()