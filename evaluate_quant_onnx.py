#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from sagemaker_training.wildlife_dataloader_sm import load_bbox_data_sm


def load_bbox_from_split(splits_dir, split_name):
    split_path = Path(splits_dir) / f"{split_name}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with split_path.open() as handle:
        samples = json.load(handle)
    bbox_dict = {}
    for sample in samples:
        image_id = sample.get("image_id")
        annotations = sample.get("annotations")
        if not image_id and sample.get("image_path_local"):
            image_id = Path(sample["image_path_local"]).stem
        if image_id and annotations:
            bbox_dict[image_id] = annotations
    return bbox_dict


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

def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def list_images(base_dir, class_names, max_per_class=None, seed=0, shuffle=False):
    samples = []
    for class_name in class_names:
        class_dir = Path(base_dir) / class_name
        if not class_dir.is_dir():
            print(f"⚠️ Skipping missing class folder: {class_dir}")
            continue
        images = sorted(
            p
            for p in class_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(images)
        if max_per_class is not None:
            images = images[: max_per_class]
        samples.extend((p, class_name) for p in images)
        print(f"✅ Found {len(images)} images for {class_name}")
    return samples


def load_image(path, transform, bbox_dict=None, use_bboxes=False):
    image = Image.open(path).convert("RGB")
    if use_bboxes and bbox_dict is not None:
        image_id = Path(path).stem
        image = crop_to_bbox(image, bbox_dict.get(image_id))
    tensor = transform(image)
    return tensor.numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate quantized ONNX model accuracy on a folder-structured dataset."
    )
    parser.add_argument("--model", default="model_quant.onnx", help="Path to ONNX model.")
    parser.add_argument(
        "--data-dir",
        default=os.path.join("data", "s3+expanded_empty"),
        help="Dataset root with class subfolders (e.g. /data/val).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"],
        help="Class folder names in label order.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="Optional cap of samples per class.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--labels-file", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--use-bboxes", action="store_true")
    args = parser.parse_args()

    transform = build_transform()
    label_to_idx = {label: idx for idx, label in enumerate(args.classes)}
    bbox_dict = {}
    if args.use_bboxes:
        if args.labels_file:
            bbox_dict = load_bbox_data_sm(args.labels_file)
        elif args.splits_dir:
            bbox_dict = load_bbox_from_split(args.splits_dir, args.split_name)
        else:
            raise SystemExit("--labels-file or --splits-dir is required when --use-bboxes is set.")
    samples = list_images(
        args.data_dir,
        args.classes,
        max_per_class=args.max_per_class,
        seed=args.seed,
        shuffle=args.shuffle,
    )

    if not samples:
        raise SystemExit("No samples found. Check --data-dir and --classes.")

    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name

    correct = 0
    total = 0
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        inputs = []
        labels = []
        for path, label in batch:
            try:
                inputs.append(
                    load_image(
                        path,
                        transform,
                        bbox_dict=bbox_dict,
                        use_bboxes=args.use_bboxes,
                    )
                )
                labels.append(label_to_idx[label])
            except Exception as exc:
                print(f"⚠️ Skipping {path}: {exc}")
        if not inputs:
            continue
        input_tensor = np.stack(inputs, axis=0)
        outputs = session.run(None, {input_name: input_tensor})[0]
        preds = np.argmax(outputs, axis=1)
        for pred, true_label in zip(preds, labels):
            total += 1
            if pred == true_label:
                correct += 1

    accuracy = correct / total if total else 0.0
    print(f"Samples evaluated: {total}")
    print(f"Top-1 accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

