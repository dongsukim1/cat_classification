#!/usr/bin/env python3
import argparse
import os
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


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


def load_image(path, transform):
    image = Image.open(path).convert("RGB")
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
    args = parser.parse_args()

    transform = build_transform()
    label_to_idx = {label: idx for idx, label in enumerate(args.classes)}
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
                inputs.append(load_image(path, transform))
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
