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

from sagemaker_training.wildlife_dataloader_sm import (
    load_bbox_data_sm,
    load_bbox_from_split,
    crop_to_bbox,
)

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
        default=["bobcat", "coyote", "deer", "empty", "fox", "mountain_lion"],
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
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Print a confusion matrix after evaluation.",
    )
    parser.add_argument(
        "--confusion-matrix-json",
        default=None,
        help="Optional path to write confusion matrix JSON output.",
    )
    parser.add_argument(
        "--prediction-summary",
        action="store_true",
        help="Print predicted class distribution and top predictions per true class.",
    )
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
    num_classes = len(args.classes)
    per_class_total = np.zeros(num_classes, dtype=int)
    per_class_correct = np.zeros(num_classes, dtype=int)
    pred_counts = np.zeros(num_classes, dtype=int)
    confusion_matrix = None
    if args.confusion_matrix or args.confusion_matrix_json or args.prediction_summary:
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
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
            per_class_total[true_label] += 1
            if pred == true_label:
                correct += 1
                per_class_correct[true_label] += 1
            pred_counts[pred] += 1
            if confusion_matrix is not None:
                confusion_matrix[true_label, pred] += 1

    accuracy = correct / total if total else 0.0
    print(f"Samples evaluated: {total}")
    print(f"Top-1 accuracy: {accuracy:.4f}")
    print("\nPer-class accuracy:")
    for idx, label in enumerate(args.classes):
        class_total = per_class_total[idx]
        class_correct = per_class_correct[idx]
        class_acc = class_correct / class_total if class_total else 0.0
        print(f"  {label}: {class_acc:.4f} ({class_correct}/{class_total})")
    if args.prediction_summary:
        print("\nPredicted class distribution:")
        for idx, label in enumerate(args.classes):
            count = pred_counts[idx]
            percent = count / total if total else 0.0
            print(f"  {label}: {percent:.4f} ({count}/{total})")
        if confusion_matrix is not None:
            print("\nTop predicted class per true label:")
            for idx, label in enumerate(args.classes):
                row = confusion_matrix[idx]
                top_idx = int(np.argmax(row)) if row.sum() else None
                if top_idx is None:
                    print(f"  {label}: no predictions")
                    continue
                top_label = args.classes[top_idx]
                top_count = int(row[top_idx])
                top_percent = top_count / row.sum() if row.sum() else 0.0
                print(
                    f"  {label}: {top_label} {top_percent:.4f} ({top_count}/{row.sum()})"
                )
    if confusion_matrix is not None:
        print("\nConfusion matrix (rows=true, cols=pred):")
        header = " " * 14 + " ".join(f"{label[:6]:>6}" for label in args.classes)
        print(header)
        for idx, label in enumerate(args.classes):
            row = " ".join(f"{count:6d}" for count in confusion_matrix[idx])
            print(f"{label[:12]:>12} {row}")
    if args.confusion_matrix_json and confusion_matrix is not None:
        output = {
            "labels": args.classes,
            "matrix": confusion_matrix.tolist(),
        }
        with open(args.confusion_matrix_json, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()