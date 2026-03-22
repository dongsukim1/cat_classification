# sagemaker_training/qat_train.py
"""Phase 2: Quantization-Aware Training."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.ao.quantization as tq
from torch.optim import Adam
from torch.utils.data import DataLoader

from models import create_student
from wildlife_dataloader_sm import WildlifeDataset


CLASS_NAMES = sorted(["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"])


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("qat-checkpoint-*.pth"))
    return checkpoints[-1] if checkpoints else None


def fuse_model_modules(student, arch):
    """Fuse Conv+BN+Activation modules before QAT. Required for correct quantization."""
    if arch == "mobilenetv3_small":
        # torchvision MobileNetV3 has well-defined fuseable modules
        from torch.ao.quantization import fuse_modules_qat
        # Fuse patterns vary per layer; use model's built-in fuse if available
        if hasattr(student, 'fuse_model'):
            student.fuse_model()
    # MobileNetV4 from timm: manual fusion may be needed.
    # If fusion is too complex, skip and rely on PTQ fallback per spec.
    return student


def prepare_qat_model(student, arch):
    """Insert fake quantization nodes for QAT."""
    student = fuse_model_modules(student, arch)
    student.train()
    student.qconfig = tq.get_default_qat_qconfig("x86")
    tq.prepare_qat(student, inplace=True)
    return student


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-arch", default="mobilenetv3_small",
                        choices=["mobilenetv3_small", "mobilenetv4_conv_s"])
    parser.add_argument("--phase1-model", required=True,
                        help="Path to best Phase 1 model.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size-train", type=int, default=32)
    parser.add_argument("--batch-size-val", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "./output"))
    parser.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_TRAIN", "./data/s3+expanded_empty"))
    parser.add_argument("--splits-dir", default=os.environ.get("SM_CHANNEL_SPLITS", "./EC2+s3/data_augmentation_pipeline/splitsv2"))
    parser.add_argument("--checkpoint-dir", default="/opt/ml/checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"  # QAT works on GPU
    label_to_idx = {label: idx for idx, label in enumerate(CLASS_NAMES)}
    num_classes = len(CLASS_NAMES)

    # Load Phase 1 best model
    student = create_student(args.student_arch, num_classes=num_classes)
    state_dict = torch.load(args.phase1_model, map_location=device)
    student.load_state_dict(state_dict)
    print(f"Loaded Phase 1 model from {args.phase1_model}")

    # Prepare for QAT
    student = prepare_qat_model(student, args.student_arch)
    student = student.to(device)

    # Data — standard single-input (no teacher needed)
    # Reuse existing split loading with unified paths
    train_samples = json.load(open(Path(args.splits_dir) / "train.json"))
    val_samples = json.load(open(Path(args.splits_dir) / "val.json"))

    def samples_to_lists(samples):
        paths, labels = [], []
        for s in samples:
            cls = s["primary_class"]
            if cls not in CLASS_NAMES:
                continue
            image_path = s.get("image_path") or s.get("image_path_aws", "")
            paths.append(str(Path(args.data_dir) / image_path))
            labels.append(cls)
        return paths, labels

    train_paths, train_labels = samples_to_lists(train_samples)
    val_paths, val_labels = samples_to_lists(val_samples)

    train_dataset = WildlifeDataset(train_paths, train_labels, mode="train", label_to_idx=label_to_idx)
    val_dataset = WildlifeDataset(val_paths, val_labels, mode="val", label_to_idx=label_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val,
                            shuffle=False, num_workers=args.num_workers)

    # Class weights
    counts = torch.zeros(num_classes)
    for _, label in train_dataset:
        counts[label] += 1
    class_weights = (1.0 / counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * num_classes

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(student.parameters(), lr=args.learning_rate)

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        student.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = state["epoch"] + 1
        best_val_acc = state.get("best_val_acc", 0.0)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        student.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = student(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        # Validate
        student.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = student(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = correct / total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{args.epochs} | Train acc={train_acc:.4f} | Val acc={val_acc:.4f}")

        # Checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        }, Path(args.checkpoint_dir) / f"qat-checkpoint-{epoch:03d}.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.model_dir, exist_ok=True)
            # Save the QAT model (with fake quant nodes) for ONNX export
            torch.save(student.state_dict(), Path(args.model_dir) / "model_qat.pth")
            print(f"  New best: {best_val_acc:.4f}")

    print(f"QAT complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
