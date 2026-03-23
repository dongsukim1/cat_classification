# sagemaker_training/distill_train.py
"""Phase 1: Bbox-conditioned feature distillation training."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models import create_student
from teacher import load_teacher
from distillation_loss import ProjectionHead, combined_loss, compute_alpha
from wildlife_dataloader_sm import DistillationDataset


CLASS_NAMES = sorted(["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"])


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*.pth"))
    return checkpoints[-1] if checkpoints else None


def compute_class_weights(split_file, label_to_idx, num_classes, device):
    """Compute inverse-frequency class weights from split JSON metadata."""
    with open(split_file) as f:
        samples = json.load(f)
    counts = torch.zeros(num_classes)
    for sample in samples:
        cls = sample.get("primary_class", "")
        if cls in label_to_idx:
            counts[label_to_idx[cls]] += 1
    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


def train_one_epoch(student, teacher, projection_head, dataloader, optimizer, alpha, class_weights, device):
    student.train()
    projection_head.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for student_imgs, teacher_imgs, labels, apply_distill in dataloader:
        student_imgs = student_imgs.to(device)
        teacher_imgs = teacher_imgs.to(device)
        labels = labels.to(device)
        apply_distill = apply_distill.to(device)

        optimizer.zero_grad()

        student_features = student.extract_features(student_imgs)
        logits = student(student_imgs)

        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    teacher_features = teacher(teacher_imgs).float()
            else:
                teacher_features = teacher(teacher_imgs)

        loss = combined_loss(
            logits=logits, labels=labels,
            student_features=student_features, teacher_features=teacher_features,
            projection_head=projection_head, apply_distill=apply_distill,
            alpha=alpha, class_weights=class_weights,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(student, dataloader, device):
    student.eval()
    correct = 0
    total = 0
    for student_imgs, _, labels, _ in dataloader:
        student_imgs = student_imgs.to(device)
        labels = labels.to(device)
        logits = student(student_imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-arch", default="mobilenetv3_small",
                        choices=["mobilenetv3_small", "mobilenetv4_conv_s", "efficientnet_lite0"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size-train", type=int, default=32)
    parser.add_argument("--batch-size-val", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--teacher-weights", default=None)
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "./output"))
    parser.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_TRAIN", "./data/s3+expanded_empty"))
    parser.add_argument("--splits-dir", default=os.environ.get("SM_CHANNEL_SPLITS", "./EC2+s3/data_augmentation_pipeline/splitsv2"))
    parser.add_argument("--checkpoint-dir", default="/opt/ml/checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_to_idx = {label: idx for idx, label in enumerate(CLASS_NAMES)}
    num_classes = len(CLASS_NAMES)

    student = create_student(args.student_arch, num_classes=num_classes).to(device)
    teacher = load_teacher(device=device, weights_path=args.teacher_weights)
    projection_head = ProjectionHead(student.feature_dim, 1024).to(device)

    train_split = str(Path(args.splits_dir) / "train.json")
    val_split = str(Path(args.splits_dir) / "val.json")

    train_dataset = DistillationDataset(args.data_dir, train_split, CLASS_NAMES, label_to_idx, mode="train")
    val_dataset = DistillationDataset(args.data_dir, val_split, CLASS_NAMES, label_to_idx, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    class_weights = compute_class_weights(train_split, label_to_idx, num_classes, device)
    optimizer = Adam(list(student.parameters()) + list(projection_head.parameters()), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_acc = 0.0
    checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    if checkpoint:
        print(f"Resuming from {checkpoint}")
        state = torch.load(checkpoint, map_location=device)
        student.load_state_dict(state["student_state_dict"])
        projection_head.load_state_dict(state["projection_head_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_val_acc = state.get("best_val_acc", 0.0)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(start_epoch, args.epochs):
        alpha = compute_alpha(epoch, args.epochs)
        print(f"\nEpoch {epoch+1}/{args.epochs} | alpha={alpha:.3f} | lr={scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_acc = train_one_epoch(student, teacher, projection_head, train_loader, optimizer, alpha, class_weights, device)
        val_acc = validate(student, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train loss={train_loss:.4f} acc={train_acc:.4f} | Val acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  New best val acc: {best_val_acc:.4f}")
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(student.state_dict(), Path(args.model_dir) / "model.pth")

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "student_state_dict": student.state_dict(),
            "projection_head_state_dict": projection_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "student_arch": args.student_arch,
        }, Path(args.checkpoint_dir) / f"checkpoint-{epoch:03d}.pth")

    with open(Path(args.model_dir) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
