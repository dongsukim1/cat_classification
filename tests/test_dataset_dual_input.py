import json
from pathlib import Path

import torch
import pytest


def _make_test_data(tmp_path):
    """Create minimal test data: 1 image per class, split JSON."""
    from PIL import Image

    classes = ["bobcat", "deer", "empty"]
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        img = Image.new("RGB", (256, 256), color="red")
        img.save(cls_dir / "img001.jpg")

    samples = [
        {
            "image_id": "img001",
            "primary_class": "bobcat",
            "image_path": "bobcat/img001.jpg",
            "annotations": [{"bbox": [10, 10, 100, 100]}],
        },
        {
            "image_id": "img001",
            "primary_class": "deer",
            "image_path": "deer/img001.jpg",
            "annotations": [{"bbox": [20, 20, 80, 80]}],
        },
        {
            "image_id": "img001",
            "primary_class": "empty",
            "image_path": "empty/img001.jpg",
            "annotations": [],
        },
    ]
    split_file = tmp_path / "train.json"
    with split_file.open("w") as f:
        json.dump(samples, f)
    return split_file


def test_distillation_dataset_returns_four_items(tmp_path):
    from sagemaker_training.wildlife_dataloader_sm import DistillationDataset

    split_file = _make_test_data(tmp_path)
    target_species = ["bobcat", "deer", "empty"]
    label_to_idx = {label: idx for idx, label in enumerate(sorted(target_species))}

    ds = DistillationDataset(
        data_dir=str(tmp_path),
        split_file=str(split_file),
        target_species=target_species,
        label_to_idx=label_to_idx,
        mode="train",
    )
    student_tensor, teacher_tensor, label, apply_distill = ds[0]
    assert student_tensor.shape == (3, 224, 224)
    assert teacher_tensor.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert isinstance(apply_distill, bool)


def test_empty_class_has_apply_distill_false(tmp_path):
    from sagemaker_training.wildlife_dataloader_sm import DistillationDataset

    split_file = _make_test_data(tmp_path)
    target_species = ["bobcat", "deer", "empty"]
    label_to_idx = {label: idx for idx, label in enumerate(sorted(target_species))}

    ds = DistillationDataset(
        data_dir=str(tmp_path),
        split_file=str(split_file),
        target_species=target_species,
        label_to_idx=label_to_idx,
        mode="train",
    )
    _, _, _, apply_distill = ds[2]
    assert apply_distill is False


def test_animal_class_has_apply_distill_true(tmp_path):
    from sagemaker_training.wildlife_dataloader_sm import DistillationDataset

    split_file = _make_test_data(tmp_path)
    target_species = ["bobcat", "deer", "empty"]
    label_to_idx = {label: idx for idx, label in enumerate(sorted(target_species))}

    ds = DistillationDataset(
        data_dir=str(tmp_path),
        split_file=str(split_file),
        target_species=target_species,
        label_to_idx=label_to_idx,
        mode="train",
    )
    _, _, _, apply_distill = ds[0]
    assert apply_distill is True
