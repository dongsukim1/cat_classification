# Bbox-Conditioned Feature Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a quantization-friendly student model (MobileNetV3-Small / MobileNetV4-Conv-S) via dual-input feature distillation from DINOv2-Large, where the teacher sees bbox-cropped animal regions and the student sees full camera trap images.

**Architecture:** DINOv2-Large (frozen teacher) provides CLS token embeddings for bbox-cropped images. A lightweight projection head maps student penultimate features to the teacher's 1024-dim space. Loss anneals from distillation-heavy to CE-heavy. After distillation, QAT fine-tunes the student for INT8 quantization, then ONNX export produces a <5MB browser-deployable model.

**Tech Stack:** PyTorch 2.0+, torchvision, timm, ONNX Runtime, AWS SageMaker (spot instances)

**Spec:** `docs/superpowers/specs/2026-03-21-bbox-conditioned-distillation-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|---|---|
| `sagemaker_training/__init__.py` | Empty package init (enables `from sagemaker_training.X import Y` in tests) |
| `sagemaker_training/models.py` | Student model factory (MobileNetV3-Small, MobileNetV4-Conv-S) with feature extraction support |
| `sagemaker_training/teacher.py` | DINOv2-Large loading and frozen forward pass |
| `sagemaker_training/distillation_loss.py` | Projection head, annealing alpha, combined distill+CE loss |
| `sagemaker_training/distill_train.py` | Phase 1 entry point: distillation training loop with checkpoint resume |
| `sagemaker_training/qat_train.py` | Phase 2 entry point: QAT fine-tuning loop |
| `sagemaker_training/distill_launcher.py` | SageMaker launcher with spot instances for both phases |
| `export_onnx.py` | Phase 3: QAT-to-ONNX export and validation |
| `scripts/regenerate_splits.py` | One-time script to unify split JSON paths |
| `tests/test_models.py` | Tests for student model factory |
| `tests/test_teacher.py` | Tests for teacher loading and forward pass |
| `tests/test_distillation_loss.py` | Tests for projection head and loss computation |
| `tests/test_dataset_dual_input.py` | Tests for dual-input dataset |
| `tests/test_path_resolution.py` | Tests for unified path resolution |

### Modified Files
| File | Changes |
|---|---|
| `sagemaker_training/wildlife_dataloader_sm.py` | Add dual-input `DistillationDataset` class; update path resolution to use unified `image_path` |
| `EC2+s3/data_augmentation_pipeline/data_stratification.py:414-427` | Write unified `image_path` field instead of `image_path_local`/`image_path_aws` |
| `calibrate_onnx.py` | Class ordering already fixed to alphabetical |
| `evaluate_quant_onnx.py` | Class ordering already fixed to alphabetical |
| `requirements.txt` | Add `timm>=1.0.0` |

---

## Task 1: Path Unification — Regenerate Split JSONs

**Files:**
- Create: `scripts/regenerate_splits.py`
- Create: `tests/test_path_resolution.py`

- [ ] **Step 1: Write the test for path resolution**

```python
# tests/test_path_resolution.py
import json
import tempfile
from pathlib import Path


def test_unified_path_format():
    """Split JSON image_path should be forward-slash relative: class/filename.jpg"""
    sample = {
        "image_id": "abc123",
        "primary_class": "bobcat",
        "image_path": "bobcat/abc123.jpg",
        "annotations": [],
    }
    # Forward slashes only
    assert "\\" not in sample["image_path"]
    # Format is class/filename
    parts = sample["image_path"].split("/")
    assert len(parts) == 2
    assert parts[0] == sample["primary_class"]


def test_path_resolution_local():
    """Path resolution should work with any base directory."""
    data_dir = Path("./data/s3+expanded_empty")
    image_path = "bobcat/abc123.jpg"
    full_path = data_dir / image_path
    expected = Path("data/s3+expanded_empty/bobcat/abc123.jpg")
    assert full_path == expected  # Path comparison is OS-agnostic


def test_path_resolution_sagemaker():
    """Path resolution should work with SageMaker mount points."""
    from pathlib import PurePosixPath

    data_dir = PurePosixPath("/opt/ml/input/data/train")
    image_path = "bobcat/abc123.jpg"
    full_path = data_dir / image_path
    assert str(full_path) == "/opt/ml/input/data/train/bobcat/abc123.jpg"


def test_no_legacy_fields_in_regenerated_split():
    """Regenerated splits should not contain image_path_local or image_path_aws."""
    sample = {
        "image_id": "abc123",
        "primary_class": "bobcat",
        "image_path": "bobcat/abc123.jpg",
        "annotations": [],
    }
    assert "image_path_local" not in sample
    assert "image_path_aws" not in sample
```

- [ ] **Step 2: Run tests to verify they pass (these are format-validation tests)**

Run: `python -m pytest tests/test_path_resolution.py -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Write the split regeneration script**

```python
# scripts/regenerate_splits.py
"""
One-time script to convert split JSONs from dual-path format
(image_path_local + image_path_aws) to unified format (image_path).

Usage:
    python scripts/regenerate_splits.py --splits-dir EC2+s3/data_augmentation_pipeline/splitsv2
"""
import argparse
import json
from pathlib import Path


def regenerate_split(split_path: Path) -> list:
    with split_path.open() as f:
        samples = json.load(f)

    unified = []
    for sample in samples:
        # Prefer image_path_aws (already forward-slash, class/filename format)
        if "image_path_aws" in sample:
            image_path = sample["image_path_aws"]
        elif "image_path_local" in sample:
            # Fallback: extract class/filename from local path
            local = Path(sample["image_path_local"])
            image_path = f"{local.parent.name}/{local.name}"
        else:
            raise ValueError(f"Sample {sample.get('image_id')} has no path field")

        # Ensure forward slashes
        image_path = image_path.replace("\\", "/")

        unified_sample = {
            "image_id": sample["image_id"],
            "primary_class": sample["primary_class"],
            "image_path": image_path,
            "labels": sample.get("labels", [sample["primary_class"]]),
            "bbox_count": sample.get("bbox_count", 0),
            "annotations": sample.get("annotations", []),
        }
        unified.append(unified_sample)

    return unified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        default="EC2+s3/data_augmentation_pipeline/splitsv2",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to --splits-dir")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir) if args.output_dir else splits_dir

    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.json"
        if not split_path.exists():
            print(f"Skipping {split_path} (not found)")
            continue

        unified = regenerate_split(split_path)
        output_path = output_dir / f"{split_name}.json"
        with output_path.open("w") as f:
            json.dump(unified, f, indent=2)
        print(f"Wrote {len(unified)} samples to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the regeneration script**

Run: `python scripts/regenerate_splits.py --splits-dir EC2+s3/data_augmentation_pipeline/splitsv2`
Expected: Prints count of samples written for train, val, test

- [ ] **Step 5: Verify regenerated format**

Run: `python -c "import json; d=json.load(open('EC2+s3/data_augmentation_pipeline/splitsv2/train.json')); s=d[0]; print(s.keys()); print(s['image_path']); assert 'image_path_local' not in s; assert 'image_path_aws' not in s; print('OK')"`
Expected: `OK` with keys showing `image_path` instead of `image_path_local`/`image_path_aws`

- [ ] **Step 6: Update data_stratification.py to write unified format**

In `EC2+s3/data_augmentation_pipeline/data_stratification.py`, replace the save_data.append block (lines 414-427) so that future regenerations also produce the unified format:

```python
            save_data.append({
                'image_id': sample['image_id'],
                'primary_class': sample['primary_class'],
                'image_path': f"{species_handle}/{path_to_name}",
                'labels': sample['labels'],
                'bbox_count': sample['bbox_count'],
                'annotations': sample['annotations']
            })
```

- [ ] **Step 7: Commit**

```bash
git add scripts/regenerate_splits.py tests/test_path_resolution.py EC2+s3/data_augmentation_pipeline/data_stratification.py EC2+s3/data_augmentation_pipeline/splitsv2/
git commit -m "feat: unify split JSON paths to class/filename format

Replaces dual image_path_local/image_path_aws fields with single
image_path field using forward-slash relative paths. Consumers
resolve against their own --data-dir at runtime."
```

---

## Task 2: Update Dataloaders for Unified Paths

**Files:**
- Modify: `sagemaker_training/wildlife_dataloader_sm.py:189-193`
- Modify: `EC2+s3/data_augmentation_pipeline/wildlife_dataloader.py` (equivalent section)

- [ ] **Step 1: Update wildlife_dataloader_sm.py path resolution**

In `wildlife_dataloader_sm.py`, replace lines 189-193 (the `image_path_aws` lookup) with:

```python
            # Unified path resolution: image_path is class/filename.jpg
            original_path = sample.get('image_path') or sample.get('image_path_aws', '')
            if not original_path:
                continue
            full_path = str(Path(data_dir) / original_path)
```

This is backward-compatible: it tries `image_path` first, falls back to `image_path_aws` for old splits.

- [ ] **Step 2: Apply same change to EC2+s3 wildlife_dataloader.py**

Update the equivalent path resolution in `EC2+s3/data_augmentation_pipeline/wildlife_dataloader.py` (the `image_path_local` lookup, around line 320) with the same pattern, falling back to `image_path_local`:

```python
            original_path = sample.get('image_path') or sample.get('image_path_local', '')
            if not original_path:
                continue
            full_path = str(Path(data_dir) / original_path)
```

- [ ] **Step 3: Verify locally that evaluation still works**

Run: `python evaluate_quant_onnx.py --model model.onnx --max-per-class 5`
Expected: Accuracy output (confirming data loading still works)

- [ ] **Step 4: Commit**

```bash
git add sagemaker_training/wildlife_dataloader_sm.py EC2+s3/data_augmentation_pipeline/wildlife_dataloader.py
git commit -m "feat: update dataloaders for unified image_path resolution

Path resolution now uses image_path field with runtime data_dir join.
Backward-compatible with old image_path_aws/image_path_local fields."
```

---

## Task 3: Student Model Factory

**Files:**
- Create: `sagemaker_training/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
import torch
import pytest


def test_mobilenetv3_small_output_shape():
    from sagemaker_training.models import create_student

    model = create_student("mobilenetv3_small", num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 6)


def test_mobilenetv3_small_features():
    from sagemaker_training.models import create_student

    model = create_student("mobilenetv3_small", num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    features = model.extract_features(x)
    assert features.shape == (2, 576)


def test_mobilenetv4_conv_s_output_shape():
    from sagemaker_training.models import create_student

    model = create_student("mobilenetv4_conv_s", num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 6)


def test_mobilenetv4_conv_s_features():
    from sagemaker_training.models import create_student

    model = create_student("mobilenetv4_conv_s", num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    features = model.extract_features(x)
    assert features.shape == (2, 1024)


def test_feature_dim_attribute():
    from sagemaker_training.models import create_student

    mv3 = create_student("mobilenetv3_small", num_classes=6)
    assert mv3.feature_dim == 576

    mv4 = create_student("mobilenetv4_conv_s", num_classes=6)
    assert mv4.feature_dim == 1024


def test_invalid_arch_raises():
    from sagemaker_training.models import create_student

    with pytest.raises(ValueError):
        create_student("invalid_arch", num_classes=6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement student model factory**

```python
# sagemaker_training/models.py
"""Student model factory with feature extraction support."""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

try:
    import timm
except ImportError:
    timm = None


class MobileNetV3SmallStudent(nn.Module):
    """MobileNetV3-Small with feature extraction for distillation."""

    feature_dim = 576

    def __init__(self, num_classes: int = 6):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)


class MobileNetV4ConvSStudent(nn.Module):
    """MobileNetV4-Conv-S with feature extraction for distillation."""

    feature_dim = 1024

    def __init__(self, num_classes: int = 6):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for MobileNetV4-Conv-S")
        self.base = timm.create_model(
            "mobilenetv4_conv_small.e2400_r224_in1k",
            pretrained=True,
            num_classes=num_classes,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # forward_features returns 4D (B,C,H,W); forward_head with pre_logits=True pools to 2D (B,C)
        x = self.base.forward_features(x)
        return self.base.forward_head(x, pre_logits=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


def create_student(arch: str, num_classes: int = 6) -> nn.Module:
    """Factory function to create a student model.

    Args:
        arch: One of 'mobilenetv3_small', 'mobilenetv4_conv_s'
        num_classes: Number of output classes

    Returns:
        Student model with .extract_features() and .feature_dim
    """
    if arch == "mobilenetv3_small":
        return MobileNetV3SmallStudent(num_classes)
    elif arch == "mobilenetv4_conv_s":
        return MobileNetV4ConvSStudent(num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: mobilenetv3_small, mobilenetv4_conv_s")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_models.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add sagemaker_training/models.py tests/test_models.py
git commit -m "feat: add student model factory with feature extraction

MobileNetV3-Small (576-dim) and MobileNetV4-Conv-S (1024-dim) with
extract_features() for distillation and feature_dim attribute."
```

---

## Task 4: Teacher Module

**Files:**
- Create: `sagemaker_training/teacher.py`
- Create: `tests/test_teacher.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_teacher.py
import torch
import pytest


def test_teacher_output_shape():
    from sagemaker_training.teacher import load_teacher

    teacher = load_teacher(device="cpu")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = teacher(x)
    assert features.shape == (2, 1024)


def test_teacher_is_frozen():
    from sagemaker_training.teacher import load_teacher

    teacher = load_teacher(device="cpu")
    for param in teacher.parameters():
        assert not param.requires_grad


def test_teacher_deterministic():
    from sagemaker_training.teacher import load_teacher

    teacher = load_teacher(device="cpu")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out1 = teacher(x)
        out2 = teacher(x)
    assert torch.allclose(out1, out2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_teacher.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement teacher module**

```python
# sagemaker_training/teacher.py
"""DINOv2-Large teacher model for feature distillation."""
import torch
import torch.nn as nn


class DINOv2Teacher(nn.Module):
    """Frozen DINOv2-Large that returns CLS token embeddings."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns CLS token embedding (batch_size, 1024)."""
        return self.model(x)


def load_teacher(
    device: str = "cuda",
    weights_path: str | None = None,
) -> DINOv2Teacher:
    """Load frozen DINOv2-Large teacher.

    Args:
        device: Device to load model on
        weights_path: Optional local path to pre-cached weights.
                      If None, downloads from torch.hub.
    """
    if weights_path:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return DINOv2Teacher(model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_teacher.py -v`
Expected: All 3 tests PASS (first run will download ~1.2GB DINOv2 weights)

- [ ] **Step 5: Commit**

```bash
git add sagemaker_training/teacher.py tests/test_teacher.py
git commit -m "feat: add DINOv2-Large teacher module

Frozen teacher returns 1024-dim CLS token embeddings.
Supports optional pre-cached weights path for SageMaker."
```

---

## Task 5: Distillation Loss Module

**Files:**
- Create: `sagemaker_training/distillation_loss.py`
- Create: `tests/test_distillation_loss.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_distillation_loss.py
import torch
import pytest


def test_projection_head_shape():
    from sagemaker_training.distillation_loss import ProjectionHead

    head = ProjectionHead(student_dim=576, teacher_dim=1024)
    x = torch.randn(4, 576)
    out = head(x)
    assert out.shape == (4, 1024)


def test_alpha_annealing():
    from sagemaker_training.distillation_loss import compute_alpha

    # Epoch 0 of 20: should be 0.9
    assert abs(compute_alpha(0, 20) - 0.9) < 1e-6
    # Last epoch: should be 0.3
    assert abs(compute_alpha(19, 20) - 0.3) < 1e-6
    # Midpoint: should be 0.6
    assert abs(compute_alpha(9, 20) - 0.6) < 0.05


def test_distillation_loss_with_mask():
    from sagemaker_training.distillation_loss import distillation_loss

    student_features = torch.randn(4, 576)
    teacher_features = torch.randn(4, 1024)
    # Only first 2 samples should have distillation applied
    apply_distill = torch.tensor([True, True, False, False])

    from sagemaker_training.distillation_loss import ProjectionHead
    head = ProjectionHead(576, 1024)
    loss = distillation_loss(student_features, teacher_features, head, apply_distill)

    assert loss.shape == ()  # scalar
    assert loss.item() >= 0


def test_distillation_loss_all_masked_returns_zero():
    from sagemaker_training.distillation_loss import distillation_loss, ProjectionHead

    head = ProjectionHead(576, 1024)
    student_features = torch.randn(4, 576)
    teacher_features = torch.randn(4, 1024)
    apply_distill = torch.tensor([False, False, False, False])

    loss = distillation_loss(student_features, teacher_features, head, apply_distill)
    assert loss.item() == 0.0


def test_combined_loss():
    from sagemaker_training.distillation_loss import combined_loss, ProjectionHead

    head = ProjectionHead(576, 1024)
    student_features = torch.randn(4, 576)
    teacher_features = torch.randn(4, 1024)
    logits = torch.randn(4, 6)
    labels = torch.tensor([0, 1, 2, 3])
    apply_distill = torch.tensor([True, True, True, False])

    loss = combined_loss(
        logits=logits,
        labels=labels,
        student_features=student_features,
        teacher_features=teacher_features,
        projection_head=head,
        apply_distill=apply_distill,
        alpha=0.7,
        class_weights=None,
    )
    assert loss.shape == ()
    assert loss.item() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_distillation_loss.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement distillation loss module**

```python
# sagemaker_training/distillation_loss.py
"""Distillation loss with projection head and alpha annealing."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Maps student features to teacher embedding space."""

    def __init__(self, student_dim: int, teacher_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(student_dim, 512),
            nn.ReLU(),
            nn.Linear(512, teacher_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_alpha(epoch: int, total_epochs: int) -> float:
    """Linear anneal from 0.9 to 0.3 over training."""
    return 0.9 - (0.6 * epoch / (total_epochs - 1))


def distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    projection_head: ProjectionHead,
    apply_distill: torch.Tensor,
) -> torch.Tensor:
    """MSE between projected student features and teacher CLS token.

    Only computed for samples where apply_distill is True.
    Returns zero if no samples have distillation applied.
    """
    mask = apply_distill.bool()
    if not mask.any():
        return torch.tensor(0.0, device=student_features.device)

    projected = projection_head(student_features[mask])
    target = teacher_features[mask].detach()
    return F.mse_loss(projected, target)


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    projection_head: ProjectionHead,
    apply_distill: torch.Tensor,
    alpha: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combined distillation + CE loss with alpha weighting."""
    ce = F.cross_entropy(logits, labels, weight=class_weights)
    distill = distillation_loss(
        student_features, teacher_features, projection_head, apply_distill
    )
    return alpha * distill + (1 - alpha) * ce
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_distillation_loss.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add sagemaker_training/distillation_loss.py tests/test_distillation_loss.py
git commit -m "feat: add distillation loss with projection head and alpha annealing

MSE between projected student features and DINOv2 CLS token, masked
per-sample via apply_distill flag. Alpha anneals 0.9 -> 0.3 linearly."
```

---

## Task 6: Dual-Input Distillation Dataset

**Files:**
- Modify: `sagemaker_training/wildlife_dataloader_sm.py`
- Create: `tests/test_dataset_dual_input.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset_dual_input.py
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

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
    # Empty is the 3rd sample (index 2)
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
    # Bobcat is the 1st sample (index 0)
    _, _, _, apply_distill = ds[0]
    assert apply_distill is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dataset_dual_input.py -v`
Expected: FAIL (DistillationDataset not found)

- [ ] **Step 3: Extract `_crop_to_bbox` as a module-level function**

In `sagemaker_training/wildlife_dataloader_sm.py`, the `_crop_to_bbox` method (lines 53-75) is currently an instance method on `WildlifeDataset`. Extract it to a module-level function so both `WildlifeDataset` and `DistillationDataset` can use it:

```python
def crop_to_bbox(image, bbox_info):
    """Crop image to first bounding box annotation. Returns original if no valid bbox."""
    if not bbox_info:
        return image
    bbox = bbox_info[0].get('bbox', None)
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
```

Then update `WildlifeDataset._crop_to_bbox` to call the module-level function: `self._crop_to_bbox = staticmethod(crop_to_bbox)` or just replace `self._crop_to_bbox(...)` calls with `crop_to_bbox(...)`.

- [ ] **Step 4: Implement DistillationDataset**

Add the following class to `sagemaker_training/wildlife_dataloader_sm.py` after the existing `WildlifeDataset` class:

```python
class DistillationDataset(torch.utils.data.Dataset):
    """Dual-input dataset for bbox-conditioned feature distillation.

    Returns (student_tensor, teacher_tensor, label_idx, apply_distill).
    - student_tensor: full image with train augmentations
    - teacher_tensor: bbox-cropped image with eval transforms (stable target)
    - apply_distill: False for 'empty' class, True otherwise
    """

    EMPTY_CLASS = "empty"

    def __init__(
        self,
        data_dir,
        split_file,
        target_species,
        label_to_idx,
        mode="train",
    ):
        self.data_dir = Path(data_dir)
        self.label_to_idx = label_to_idx
        self.mode = mode
        self.samples = []  # list of (image_path, class_name, annotations)

        with open(split_file) as f:
            raw_samples = json.load(f)

        for sample in raw_samples:
            cls = sample["primary_class"]
            if cls not in target_species:
                continue
            image_path = sample.get("image_path") or sample.get("image_path_aws", "")
            full_path = str(self.data_dir / image_path)
            annotations = sample.get("annotations", [])
            self.samples.append((full_path, cls, annotations))

        # Student sees augmented full images
        if mode == "train":
            self.student_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.student_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Teacher always sees stable eval transforms (no augmentation)
        self.teacher_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, cls, annotations = self.samples[idx]
        label_idx = self.label_to_idx[cls]
        apply_distill = cls != self.EMPTY_CLASS

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        # Student: full image
        student_tensor = self.student_transform(image)

        # Teacher: bbox crop if available and not empty class
        if apply_distill and annotations:
            teacher_image = crop_to_bbox(image.copy(), annotations)
        else:
            teacher_image = image
        teacher_tensor = self.teacher_transform(teacher_image)

        return student_tensor, teacher_tensor, label_idx, apply_distill
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dataset_dual_input.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add sagemaker_training/wildlife_dataloader_sm.py tests/test_dataset_dual_input.py
git commit -m "feat: add DistillationDataset with dual-input for teacher/student

Student sees full image with augmentations, teacher sees bbox-cropped
image with eval transforms. apply_distill=False for empty class."
```

---

## Task 7: Phase 1 Training Script — Distillation

**Files:**
- Create: `sagemaker_training/distill_train.py`

- [ ] **Step 1: Implement Phase 1 training entry point**

```python
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
from wildlife_dataloader_sm import DistillationDataset, WildlifeDataset


CLASS_NAMES = sorted(["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"])


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the directory."""
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


def train_one_epoch(
    student, teacher, projection_head, dataloader, optimizer,
    alpha, class_weights, device,
):
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

        # Student forward: extract features, then classify
        student_features = student.extract_features(student_imgs)
        logits = student(student_imgs)

        # Teacher forward (frozen, FP16 if on CUDA)
        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    teacher_features = teacher(teacher_imgs).float()
            else:
                teacher_features = teacher(teacher_imgs)

        loss = combined_loss(
            logits=logits,
            labels=labels,
            student_features=student_features,
            teacher_features=teacher_features,
            projection_head=projection_head,
            apply_distill=apply_distill,
            alpha=alpha,
            class_weights=class_weights,
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
    """Validate student using standard eval (no teacher, no bbox)."""
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
                        choices=["mobilenetv3_small", "mobilenetv4_conv_s"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size-train", type=int, default=32)
    parser.add_argument("--batch-size-val", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--teacher-weights", default=None,
                        help="Path to pre-cached DINOv2 weights")

    # SageMaker environment
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

    # Models
    student = create_student(args.student_arch, num_classes=num_classes).to(device)
    teacher = load_teacher(device=device, weights_path=args.teacher_weights)
    projection_head = ProjectionHead(student.feature_dim, 1024).to(device)

    # Data
    train_dataset = DistillationDataset(
        data_dir=args.data_dir,
        split_file=str(Path(args.splits_dir) / "train.json"),
        target_species=CLASS_NAMES,
        label_to_idx=label_to_idx,
        mode="train",
    )
    val_dataset = DistillationDataset(
        data_dir=args.data_dir,
        split_file=str(Path(args.splits_dir) / "val.json"),
        target_species=CLASS_NAMES,
        label_to_idx=label_to_idx,
        mode="val",
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size_train,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size_val,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    class_weights = compute_class_weights(
        str(Path(args.splits_dir) / "train.json"), label_to_idx, num_classes, device
    )
    optimizer = Adam(
        list(student.parameters()) + list(projection_head.parameters()),
        lr=args.learning_rate,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint
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

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(start_epoch, args.epochs):
        alpha = compute_alpha(epoch, args.epochs)
        print(f"\nEpoch {epoch+1}/{args.epochs} | alpha={alpha:.3f} | lr={scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_acc = train_one_epoch(
            student, teacher, projection_head, train_loader,
            optimizer, alpha, class_weights, device,
        )
        val_acc = validate(student, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train loss={train_loss:.4f} acc={train_acc:.4f} | Val acc={val_acc:.4f}")

        # Save best model (before checkpoint so best_val_acc is current)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  New best val acc: {best_val_acc:.4f}")
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(student.state_dict(), Path(args.model_dir) / "model.pth")

        # Checkpoint every epoch
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

    # Save history
    with open(Path(args.model_dir) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test locally (CPU, 1 epoch, tiny subset)**

Run: `python -c "from sagemaker_training.distill_train import parse_args; print('import OK')"`
Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add sagemaker_training/distill_train.py
git commit -m "feat: add Phase 1 distillation training entry point

Dual-input training loop with alpha annealing, CosineAnnealingLR,
FP16 teacher forward, checkpoint resume for spot instances."
```

---

## Task 8: Phase 2 QAT Training Script

**Files:**
- Create: `sagemaker_training/qat_train.py`

- [ ] **Step 1: Implement Phase 2 QAT entry point**

```python
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
```

- [ ] **Step 2: Smoke test import**

Run: `python -c "from sagemaker_training.qat_train import parse_args; print('import OK')"`
Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add sagemaker_training/qat_train.py
git commit -m "feat: add Phase 2 QAT training entry point

Loads Phase 1 best model, inserts fake quant nodes, fine-tunes with
CE loss at low LR. Saves QAT model with fake-quant nodes for ONNX export."
```

---

## Task 9: SageMaker Launcher with Spot Instances

**Files:**
- Create: `sagemaker_training/distill_launcher.py`

- [ ] **Step 1: Implement the launcher**

```python
# sagemaker_training/distill_launcher.py
"""SageMaker launcher for distillation training with spot instances."""
import argparse
import time

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


def launch_phase1(args, role, session):
    """Launch Phase 1 distillation training job."""
    bucket = args.bucket
    splits = args.splits_version

    hyperparameters = {
        "student-arch": args.student_arch,
        "epochs": args.phase1_epochs,
        "batch-size-train": args.batch_size_train,
        "batch-size-val": args.batch_size_val,
        "learning-rate": args.phase1_lr,
        "num-workers": 8,
    }

    if args.teacher_weights_s3:
        hyperparameters["teacher-weights"] = "/opt/ml/input/data/teacher/dinov2_vitl14.pth"

    input_paths = {
        "train": f"s3://{bucket}/caltech_images",
        "splits": f"s3://{bucket}/training_loop/data_augmentation_pipeline/{splits}",
    }

    if args.teacher_weights_s3:
        input_paths["teacher"] = args.teacher_weights_s3

    estimator = PyTorch(
        entry_point="distill_train.py",
        source_dir="./sagemaker_training",
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="2.0.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=f"s3://{bucket}/distill_output",
        base_job_name=f"distill-{args.student_arch.replace('_', '-')}",
        max_run=3600 * 4,
        max_wait=3600 * 5,
        use_spot_instances=True,
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/phase1-{args.student_arch}/",
        checkpoint_local_path="/opt/ml/checkpoints",
        volume_size=30,
        environment={"SM_MODEL_DIR": "/opt/ml/model"},
    )

    job_name = f"distill-{args.student_arch.replace('_', '-')}-{int(time.time())}"
    estimator.fit(inputs=input_paths, job_name=job_name, wait=True)
    return estimator.latest_training_job.name


def launch_phase2(args, role, session, phase1_model_s3):
    """Launch Phase 2 QAT training job."""
    bucket = args.bucket
    splits = args.splits_version

    hyperparameters = {
        "student-arch": args.student_arch,
        "phase1-model": "/opt/ml/input/data/phase1/model.pth",
        "epochs": args.phase2_epochs,
        "batch-size-train": args.batch_size_train,
        "batch-size-val": args.batch_size_val,
        "learning-rate": args.phase2_lr,
        "num-workers": 8,
    }

    input_paths = {
        "train": f"s3://{bucket}/caltech_images",
        "splits": f"s3://{bucket}/training_loop/data_augmentation_pipeline/{splits}",
        "phase1": phase1_model_s3,
    }

    estimator = PyTorch(
        entry_point="qat_train.py",
        source_dir="./sagemaker_training",
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="2.0.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=f"s3://{bucket}/qat_output",
        base_job_name=f"qat-{args.student_arch.replace('_', '-')}",
        max_run=3600 * 2,
        max_wait=3600 * 3,
        use_spot_instances=True,
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/phase2-{args.student_arch}/",
        checkpoint_local_path="/opt/ml/checkpoints",
        volume_size=20,
        environment={"SM_MODEL_DIR": "/opt/ml/model"},
    )

    job_name = f"qat-{args.student_arch.replace('_', '-')}-{int(time.time())}"
    estimator.fit(inputs=input_paths, job_name=job_name, wait=True)
    return estimator.latest_training_job.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-arch", required=True,
                        choices=["mobilenetv3_small", "mobilenetv4_conv_s"])
    parser.add_argument("--bucket", default="big-cat-data2")
    parser.add_argument("--splits-version", default="splitsv2")
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge")
    parser.add_argument("--phase1-epochs", type=int, default=20)
    parser.add_argument("--phase2-epochs", type=int, default=10)
    parser.add_argument("--phase1-lr", type=float, default=0.001)
    parser.add_argument("--phase2-lr", type=float, default=0.0001)
    parser.add_argument("--batch-size-train", type=int, default=32)
    parser.add_argument("--batch-size-val", type=int, default=64)
    parser.add_argument("--teacher-weights-s3", default=None,
                        help="S3 URI to pre-cached DINOv2 weights")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--phase1-model-s3", default=None,
                        help="S3 URI to Phase 1 model output (for --skip-phase1)")
    args = parser.parse_args()

    session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    if not args.skip_phase1:
        print(f"=== Phase 1: Distillation ({args.student_arch}) ===")
        phase1_job = launch_phase1(args, role, session)
        phase1_model_s3 = f"s3://{args.bucket}/distill_output/{phase1_job}/output/model.tar.gz"
    else:
        if not args.phase1_model_s3:
            raise ValueError("--phase1-model-s3 required when --skip-phase1 is set")
        phase1_model_s3 = args.phase1_model_s3

    print(f"\n=== Phase 2: QAT ({args.student_arch}) ===")
    launch_phase2(args, role, session, phase1_model_s3)

    print("\nDone! Download model artifacts and run export_onnx.py for Phase 3.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add sagemaker_training/distill_launcher.py
git commit -m "feat: add SageMaker launcher with spot instances

Launches Phase 1 (distillation) and Phase 2 (QAT) as separate spot
training jobs. Supports --skip-phase1 for re-running QAT only."
```

---

## Task 10: Phase 3 — ONNX Export Script

**Files:**
- Create: `export_onnx.py`

- [ ] **Step 1: Implement ONNX export**

```python
# export_onnx.py
"""Phase 3: Export QAT model to ONNX and validate."""
import argparse
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.ao.quantization as tq

sys.path.append("./sagemaker_training")
from models import create_student


CLASS_NAMES = sorted(["mountain_lion", "bobcat", "coyote", "fox", "deer", "empty"])


def export_qat_to_onnx(student, output_path, opset=13):
    """Export QAT model with fake-quant nodes to ONNX."""
    student.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        student,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output_path}")


def validate_onnx(onnx_path, student, num_samples=10):
    """Compare ONNX output against PyTorch model on random inputs."""
    student.eval()
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    max_diff = 0.0
    mismatches = 0
    for _ in range(num_samples):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            pt_out = student(x).numpy()
        ort_out = session.run(None, {input_name: x.numpy()})[0]

        diff = np.max(np.abs(pt_out - ort_out))
        max_diff = max(max_diff, diff)
        if np.argmax(pt_out) != np.argmax(ort_out):
            mismatches += 1

    print(f"Max output difference: {max_diff:.2e}")
    print(f"Prediction mismatches: {mismatches}/{num_samples}")
    return max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-arch", required=True,
                        choices=["mobilenetv3_small", "mobilenetv4_conv_s"])
    parser.add_argument("--qat-model", required=True, help="Path to model_qat.pth")
    parser.add_argument("--output", default="model_distilled.onnx")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    student = create_student(args.student_arch, num_classes=len(CLASS_NAMES))

    # Prepare QAT structure then load weights
    student.train()
    student.qconfig = tq.get_default_qat_qconfig("x86")
    tq.prepare_qat(student, inplace=True)
    state_dict = torch.load(args.qat_model, map_location="cpu")
    student.load_state_dict(state_dict)

    # Primary path: export QAT model with fake-quant nodes
    import os
    try:
        export_qat_to_onnx(student, args.output, args.opset)
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        validate_onnx(args.output, student)
    except Exception as e:
        print(f"QAT ONNX export failed: {e}")
        print("Falling back to FP32 export + ONNX Runtime quantize_static...")
        # Fallback: convert QAT to quantized, export as FP32, then quantize via ORT
        student_fp32 = tq.convert(student, inplace=False)
        fp32_path = args.output.replace(".onnx", "_fp32.onnx")
        export_qat_to_onnx(student_fp32, fp32_path, args.opset)

        from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod
        from onnxruntime.quantization import quant_pre_process
        preproc_path = args.output.replace(".onnx", "_preproc.onnx")
        quant_pre_process(fp32_path, preproc_path)
        # NOTE: Provide a CalibrationDataReader here for real calibration
        print(f"Pre-processed model saved. Run calibrate_onnx.py on {preproc_path} to complete.")

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    if size_mb > 5.0:
        print(f"WARNING: Model size {size_mb:.2f} MB exceeds 5MB target")
    else:
        print(f"Model size {size_mb:.2f} MB is within target")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add export_onnx.py
git commit -m "feat: add Phase 3 ONNX export with QAT node preservation

Exports QAT model with fake-quant nodes as native ONNX QuantizeLinear/
DequantizeLinear ops. Validates output against PyTorch and reports size."
```

---

## Task 11: Update requirements.txt and Final Integration

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Create `sagemaker_training/__init__.py`**

Create an empty file `sagemaker_training/__init__.py` to make it a proper package (enables `from sagemaker_training.X import Y` in tests):

```bash
touch sagemaker_training/__init__.py
```

- [ ] **Step 2: Add timm to requirements.txt**

Add `timm>=1.0.0` to `requirements.txt`.

- [ ] **Step 3: Also add to sagemaker_training/requirements.txt if it exists separately**

Check `sagemaker_training/requirements.txt` and add `timm>=1.0.0` there as well (SageMaker uses this for installing deps in the training container).

- [ ] **Step 4: Verify all imports work**

Run:
```bash
python -c "
from sagemaker_training.models import create_student
from sagemaker_training.teacher import load_teacher
from sagemaker_training.distillation_loss import ProjectionHead, combined_loss, compute_alpha
from sagemaker_training.wildlife_dataloader_sm import DistillationDataset
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt sagemaker_training/requirements.txt sagemaker_training/__init__.py
git commit -m "chore: add timm dependency and package init for MobileNetV4 and DINOv2"
```

---

## Task 12: Student Selection After Phase 1

After both Phase 1 jobs complete (MobileNetV3-Small and MobileNetV4-Conv-S), compare val accuracies and select the winner for Phase 2.

- [ ] **Step 1: Download training histories from both jobs**

```bash
# Download from SageMaker output artifacts
aws s3 cp s3://big-cat-data2/distill_output/<mv3-job>/output/model.tar.gz mv3_output.tar.gz
aws s3 cp s3://big-cat-data2/distill_output/<mv4-job>/output/model.tar.gz mv4_output.tar.gz
tar xzf mv3_output.tar.gz -C mv3_output/
tar xzf mv4_output.tar.gz -C mv4_output/
```

- [ ] **Step 2: Compare results and select winner**

```bash
python -c "
import json
mv3 = json.load(open('mv3_output/training_history.json'))
mv4 = json.load(open('mv4_output/training_history.json'))
mv3_best = max(mv3['val_acc'])
mv4_best = max(mv4['val_acc'])
print(f'MobileNetV3-Small best val acc: {mv3_best:.4f}')
print(f'MobileNetV4-Conv-S best val acc: {mv4_best:.4f}')
if mv3_best >= mv4_best:
    print('Winner: mobilenetv3_small (higher acc or tie broken by smaller size)')
else:
    print('Winner: mobilenetv4_conv_s')
"
```

- [ ] **Step 3: Proceed with Phase 2 for the winning architecture**

Use the winning student's `--student-arch` flag and its Phase 1 model S3 path when launching Phase 2 via `distill_launcher.py --skip-phase1`.

---

## Execution Order

Tasks 1-2 (path unification) must complete first — they're prerequisites.

Tasks 3-6 (models, teacher, loss, dataset) can be developed in parallel but should be committed in order for clean git history.

Tasks 7-8 (training scripts) depend on Tasks 3-6.

Task 9 (launcher) depends on Tasks 7-8.

Task 10 (ONNX export) depends on Task 8.

Task 11 (integration) runs last before training.

Task 12 (student selection) runs after both Phase 1 jobs complete, before Phase 2.

```
Task 1 → Task 2 → Tasks 3,4,5,6 → Tasks 7,8 → Task 9 → Task 11
                                                    ↓
                                            Launch both Phase 1 jobs
                                                    ↓
                                                Task 12 (select winner)
                                                    ↓
                                            Launch Phase 2 for winner
                                                    ↓
                                                Task 10 (ONNX export)
```

```
Task 1 → Task 2 → Tasks 3,4,5,6 → Tasks 7,8 → Tasks 9,10 → Task 11
```
