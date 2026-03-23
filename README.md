# Wildlife Camera Trap Classifier — Edge-Deployed via Knowledge Distillation

**42MB EfficientNet-B3 → 3.7MB quantized EfficientNet-Lite0, achieving 91.2% accuracy via bbox-conditioned feature distillation from a 300M-parameter vision transformer.**

This project compresses DINOv2-Large's visual understanding into a model 80x smaller through bbox-conditioned feature distillation and ONNX INT8 quantization — producing a 6-class wildlife classifier that runs in <100ms on commodity hardware via WebAssembly.

---

## Results

### Model Progression

| Model | FP32 Accuracy | INT8 Accuracy | INT8 Size | Quantization |
|---|---|---|---|---|
| EfficientNet-B3 (baseline) | 94.6% | Collapsed (0.7%) | 12MB | Failed — SE blocks + SiLU destroy INT8 signal |
| MobileNetV3-Small (distilled) | 95.4% | Collapsed (24.7%) | 3.5MB | Failed — HardSigmoid + SE blocks still break PTQ |
| **EfficientNet-Lite0 (distilled)** | **97.1%** | **91.2%** | **3.7MB** | **Success — no SE blocks, pure Conv+ReLU6** |

### Per-Class Accuracy (Final Quantized Model)

| Class | FP32 | INT8 | Drop | Main Confusion |
|---|---|---|---|---|
| Bobcat | 95.8% | 93.8% | -2.0% | → coyote, empty |
| Coyote | 97.6% | 94.4% | -3.2% | → empty, bobcat |
| Deer | 97.2% | 95.2% | -2.0% | → coyote |
| Empty | 95.6% | 84.4% | -11.2% | → bobcat, coyote |
| Fox | 99.0% | 87.6% | -11.4% | → coyote |
| Mountain Lion | 100% | 96.9% | -3.1% | — |

---

## Technical Highlights

- **Bbox-conditioned feature distillation** — DINOv2-Large teacher receives bbox-cropped animal regions while the student sees full camera trap images, teaching the student to localize animals without needing bounding boxes at inference
- **11x model compression** — 42MB FP32 EfficientNet-B3 → 3.7MB INT8 EfficientNet-Lite0 while maintaining 91.2% overall accuracy
- **Systematic quantization investigation** — Diagnosed why EfficientNet-B3 and MobileNetV3 fail under INT8 (Squeeze-and-Excitation blocks + HardSigmoid produce activation ranges that collapse under quantization), validated with per-layer ablation
- **Dual-input data pipeline** — custom `Dataset` serving separate transform paths (augmented full-frame for student, deterministic bbox-crop for teacher) with per-sample loss masking for the empty class
- **AWS spot training infrastructure** — SageMaker spot instances with epoch-level checkpointing for automatic recovery, total pipeline cost ~$2

---

## Why This Architecture

### The Problem

The original approach fine-tuned EfficientNet-B3 (42MB, 12M params) on camera trap images. It achieved 94.6% accuracy at FP32 but was **quantization-hostile** — every INT8 conversion method (static PTQ with MinMax/Entropy calibration, per-channel/per-tensor, with/without ONNX pre-processing) caused catastrophic accuracy collapse. The model predicted a single class for all inputs.

Root cause: EfficientNet-B3's Squeeze-and-Excitation blocks produce channel attention values in [0, 1] via sigmoid/SiLU. These tiny activation ranges are destroyed by INT8 quantization, and the damage compounds through 7 MBConv stages.

MobileNetV3-Small was tried next (designed by Google for quantization with HardSigmoid replacing sigmoid), but even quantizing just 5 Conv layers collapsed accuracy — the 28 HardSigmoid → Mul attention patterns still interact badly with ONNX Runtime's QuantizeLinear/DequantizeLinear node insertion.

### The Solution

**EfficientNet-Lite0** — Google's variant that strips out all SE blocks and replaces Swish with ReLU6, creating a model that is architecturally identical to EfficientNet-B0 but with every quantization-hostile operation removed.

Combined with **bbox-conditioned feature distillation** from DINOv2-Large, the student learns the teacher's rich visual representations (fine-grained species discrimination, spatial attention) while using only quantization-safe operations.

```
Training:                                 Inference:
┌─────────────┐    bbox crop    ┌────────────────┐
│  Raw Image  │───────────────→│ DINOv2 Teacher  │──→ 1024-d target
│  (full)     │                └────────────────┘         │
│             │    full image   ┌────────────────┐        │ MSE
│             │───────────────→│  Lite0 Student  │──→ projected──┘
└─────────────┘                └────────────────┘
                                       │
                                    logits──→ CE Loss

                               ┌────────────────┐
                  full image → │  Lite0 Student  │ → class prediction
                               └────────────────┘
                               (no teacher, no bbox)
```

---

## Model Architecture

### Teacher: DINOv2-Large (ViT-L/14)

| | |
|---|---|
| Parameters | ~300M (frozen throughout training) |
| Output | 1024-dim CLS token embeddings |
| Input | Bbox-cropped images, eval transforms only |
| Optimization | FP16 autocast on CUDA to halve VRAM on 16GB T4 |

### Student: EfficientNet-Lite0

| | |
|---|---|
| Parameters | 4.7M |
| FP32 Size | 12.85 MB |
| INT8 Size | 3.70 MB |
| Feature Dim | 1280 |
| Key Property | No SE blocks, no Sigmoid/HardSigmoid, ReLU6 only |
| Source | timm (`efficientnet_lite0`, ImageNet-pretrained) |

### Why Not the Others?

| Architecture | Outcome | Root Cause |
|---|---|---|
| EfficientNet-B3 | INT8 collapsed to single class | SE blocks + SiLU activations |
| MobileNetV3-Small | INT8 collapsed (even 5 layers quantized) | 28 HardSigmoid→Mul SE patterns break ONNX Runtime Q/DQ insertion |
| MobileNetV4-Conv-S | Not tested (deprioritized after MV3 failure) | — |
| **EfficientNet-Lite0** | **91.2% INT8** | **No SE blocks, pure Conv+ReLU6** |

### Projection Head

A 2-layer MLP (1280 → 512 → 1024) bridges the student's feature space to the teacher's CLS token dimension. **Discarded after training** — zero overhead in the deployed model.

---

## Training Pipeline

### Phase 1: Bbox-Conditioned Feature Distillation

**Loss:**
```
L = α(t) · L_distill + (1 − α(t)) · L_ce

L_distill = MSE(ProjectionHead(student_features), teacher_CLS)    # masked for empty class
L_ce      = CrossEntropy(student_logits, labels)                   # weighted for class imbalance
α(t)      = 0.9 − 0.6 · (t / T)                                  # linear anneal: 0.9 → 0.3
```

Alpha starts distillation-dominant (0.9) to imprint the teacher's spatial features early, then shifts toward classification (0.3) so the student develops its own decision boundaries. Per-sample masking excludes empty-class images from distillation.

| Hyperparameter | Value |
|---|---|
| Epochs | 20 |
| Optimizer | Adam, LR=1e-3 with CosineAnnealingLR |
| Batch size | 32 train / 64 val |
| Training time | ~3 hours on ml.g4dn.xlarge |
| Result | 93.4% val accuracy (97.1% on eval set) |

### Phase 2: ONNX Export & Post-Training Quantization

The distilled EfficientNet-Lite0 is exported to ONNX (opset 13) and quantized via ONNX Runtime's `quantize_static`:

| Config | Value |
|---|---|
| Weight quantization | Per-channel QInt8 |
| Activation quantization | Per-tensor QUInt8 |
| Calibration | MinMax, 532 images (100 per class) |
| Calibration method | Static quantization |
| Result | 91.2% accuracy, 3.70 MB |

Note: QAT was attempted on both MobileNetV3-Small and EfficientNet-Lite0 but module fusion issues with `torch.ao.quantization` caused accuracy degradation rather than improvement. Direct PTQ on Lite0's quantization-friendly architecture proved more effective.

---

## Data Pipeline

### Dual-Input Dataset

The custom `DistillationDataset` returns 4 values per sample:

```python
(student_tensor, teacher_tensor, label, apply_distill)
```

| Input | Source | Transforms | Purpose |
|---|---|---|---|
| `student_tensor` | Full image | RandomResizedCrop, HFlip, ColorJitter, normalize | Augmented view for robust training |
| `teacher_tensor` | Bbox crop | Resize(256), CenterCrop(224), normalize | Deterministic view for stable targets |
| `apply_distill` | Class label | `False` for empty, `True` otherwise | Per-sample loss masking |

### Bbox Handling

| Condition | Teacher sees | Student sees | Loss |
|---|---|---|---|
| Has bbox (animal) | Bbox crop | Full image | CE + distillation |
| No bbox (animal, ~2%) | Full image fallback | Full image | CE + distillation |
| Empty class | N/A | Full image | CE only |

### Path Unification

Split JSONs use a unified `image_path` format (`"bobcat/abc123.jpg"`) resolved at runtime against a configurable `--data-dir`. The same splits work across local development (`./data/s3+expanded_empty`) and SageMaker (`/opt/ml/input/data/train`).

---

## Infrastructure

### SageMaker Spot Training

| Config | Value |
|---|---|
| Instance | `ml.g4dn.xlarge` (NVIDIA T4, 16GB VRAM) |
| Spot instances | Enabled (~70% cost savings) |
| Checkpointing | Every epoch to S3, auto-resume on spot interruption |
| Phase 1 time | ~3 hours (distillation with DINOv2-Large teacher) |
| Total cost | ~$2 for full pipeline (both student architectures + QAT attempts) |

---

## Dataset

Derived from the [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/) dataset (ECCV 2018), filtered to 6 target species across ~22,500 images with COCO-format bounding box annotations.

| Class | Train | Val | Test |
|---|---|---|---|
| Bobcat | ~4,900 | — | — |
| Coyote | ~6,200 | — | — |
| Deer | ~6,000 | — | — |
| Empty | ~4,200 | — | — |
| Fox | ~1,100 | — | — |
| Mountain Lion | ~32 | — | — |

Class imbalance addressed through per-class loss weighting. Mountain lion has very few samples but achieves near-perfect accuracy due to distinctive visual features.

**Citation:** Sara Beery, Grant Van Horn, Pietro Perona. *Recognition in Terra Incognita.* Proceedings of the 15th European Conference on Computer Vision (ECCV 2018).

---

## Project Structure

```
├── sagemaker_training/
│   ├── models.py                  # Student factory: MobileNetV3/V4/Lite0 with feature extraction
│   ├── teacher.py                 # DINOv2-Large: frozen CLS token extraction with AMP
│   ├── distillation_loss.py       # Projection head, alpha annealing, per-sample masked loss
│   ├── wildlife_dataloader_sm.py  # Dual-input dataset with bbox-aware cropping
│   ├── distill_train.py           # Phase 1: distillation training loop
│   ├── qat_train.py               # Phase 2: quantization-aware fine-tuning
│   ├── distill_launcher.py        # SageMaker orchestration with spot instances
│   └── model_efficient_net.py     # Legacy EfficientNet-B3 (original baseline)
├── EC2+s3/
│   ├── onnx_quantization/
│   │   ├── export_onnx.py         # PyTorch → ONNX export
│   │   ├── calibrate_onnx.py      # ONNX Runtime static quantization
│   │   └── evaluate_quant_onnx.py # Per-class accuracy, confusion matrices
│   ├── training/                  # Training run history (runs 1-6)
│   └── data_augmentation_pipeline/
│       ├── data_stratification.py # Train/val/test split creation
│       └── wildlife_dataloader.py # Local data loading with bbox support
├── scripts/
│   └── regenerate_splits.py       # Split JSON path unification
├── tests/                         # 18 tests covering models, loss, dataset, paths
├── docs/superpowers/
│   ├── specs/                     # Design specification
│   └── plans/                     # Implementation plan
└── Dockerfile
```

---

## Training Run History

| Run | Model | Approach | Best Val Acc | Quantizable? |
|---|---|---|---|---|
| 1 | Basic CNN | Direct training | — | N/A |
| 2-3 | Basic CNN | + expanded empty class | — | N/A |
| 4 | EfficientNet-B3 | Direct training | 89.6% | No (SE + SiLU) |
| 5 | MobileNetV3-Small | DINOv2 distillation | 91.6% | No (HardSigmoid + SE) |
| **6** | **EfficientNet-Lite0** | **DINOv2 distillation** | **93.4%** | **Yes (91.2% INT8)** |

---

## Key Dependencies

PyTorch 2.0+ | torchvision | timm >= 1.0 | ONNX Runtime >= 1.16 | SageMaker SDK v2 | boto3
