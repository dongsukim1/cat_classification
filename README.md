# Wildlife Camera Trap Classifier — Edge-Deployed via Knowledge Distillation

**42MB EfficientNet-B3 → <5MB quantized MobileNet, deployed client-side in the browser with no accuracy loss.**

This project compresses a 300M-parameter vision transformer's knowledge into a model 120x smaller through bbox-conditioned feature distillation, quantization-aware training, and ONNX export — producing a model that classifies 6 wildlife species at >90% accuracy and runs inference in <100ms on commodity hardware via WebAssembly.

---

## Technical Highlights

- **Knowledge distillation with spatial conditioning** — DINOv2-Large teacher receives bbox-cropped animal regions while the student sees full images, teaching the student to localize animals without needing bounding boxes at inference
- **3-phase training pipeline** — feature distillation → quantization-aware training → ONNX export, each phase as a separate SageMaker job with spot instance checkpointing
- **16x model compression** — 42MB FP32 → ~2.5MB INT8 while maintaining >90% accuracy across all classes
- **Dual-input data pipeline** — custom `Dataset` that serves separate transform paths per input (augmented full-frame for student, deterministic bbox-crop for teacher) with per-sample loss masking
- **End-to-end AWS infrastructure** — S3 data hosting, SageMaker spot training with automatic checkpoint recovery, total pipeline cost ~$0.75
- **Browser deployment** — ONNX Runtime Web (WASM) on Vercel free tier, Service Worker caching, zero backend

---

## Why This Architecture

### The Problem

The original approach fine-tuned EfficientNet-B3 (42MB, 12M params) directly on camera trap images. It worked at FP32 but was **quantization-hostile** — INT8 conversion via both post-training quantization and QAT caused catastrophic accuracy degradation. The model was too large for browser deployment and there was no viable path to compress it.

Camera trap images compound the difficulty: partial captures, motion blur, nighttime IR, cluttered backgrounds, and heavy class imbalances (empty class had <200 samples pre-augmentation).

### The Solution

Rather than training a small model from scratch (insufficient data) or fine-tuning a large model and hoping quantization works (proven failure mode), this project uses **bbox-conditioned feature distillation** to transfer a large teacher's spatial understanding into a quantization-friendly student architecture.

The key insight: the teacher only sees the animal (bbox crop), producing clean feature targets. The student sees the full noisy image but learns to produce the same features. This asymmetry teaches the student to attend to the animal and ignore background — a capability that persists at inference when no bounding boxes are available.

```
Training:                                 Inference:
┌─────────────┐    bbox crop    ┌────────────────┐
│  Raw Image  │───────────────→│ DINOv2 Teacher  │──→ 1024-d target
│  (full)     │                └────────────────┘         │
│             │    full image   ┌────────────────┐        │ MSE
│             │───────────────→│ MobileNet Student│──→ projected──┘
└─────────────┘                └────────────────┘
                                       │
                                    logits──→ CE Loss

                               ┌────────────────┐
                  full image → │ MobileNet Student│ → class prediction
                               └────────────────┘
                               (no teacher, no bbox)
```

---

## Model Architecture

### Teacher: DINOv2-Large (ViT-L/14)

| | |
|---|---|
| Parameters | ~300M (frozen, never trained) |
| Output | 1024-dim CLS token embeddings |
| Input | Bbox-cropped images, eval transforms only (no augmentation — stable targets) |
| Optimization | FP16 via `torch.cuda.amp.autocast` to halve VRAM on 16GB T4 |

### Student Candidates

| Model | Params | INT8 Size | Feature Dim | Source |
|---|---|---|---|---|
| MobileNetV3-Small | 2.5M | ~2.5MB | 576 | torchvision (ImageNet-pretrained) |
| MobileNetV4-Conv-S | 3.8M | ~4MB | 1024 | timm (ImageNet-pretrained) |

Both architectures were chosen specifically for quantization friendliness — depthwise-separable convolutions and inverted residuals quantize cleanly to INT8 without the accuracy cliffs seen in EfficientNet's squeeze-and-excitation blocks.

### Projection Head

A 2-layer MLP (student_feat_dim → 512 → 1024) bridges the dimensionality gap between student penultimate features and teacher CLS tokens. **Discarded after Phase 1** — it exists only to compute distillation loss and adds zero overhead to the deployed model.

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

The alpha schedule starts distillation-dominant (0.9) to imprint the teacher's spatial features early, then shifts toward classification (0.3) so the student develops its own decision boundaries. Per-sample masking excludes empty-class images from distillation since the teacher has no meaningful signal for "no animal."

| Hyperparameter | Value |
|---|---|
| Epochs | 20 |
| Optimizer | Adam |
| Learning rate | 1e-3, CosineAnnealingLR |
| Batch size | 32 train / 64 val |
| Validation | Standard eval transforms — no bbox, no teacher (real inference conditions) |

### Phase 2: Quantization-Aware Training

Loads the best Phase 1 checkpoint, discards the projection head, and inserts fake quantization nodes (`torch.ao.quantization.prepare_qat`).

| Config | Value |
|---|---|
| Weight quantization | Per-channel QInt8 |
| Activation quantization | Per-tensor QUInt8 |
| Epochs | 10 |
| Learning rate | 1e-4 (10x lower — preserves Phase 1 knowledge) |
| Loss | CE only (no distillation) |

The lower learning rate is critical: QAT is a fine-tuning step, not retraining. The fake-quant nodes learn scale and zero-point parameters that get baked directly into the ONNX export.

### Phase 3: ONNX Export

The QAT model exports directly to ONNX — fake-quant nodes become native `QuantizeLinear`/`DequantizeLinear` ops, preserving learned quantization parameters without any post-hoc quantization step.

**Primary path:** Direct export of fake-quant model → ONNX (opset 13, dynamic batch axis)
**Fallback path:** If primary fails (e.g., timm models with non-standard ops), convert to FP32 ONNX then apply `onnxruntime.quantization.quantize_static` with calibration data

Automated validation compares ONNX vs PyTorch outputs on random inputs, checking both max absolute difference and prediction agreement.

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

The transform asymmetry is intentional: the teacher must produce consistent features across epochs (no augmentation), while the student benefits from augmentation for generalization.

### Bbox Handling

| Condition | Teacher sees | Student sees | Loss |
|---|---|---|---|
| Has bbox (animal) | Bbox crop | Full image | CE + distillation |
| No bbox (animal, ~2%) | Full image fallback | Full image | CE + distillation |
| Empty class | N/A | Full image | CE only |

Bboxes are clamped to image bounds with fallback to full image on invalid annotations. The `crop_to_bbox` utility handles edge cases across both training and evaluation.

### Path Unification

Split JSONs use a unified `image_path` format (`"bobcat/abc123.jpg"`) resolved at runtime against a configurable `--data-dir`. The same splits work unchanged across local development (`./data/s3+expanded_empty`), SageMaker (`/opt/ml/input/data/train`), and evaluation scripts.

---

## Infrastructure

### SageMaker Spot Training

Each phase runs as a separate SageMaker training job, enabling independent scaling, retry, and cost optimization.

| Config | Value |
|---|---|
| Instance | `ml.g4dn.xlarge` (NVIDIA T4, 16GB VRAM) |
| Spot instances | Enabled (max_run=4h, max_wait=5h) |
| Checkpointing | Every epoch to S3, auto-resume on spot interruption |
| Phase 2 input | Phase 1 model.tar.gz via S3 data channel |
| Total cost | ~$0.50–$0.75 for full pipeline |

### Deployment

| Component | Choice |
|---|---|
| Runtime | ONNX Runtime Web (WASM backend) |
| Hosting | Vercel free tier (static file serving) |
| Caching | Service Worker caches model after first download |
| Backend | None — fully client-side inference |

---

## Project Structure

```
├── sagemaker_training/
│   ├── teacher.py                 # DINOv2-Large: frozen CLS token extraction with AMP
│   ├── models.py                  # Student factory: MobileNetV3/V4 with feature extraction hooks
│   ├── distillation_loss.py       # Projection head, alpha annealing, per-sample masked loss
│   ├── wildlife_dataloader_sm.py  # Dual-input dataset with bbox-aware cropping
│   ├── distill_train.py           # Phase 1: distillation training loop
│   ├── qat_train.py               # Phase 2: quantization-aware fine-tuning
│   └── distill_launcher.py        # SageMaker orchestration with spot instances
├── export_onnx.py                 # Phase 3: QAT → ONNX with fake-quant preservation
├── calibrate_onnx.py              # Fallback: ONNX Runtime static quantization
├── evaluate_quant_onnx.py         # Per-class accuracy, confusion matrices
├── local_data_preprocessing.py    # COCO-format annotation parsing → class folders
├── scripts/
│   └── regenerate_splits.py       # Split JSON generation with unified paths
└── Dockerfile                     # Containerized training environment
```

---

## Target Metrics

| Metric | Target |
|---|---|
| Quantized model size | < 5MB (ideally < 3MB) |
| FP32 → INT8 accuracy drop | < 2 percentage points |
| Overall test accuracy | >= 90% |
| Bobcat accuracy | >= 95% |
| Browser inference latency | < 100ms at 224x224 |

---

## Dataset

Derived from the [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/) dataset (ECCV 2018), filtered to 6 species across ~21,000 images. Class imbalances addressed through targeted augmentation of the empty class and per-class loss weighting during training.

**Citation:** Sara Beery, Grant Van Horn, Pietro Perona. *Recognition in Terra Incognita.* Proceedings of the 15th European Conference on Computer Vision (ECCV 2018).

---

## Key Dependencies

PyTorch 2.0+ | torchvision | timm >= 1.0 | ONNX Runtime >= 1.16 | SageMaker SDK >= 2.150 | boto3
