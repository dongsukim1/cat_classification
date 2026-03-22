# Automated Wildlife Camera Trap Classifier

A lightweight, browser-deployable wildlife classification model trained via bbox-conditioned feature distillation from DINOv2-Large into a quantized MobileNet student. The goal is a <5MB INT8 ONNX model that runs client-side at >90% accuracy across 6 species, with particular emphasis on bobcat classification.

## Background

The dataset is derived from the [LILA BC Caltech Camera Trap](https://lila.science/datasets/caltech-camera-traps/) dataset, originally published in "Recognition in Terra Incognita" (ECCV 2018). The original publication benchmarked SOTA computer vision models on cross-location generalization across 21 species and ~243,000 images. This project filters to 6 species that match the geographical distributions of interest:

| Class | Description |
|---|---|
| bobcat | Primary target species |
| coyote | |
| deer | |
| empty | No animal present |
| fox | |
| mountain_lion | |

### Data Collection and Preprocessing

Images were filtered from an AWS S3 mirror of the original dataset into 6 class folders. Exploratory data analysis revealed several challenges common to camera trap data: partial animal captures, motion blur, nighttime IR images, and heavy class imbalances (the empty class had <200 images initially). The empty class was augmented with images from a similar dataset, improving accuracy across all classes by ~10%. Split JSONs contain ~15,750 train / ~2,771 val / ~2,779 test samples, with ~98% of non-empty images having bounding box annotations.

## Architecture

### The Problem with the Previous Approach

The original pipeline used EfficientNet-B3 (42MB FP32), which proved quantization-hostile — INT8 conversion caused severe accuracy degradation with no practical path to recovery.

### Bbox-Conditioned Feature Distillation

The core insight: a large frozen teacher (DINOv2-Large) sees bbox-cropped animal regions and produces stable feature targets, while a small student sees full camera trap images and learns to extract the same focused features. At inference time, the student doesn't need bounding boxes — it has learned to ignore irrelevant background through distillation.

**Teacher**: DINOv2-Large (ViT-L/14, ~300M params, frozen)
- Produces 1024-dim CLS token embeddings as distillation targets
- Receives bbox-cropped images with eval-only transforms (no augmentation) for stable targets
- Runs in FP16 via AMP to halve VRAM usage

**Students** (train both, pick winner):

| Model | Params | FP32 Size | INT8 Size | Source |
|---|---|---|---|---|
| MobileNetV3-Small | 2.5M | ~10MB | ~2.5MB | torchvision |
| MobileNetV4-Conv-S | 3.8M | ~15MB | ~4MB | timm |

**Projection Head**: 2-layer MLP (student_feat_dim → 512 → 1024) that maps student penultimate features into teacher embedding space. Discarded after Phase 1.

### Dual-Input Data Pipeline

Each training sample produces two views:

- **Student input**: Full image with train augmentations (RandomResizedCrop, HorizontalFlip, ColorJitter, ImageNet normalize)
- **Teacher input**: Bbox-cropped image with eval transforms only (Resize(256), CenterCrop(224), ImageNet normalize)

Empty class samples skip distillation entirely (CE loss only), forcing the student to learn empty-scene features independently.

## Training Pipeline

### Phase 1: Feature Distillation (~20 epochs)

Combined loss with linear alpha annealing:

```
L = alpha(t) * L_distill + (1 - alpha(t)) * L_ce
```

- `L_distill`: MSE between projected student features and DINOv2 CLS tokens (masked per-sample for empty class)
- `L_ce`: CrossEntropyLoss with per-class weights for imbalance handling
- `alpha(t)`: Anneals linearly from 0.9 → 0.3 over training, gradually shifting emphasis from distillation to classification
- Optimizer: Adam, LR=0.001 with CosineAnnealingLR
- Validation uses standard eval transforms (no bbox, no teacher) to reflect real inference conditions

### Phase 2: Quantization-Aware Training (~10 epochs)

- Loads best Phase 1 checkpoint (projection head discarded)
- Inserts fake quantization nodes via `torch.ao.quantization.prepare_qat`
- QAT config: per-channel QInt8 weights, per-tensor QUInt8 activations
- Fine-tunes with CE loss only at LR=0.0001 (10x lower to preserve Phase 1 knowledge)

### Phase 3: ONNX Export

Exports the QAT model directly to ONNX — fake-quant nodes become native `QuantizeLinear`/`DequantizeLinear` ops, preserving learned quantization parameters. Fallback path applies ONNX Runtime `quantize_static` with calibration data if direct export fails.

Validation: compares ONNX output vs PyTorch on random inputs, checks max diff and prediction agreement.

## SageMaker Infrastructure

Both phases run as separate SageMaker training jobs on `ml.g4dn.xlarge` (T4 16GB) with spot instances:

- Spot training: max_run=4h, max_wait=5h
- Epoch-based checkpointing to S3 for spot interruption recovery
- Phase 2 receives Phase 1 model via S3 data channel
- Estimated total cost: ~$0.50–$0.75

## Project Structure

```
├── sagemaker_training/
│   ├── teacher.py              # DINOv2-Large teacher (frozen, CLS token extraction)
│   ├── models.py               # Student model factory (MobileNetV3/V4 with feature extraction)
│   ├── distillation_loss.py    # Projection head, alpha annealing, combined loss
│   ├── wildlife_dataloader_sm.py  # Dual-input dataset with bbox-aware cropping
│   ├── distill_train.py        # Phase 1 distillation training loop
│   ├── qat_train.py            # Phase 2 QAT training loop
│   ├── distill_launcher.py     # SageMaker job orchestration with spot instances
│   ├── sagemaker_launcher.py   # Legacy SageMaker launcher
│   └── sagemaker_train.py      # Legacy training script
├── export_onnx.py              # Phase 3 ONNX export with QAT support
├── calibrate_onnx.py           # ONNX Runtime static quantization calibration
├── evaluate_quant_onnx.py      # Per-class accuracy, confusion matrix evaluation
├── local_data_preprocessing.py # One-time data organization from COCO-format annotations
├── scripts/
│   └── regenerate_splits.py    # Regenerate split JSONs with unified image_path format
├── ONNX_conversion.py          # Legacy EfficientNet ONNX export
├── ONNX_diagnostics.py         # Legacy ONNX validation
├── Dockerfile                  # Python 3.9 training container
└── requirements.txt
```

## Success Criteria

| Metric | Target |
|---|---|
| Quantized model size | < 5MB (ideally < 3MB) |
| FP32 → INT8 accuracy drop | < 2% |
| Overall test accuracy | >= 90% |
| Bobcat test accuracy | >= 95% |
| Browser inference latency | < 100ms at 224x224 |

Minimum viable fallback: 87% overall with <3MB is acceptable as an intermediate result.

## Deployment Target

Client-side browser inference via ONNX Runtime Web (WASM backend) on Vercel free tier. The quantized ONNX model is served as a static file and cached via Service Worker after first download. No bounding boxes needed at inference — the student learned to focus on animals through distillation.

## Key Dependencies

- PyTorch 2.0+ with torchvision
- timm >= 1.0.0 (MobileNetV4-Conv-S, DINOv2)
- ONNX Runtime >= 1.16.0 (inference and quantization)
- SageMaker SDK >= 2.150.0

## Citation

Sara Beery, Grant Van Horn, Pietro Perona. *Recognition in Terra Incognita.* Proceedings of the 15th European Conference on Computer Vision (ECCV 2018).
