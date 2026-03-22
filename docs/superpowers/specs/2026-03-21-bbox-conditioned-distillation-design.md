# Bbox-Conditioned Feature Distillation for Quantized Wildlife Classification

## Overview

Replace the current EfficientNet-B3 (42MB, quantization-hostile) with a small, quantization-friendly student model trained via dual-input feature distillation from DINOv2-Large. The teacher sees bbox-cropped animal regions; the student sees full camera trap images. The student learns to extract focused animal features without needing bboxes at inference time.

**Goal**: A quantized ONNX model under 5MB that runs in-browser at >90% accuracy, with improved bobcat classification.

## Architecture & Models

### Teacher: DINOv2-Large
- ViT-L/14, ~300M params, frozen throughout training
- Provides CLS token embeddings (1024-dim) as distillation targets
- Loaded via `torch.hub` or `timm`
- Runs forward passes only on bbox-cropped images (no gradients stored)

### Students (train both, pick winner)
| Model | Params | FP32 Size | INT8 Size | Source |
|---|---|---|---|---|
| MobileNetV3-Small | 2.5M | ~10MB | ~2.5MB | torchvision |
| MobileNetV4-Conv-S | 3.8M | ~15MB | ~4MB | timm |

Both use ImageNet-pretrained weights. Classifier head replaced with `nn.Linear(feat_dim, 6)`.

### Projection Head
Maps student penultimate features to teacher CLS token space:
```
nn.Sequential(
    nn.Linear(student_feat_dim, 512),
    nn.ReLU(),
    nn.Linear(512, 1024)
)
```
Discarded after Phase 1 training. Used only for computing distillation loss.

### Class Ordering
Alphabetical, matching training code's `sorted()` convention:
```python
["bobcat", "coyote", "deer", "empty", "fox", "mountain_lion"]
```

## Data Pipeline

### Dual-Input Construction
Each training sample produces two views:

1. **Student input**: Full camera trap image with train augmentations (RandomResizedCrop(224), RandomHorizontalFlip, ColorJitter, ImageNet normalize)
2. **Teacher input**: Same image cropped to bbox, then eval transforms only (Resize(256), CenterCrop(224), ImageNet normalize). No augmentation — teacher features must be a stable distillation target.

### Bbox Handling Rules
| Condition | Teacher Input | Student Input | Loss |
|---|---|---|---|
| Has bbox (animal class) | Bbox crop | Full image | CE + distillation |
| No bbox (animal class, ~2%) | Full image | Full image | CE + distillation |
| Empty class | Not computed | Full image | CE only |

### Dataset Implementation
Modify `WildlifeDataset.__getitem__()` to return:
```python
(student_tensor, teacher_tensor, label, has_bbox)
```
- `has_bbox` is False for empty class samples, True otherwise
- Controls whether distillation loss is applied per-sample

### Data Source
- Existing split JSON files (v2) with embedded bbox annotations
- ~15,750 training samples, ~2,771 val, ~2,779 test
- ~98% of non-empty images have bounding boxes

## Training Pipeline

### Phase 1: Bbox-Conditioned Feature Distillation (~20 epochs)

**Loss function:**
```
L = alpha(t) * L_distill + (1 - alpha(t)) * L_ce
```
- `L_ce`: CrossEntropyLoss with class weights (handles class imbalance)
- `L_distill`: MSE between projected student features and DINOv2 CLS token. Applied only when `has_bbox=True`.
- `alpha(t)`: Linear anneal from 0.9 to 0.3 over training epochs

**Training details:**
- Optimizer: Adam, LR=0.001
- Batch size: 32
- Teacher: `torch.no_grad()`, `.eval()` mode always
- Student: full backbone unfrozen, ImageNet-pretrained init
- Projection head trained alongside student

**Validation:** Student evaluated on val set using standard eval transforms (no bbox, no teacher) after each epoch. This reflects real-world inference conditions. Best checkpoint saved by val accuracy.

**Checkpointing:** Every epoch, save to `/opt/ml/checkpoints/`:
- model_state_dict
- optimizer_state_dict
- projection_head_state_dict
- epoch number
- best_val_acc

### Phase 2: Quantization-Aware Training (~10 epochs)

- Load best Phase 1 checkpoint (projection head discarded)
- Insert fake quantization nodes via `torch.ao.quantization.prepare_qat`
- QAT config: per-channel QInt8 weights, per-tensor QUInt8 activations
- Fine-tune with CE loss only (class weights), LR=0.0001
- Standard train/val loop, save best checkpoint
- Separate SageMaker job from Phase 1

### Phase 3: ONNX Export & Quantization

1. Convert QAT model to quantized PyTorch model via `torch.ao.quantization.convert`
2. Export to ONNX (opset 13, dynamic batch axis, constant folding)
3. Run `onnxruntime.quantization.quant_pre_process`
4. Apply `quantize_static` with:
   - Per-channel QInt8 weights
   - Per-tensor QUInt8 activations
   - MinMax calibration
   - Exclude final classifier layer
5. Validate: compare FP32 vs INT8 accuracy on test set

## Success Criteria

| Metric | Target |
|---|---|
| Quantized model size | < 5MB (ideally < 3MB) |
| FP32 → INT8 accuracy drop | < 2% |
| Overall test accuracy | >= 90% |
| Bobcat test accuracy | >= 95% (improvement over current 94.8%) |
| Browser inference latency | < 100ms at 224x224 |

## SageMaker Infrastructure

### Spot Training
```python
use_spot_instances=True
max_run=3600 * 4          # 4 hours max compute
max_wait=3600 * 5         # 5 hours including spot wait
checkpoint_s3_uri=f's3://{bucket}/checkpoints/'
checkpoint_local_path='/opt/ml/checkpoints'
```

### Instance
`ml.g4dn.xlarge` (T4 16GB) for both phases. DINOv2-Large frozen forward + student training fits in 16GB. If VRAM is tight, reduce batch size to 16.

### Job Structure
- Phase 1 and Phase 2 run as separate SageMaker training jobs
- Both student models (MobileNetV3-Small, MobileNetV4-Conv-S) trained as separate Phase 1 jobs
- Phase 2 runs only on the better-performing student from Phase 1
- Total estimated cost with spot: ~$0.50-$0.75

### Dependencies
Add to requirements.txt:
- `timm` (for MobileNetV4-Conv-S and DINOv2)

### Training Script
- Unified entry point with `--phase` flag (1 or 2) and `--student-arch` flag
- Checkpoint resume logic: scan `/opt/ml/checkpoints/` on startup
- Phase 1: loads teacher, builds projection head, runs distillation loop
- Phase 2: loads Phase 1 best checkpoint, inserts QAT nodes, fine-tunes

## Deployment (Out of Scope for Training Plan)

Client-side browser inference on Vercel free tier:
- Quantized ONNX served as static file from `/public/`
- ONNX Runtime Web (WASM backend) for inference
- No bboxes needed at inference — student learned to focus on animals via distillation
- Model cached via Service Worker after first download
- Deployment implementation will be planned separately after a working quantized model is produced

## Upgrade Path

If penultimate-layer feature alignment proves insufficient:
- Add multi-scale alignment at 2-3 intermediate stages with lightweight projection heads
- This increases training signal but adds implementation complexity
- Only pursue if Phase 1 accuracy is unsatisfactory
