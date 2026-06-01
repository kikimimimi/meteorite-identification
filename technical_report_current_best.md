# Technical Report: Current Best Pipeline

Updated: 2026-06-01

## Current Best Route

The strongest recorded route is now `train_dinov2_lora_v2.py`:

```text
original images
+ DINOv2 ViT-B/14
+ LoRA / PEFT fine-tuning
+ CLS + patch-token mean pooling
+ deeper MLP head
+ BatchNorm + Dropout
+ label smoothing
+ conservative regularization
+ flip TTA
```

Best recorded public score:

```text
0.80628
```

Best recorded command:

```bash
python train_dinov2_lora_v2.py --data-root .. --batch-size 4 --amp
```

Path note: `--data-root ..` means the dataset is expected in the parent directory of the current shell location. If running from the repository root with data files in the root, use `--data-root .`.

## Current Root Code

The root directory keeps the current LoRA route and shared utilities:

```text
dataset.py
submission_utils.py
train_dinov2_lora.py
train_dinov2_lora_v2.py
```

`train_dinov2_lora_v2.py` is the best recorded route. `train_dinov2_lora.py` is the earlier LoRA v1 baseline.

Historical scripts live under:

```text
historical_attempts/
```

## What Worked

### DINOv2 LoRA v2 / PEFT Fine-Tuning

The largest improvement came from adapting DINOv2 to the meteorite domain instead of using it only as a frozen feature extractor.

Recorded scores:

```text
original images + frozen DINOv2 ViT-B/14 multicenter top105: ~0.74
frozen DINOv2 ViT-B/14 + strong INR top105: 0.76439
DINOv2 LoRA v1 top105: 0.79581
DINOv2 LoRA v1 top106: 0.80208
DINOv2 LoRA v2: 0.80628
```

Interpretation:

```text
Frozen DINOv2 already gives a strong visual prior.
LoRA adapts a small number of parameters to the meteorite/non-meteorite boundary.
The v2 head and regularization reduce overfitting on the small dataset.
```

### LoRA v2 Improvements

Compared with v1, the v2 script adds:

```text
MLP classification head
BatchNorm
Dropout
Label-smoothed BCE
lower learning rate
stronger weight decay
5-fold prediction averaging by default
```

Default v2 settings:

```text
model_name = dinov2_vitb14
image_size = 518
pool = cls_mean
folds = 5
epochs = 15
patience = 3
lr = 1e-4
weight_decay = 1e-3
lora_rank = 8
lora_alpha = 16
lora_last_n_blocks = 4
head_hidden_dim = 256
head_dropout = 0.3
label_smoothing = 0.05
tta = flip
```

### Original Images

Original images outperformed the SAM-cropped branch. The likely reason is that global shape, scale, edge context, shadows, and capture conditions still carry useful signal.

Recorded scores:

```text
SAM/CLIP crop + DINOv2 ViT-B/14: ~0.70
original images + frozen DINOv2 ViT-B/14: ~0.74
```

### CLS + Patch Mean Pooling

The model pools:

```text
concat(CLS token, mean(patch tokens))
```

This keeps both image-level semantics and local surface information. The local patch signal is important because meteorite cues often appear in texture, crust, pores, cracks, and granular surface structure.

## Not Used in the Final Route

### INR / SIREN Descriptor Branch

The current final code does **not** use INR.

INR fits a small SIREN network to each image and extracts fitting statistics such as reconstruction error, PSNR, residual statistics, activation statistics, and fitting dynamics.

It helped before LoRA:

```text
frozen DINOv2 ViT-B/14 baseline: ~0.74
frozen DINOv2 + strong INR: 0.76439
```

But it did not help the LoRA route:

```text
LoRA v1 top106: 0.80208
LoRA + INR blend top106: 0.79166
LoRA v2: 0.80628
```

Conclusion: INR is a valuable historical experiment, but not part of the final highest-scoring method.

### SAM / CLIP

SAM/CLIP preprocessing was tested to isolate the stone object. It did not improve the final score, likely because it removes useful context and shifts the image distribution.

### Label Propagation

Label propagation used train/test feature-space neighbors to propagate labels. It underperformed because nearest visual neighbors are not always same-class examples in this task.

### kNN Analysis

kNN was useful for manual inspection and understanding suspicious samples, but not as a final classifier or submission generator.

### Meta-INR

Meta-INR tried to learn a nonlinear representation over INR descriptors. It did not improve the final route, likely because the dataset is too small for a reliable meta-encoder.

### K-fold Multi-seed INR Rank Average

The K-fold multi-seed INR route was more stable but lower scoring:

```text
strong INR single split top105: 0.76439
K-fold multi-seed INR top104: 0.75789
K-fold multi-seed INR top105: 0.75392
```

### LoRA Seed Sensitivity

Not all LoRA seeds are useful. Seed4 top105 produced:

```text
0.74345
```

This confirms that future LoRA ensembling should filter bad seeds before averaging.

## Practical Recommendation

Use `train_dinov2_lora_v2.py` as the main route:

```bash
python train_dinov2_lora_v2.py --data-root .. --batch-size 4 --amp
```

Avoid relying on:

```text
INR blended into current LoRA
SAM-crop-only branch
label propagation
blind multi-seed averaging
manual top-k tuning without ranking improvement
```

Next algorithmic improvements worth trying:

```text
DINOv2 ViT-L/14 LoRA
multi-crop / MIL pooling
supervised contrastive loss with BCE
carefully filtered LoRA seed ensemble
```

## Summary

The successful path was:

```text
1. Use original images instead of SAM crops.
2. Use DINOv2 ViT-B/14 as the visual foundation model.
3. Move from frozen features to LoRA / PEFT fine-tuning.
4. Improve v1 with an MLP head, label smoothing, and stronger regularization.
5. Use CLS + patch-token mean pooling, augmentation, and flip TTA.
```

Final recorded result:

```text
0.80628 public F1
```
