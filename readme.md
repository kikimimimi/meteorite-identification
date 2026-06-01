# Meteorite Identification

This repository contains the final working route and historical experiments for the meteorite binary classification task.

The current best recorded result is produced by the **DINOv2 LoRA v2** pipeline:

```text
original images
+ DINOv2 ViT-B/14
+ LoRA / PEFT adaptation
+ CLS + patch-token mean pooling
+ deeper MLP classification head
+ label smoothing
+ regularized training
+ flip TTA
= 0.80628 public F1
```

## Repository Layout

The root directory keeps the current runnable LoRA pipeline:

```text
dataset.py
submission_utils.py
train_dinov2_lora.py
train_dinov2_lora_v2.py
```

`train_dinov2_lora_v2.py` is now the best recorded route. `train_dinov2_lora.py` is the earlier LoRA v1 baseline that reached `0.80208`.

Older experiments are stored in:

```text
historical_attempts/
```

That folder contains previous frozen DINOv2, INR, SAM, label propagation, kNN, weighted voting, and meta-INR scripts. They are kept for reproducibility and presentation discussion, but they are not part of the final code path.

## Dataset Layout

The dataset directory should contain:

```text
train_labels.csv
sample_submission.csv
train_images/
test_images/
```

Nested image directories such as `train_images/train_images` and `test_images/test_images` are also supported by `dataset.py`.

## Best Recorded Command

The current best v2 run was launched with:

```bash
python train_dinov2_lora_v2.py --data-root .. --batch-size 4 --amp
```

This command uses the v2 defaults:

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
topk_list = 85,90,95,100,105
```

Path note: `--data-root ..` means the dataset is expected in the parent directory of the current shell location. If you run from the repository root and the data files are also in the repository root, use `--data-root .` instead.

The command writes:

```text
dinov2_lora_v2_detail.csv
dinov2_lora_v2_folds.csv
submission_dinov2_lora_v2_top85.csv
submission_dinov2_lora_v2_top90.csv
submission_dinov2_lora_v2_top95.csv
submission_dinov2_lora_v2_top100.csv
submission_dinov2_lora_v2_top105.csv
```

Best recorded public F1 from the v2 run:

```text
0.80628
```

## Main Method

`train_dinov2_lora_v2.py` does the following:

1. Loads train/test images through `StoneDataset`.
2. Loads a pretrained DINOv2 backbone from `torch.hub`.
3. Freezes the pretrained DINOv2 backbone.
4. Injects LoRA adapters into selected Linear layers of the last ViT blocks.
5. Uses `CLS + mean(patch tokens)` as the image representation.
6. Trains a deeper MLP binary head with BatchNorm and Dropout.
7. Uses label-smoothed BCE to reduce overfitting.
8. Applies random crop, flips, rotation, and color jitter during training.
9. Uses flip TTA at inference.
10. Generates top-k submission files.

The important v2 improvements over v1 are:

```text
deeper MLP head instead of a single linear head
label smoothing
stronger regularization
more conservative learning rate and weight decay
5-fold prediction averaging by default
```

## What Is Not Used in the Final Code

The final v2 code does **not** use INR.

INR/SIREN descriptors were useful during exploration and improved an older frozen-feature route, but they did not improve the final LoRA route:

```text
frozen DINOv2 + strong INR: 0.76439
LoRA v1 top106: 0.80208
LoRA + INR blend top106: 0.79166
LoRA v2: 0.80628
```

Therefore INR is documented as a tried-but-unused technique, not a final method.

## Historical Attempts

The following approaches were tried and moved to `historical_attempts/`:

```text
SAM + CLIP preprocessing
frozen DINOv2 multi-view classifiers
feature ensemble and weighted voting
INR / SIREN descriptor branch
Meta-INR
K-fold multi-seed INR rank average
label propagation
kNN / nearest-neighbor analysis
```

These experiments are useful for the presentation because they explain why the final route became LoRA v2 rather than segmentation, label propagation, or INR.

## Documents

Useful project notes:

```text
technical_report_current_best.md
key_technical_explanation_cn.md
algorithm_change_log.md
submission_log.csv
inferred_test_labels.md
```

`key_technical_explanation_cn.md` is the presentation-oriented Chinese explanation. `technical_report_current_best.md` records the current best route and experiment interpretation.
