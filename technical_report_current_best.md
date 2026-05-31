# Technical Report: Current Best Meteorite Stage-2 Pipeline

## Current Best Route

As of 2026-05-31, the strongest working route is:

```text
original images
+ DINOv2 ViT-B/14
+ LoRA / PEFT fine-tuning on the meteorite labels
+ CLS + patch-token mean pooling
+ strong geometric/color augmentation
+ flip TTA at inference
+ top106 submission
```

The strongest recorded public score is approximately:

```text
0.80208
```

from:

```text
submission_dinov2_lora_quick_top106.csv
```

The current best command pattern is:

```bash
python train_dinov2_lora.py \
  --data-root . \
  --model-name dinov2_vitb14 \
  --image-size 518 \
  --pool cls_mean \
  --folds 1 \
  --seed 42 \
  --batch-size 4 \
  --epochs 18 \
  --patience 5 \
  --lr 0.0002 \
  --weight-decay 0.0002 \
  --amp \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-last-n-blocks 4 \
  --tta flip \
  --detail-output dinov2_lora_quick_detail.csv \
  --fold-output dinov2_lora_quick_folds.csv \
  --output submission_dinov2_lora_quick.csv \
  --topk-list 103,104,105,106,107
```

## What Worked

### 0. DINOv2 LoRA / PEFT Fine-Tuning Is the New Best Method

The biggest algorithmic improvement came from adapting DINOv2 to the meteorite domain with LoRA instead of using DINOv2 only as a frozen feature extractor.

Recorded public scores:

```text
original images + frozen dinov2_vitb14 multicenter top105: ~0.74
frozen dinov2_vitb14 + strong INR top105: 0.76439
DINOv2 LoRA quick top105: 0.79581
DINOv2 LoRA quick top106: 0.80208
```

Interpretation:

```text
Frozen DINOv2 already provides a strong visual prior.
LoRA gives the model a small number of trainable parameters to adapt to meteorite-specific surface, shape, and texture cues.
This corrected about four additional top-k positives compared with the strong INR route.
```

The LoRA model uses:

```text
backbone = dinov2_vitb14
image_size = 518
pooling = CLS token + patch-token mean
trainable layers = LoRA adapters in the last 4 ViT blocks
classification head = lightweight binary linear head
augmentation = resized crop, flips, rotation, color jitter
inference = flip TTA
```

Why it helped:

```text
The frozen feature route can only separate images using generic pretrained features.
LoRA performs parameter-efficient domain adaptation, so DINOv2 keeps its general visual representation but shifts its high-level features toward the competition's meteorite/non-meteorite boundary.
Patch-token mean pooling also exposes local texture information, which is likely important for meteorite surfaces.
```

Current recommendation:

```text
Use DINOv2 LoRA top106 as the best submission route.
Keep the frozen DINOv2 + INR route as an interpretable baseline and backup.
```

### 1. Original Images Beat SAM-Cropped Images

The SAM/CLIP crop route was initially useful, but experiments showed that original images match Stage-2 test distribution better.

Recorded public scores:

```text
SAM/CLIP crop + dinov2_vitb14 multicenter top105: ~0.70
original images + dinov2_vitb14 multicenter top105: ~0.74
```

Interpretation:

```text
SAM cropping likely removes useful context, scale, border, background, or global shape cues.
```

Current recommendation:

```text
Use original images as the main route.
Keep SAM crop only as an auxiliary probe, not as the default branch.
```

### 2. DINOv2 ViT-B/14 Multi-Center Features Are the Main Backbone

The strongest base representation is:

```text
backend = dinov2_vitb14
views = full,center90,center80,center75,center70,center60
```

This works better than the smaller default `dinov2` model.

The multi-center views help because the object scale varies. The model sees:

```text
full image context
slightly zoomed center crops
tighter object/texture crops
```

### 3. Strong INR Helps as a Boundary Re-Ranker

The first light INR version had little effect. Stronger INR descriptors improved the public score.

Current useful INR configuration:

```text
image_size = 64
steps = 160
pixels_per_step = 2048
hidden_dim = 64
hidden_layers = 3
classifier = LogisticRegression
blend weight = 0.40
```

Recorded public scores:

```text
original images + dinov2_vitb14 multicenter top105: ~0.74
same baseline + strong INR w0.40 top105: ~0.764
DINOv2 LoRA quick top106: 0.80208
```

Interpretation:

```text
INR is not the primary classifier.
INR is useful for re-ranking boundary samples near top105.
After LoRA fine-tuning, INR is no longer the current best route, but it remains useful evidence that low-level texture/reconstruction cues matter.
```

The INR contributes image-fitting descriptors such as:

```text
initial reconstruction error
final reconstruction error
PSNR
loss trajectory
parameter delta statistics
RGB residual statistics
hidden activation statistics
```

These signals complement DINOv2 by capturing low-level texture, reconstruction difficulty, and surface structure.

## What Did Not Work Well

### Label Propagation

Label Propagation helped the weaker `dinov2` baseline, but hurt or failed to improve the stronger original-image `dinov2_vitb14` route.

Recorded public scores:

```text
original vitb14 multicenter: ~0.74
original vitb14 + LP: ~0.69
original vitb14 + LP + INR: ~0.71
```

Recommendation:

```text
Do not include LP in the default strong pipeline.
Use it only as an exploratory analysis tool.
```

### kNN / Nearest-Neighbor Voting

Train-label kNN voting matched but did not improve the `vitb14` multi-center baseline.

Recorded public score:

```text
vitb14 multicenter + kNN blend: ~0.70 on SAM crop branch
```

Recommendation:

```text
Use nearest-neighbor analysis for inspection and possible duplicate discovery.
Do not rely on it as a primary submission route yet.
```

### SVM INR Classifier

The strong INR SVM classifier underperformed the strong INR LogisticRegression classifier.

Recorded public scores:

```text
strong INR LogisticRegression w0.40: ~0.764
strong INR LinearSVM w0.40: ~0.75
```

Recommendation:

```text
Keep LogisticRegression as the current INR classifier.
```

## About INR Image Size

The current strong INR uses:

```text
--inr-image-size 64
```

This means each image is resized to `64 x 64` only for INR fitting. It does not affect DINOv2 features. DINOv2 still uses its own 518-pixel preprocessing.

### Is 64 x 64 Too Small?

Possibly. It is a compromise:

```text
64 x 64 is large enough to capture broad texture and shape.
64 x 64 is small enough to fit one INR per image quickly.
```

But meteorite identification may depend on fine-grained texture, small surface details, chondrules, fusion crust, or metallic specks. These can be weakened by resizing to 64.

### Recommended Next Tests

Try one stronger INR setting:

```bash
python train_inr_classifier.py \
  --data-root . \
  --foundation-backend dinov2_vitb14 \
  --foundation-views full,center90,center80,center75,center70,center60 \
  --classifier logreg \
  --inr-image-size 80 \
  --inr-steps 260 \
  --inr-pixels-per-step 4096 \
  --inr-hidden-dim 96 \
  --inr-hidden-layers 4 \
  --batch-size 8 \
  --output-dir foundation_checkpoints/inr_xstrong
```

Then predict:

```bash
python predict_inr_classifier.py \
  --checkpoint foundation_checkpoints/inr_xstrong/dinov2_vitb14_full-center90-center80-center75-center70-center60_plus_inr_logreg.pkl \
  --data-root . \
  --external-prob-file ensemble_original_vitb_multicenter_detail.csv \
  --external-prob-col weighted_prob \
  --inr-weight 0.40 \
  --detail-output original_inr_xstrong_w040_detail.csv \
  --output submission_original_inr_xstrong_w040.csv \
  --topk-list 105
```

If the top105 set changes by only a few samples, it is a low-risk candidate. If it changes many samples, inspect the changed ids before submitting.

## Current Working Hypothesis

The best current model is not solving the task by pure semantic classification alone. The experiments suggest a progression:

```text
Frozen DINOv2 ViT-B/14:
  strong generic visual representation, object shape, context, global semantics

INR:
  texture, reconstruction dynamics, residual patterns, surface complexity

DINOv2 LoRA:
  parameter-efficient domain adaptation that moves the pretrained representation toward the meteorite/non-meteorite decision boundary
```

The successful path from `0.764` to `0.80208` was:

```text
1. Keep original images instead of SAM-cropped images.
2. Use DINOv2 ViT-B/14 as the visual foundation model.
3. Move from frozen features to LoRA/PEFT fine-tuning.
4. Preserve local information with CLS + patch-token mean pooling.
5. Use top-k calibration around 105-107, with best current top106.
```

## References And Literature Support

### DINOv2 And Parameter-Efficient Adaptation

Oquab et al., **DINOv2: Learning Robust Visual Features without Supervision**, 2023.

DINOv2 provides the visual foundation used throughout the project. Our results support the paper's core claim that self-supervised ViT features are strong general-purpose representations, and the LoRA branch shows that those features can be further adapted to the meteorite domain.

Link:

- https://arxiv.org/abs/2304.07193

Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**, 2021.

LoRA was introduced for language models, but the same low-rank-adapter idea is now widely used for vision transformers. Our DINOv2 LoRA branch freezes the pretrained backbone and trains small low-rank updates in selected Linear layers, giving a strong accuracy gain without full fine-tuning.

Link:

- https://arxiv.org/abs/2106.09685

The current INR branch is not copied directly from one paper. It is a practical hybrid:

```text
fit a small coordinate MLP/SIREN to each image
extract reconstruction dynamics and fitted-network statistics
use those descriptors as auxiliary classification features
blend them with DINOv2 probabilities
```

The closest literature support comes from the following areas.

### SIREN And INR Signal Fitting

Sitzmann et al., **Implicit Neural Representations with Periodic Activation Functions**, NeurIPS 2020.

This is the foundational SIREN paper. It shows that sinusoidal networks are well suited for representing natural signals such as images, video, audio, and derivatives. Our `TinySiren` follows this coordinate-to-RGB reconstruction idea.

Links:

- https://arxiv.org/abs/2006.09661
- https://github.com/vsitzmann/siren

### Treating Fitted INRs As Data Representations

Dupont et al., **From data to functa: Your data point is a function and you can treat it like one**, ICML 2022.

This is the most relevant conceptual support for our feature strategy. The paper proposes converting each data point into an implicit neural representation, called a *functa*, and then doing downstream learning over those functional representations. It explicitly studies downstream tasks including classification.

Link:

- https://arxiv.org/abs/2201.12204

Bauer et al., **Spatial Functa: Scaling Functa to ImageNet Classification and Generation**, 2023.

This extends the Functa idea to larger image datasets and explicitly targets ImageNet-scale classification/generation by improving the latent representation of neural fields. It supports the broader idea that INR-derived representations can be useful for image classification, although their representation is more sophisticated than our simple descriptor extraction.

Link:

- https://arxiv.org/abs/2302.03130

### Neural Field Hyperparameters Matter For Downstream Classification

Papa et al., **How to Train Neural Field Representations: A Comprehensive Study and Benchmark**, CVPR 2024.

This is highly relevant to our recent finding that stronger INR settings improved the score. The paper studies how neural field architecture, initialization, fitting, and overtraining affect downstream representation quality, including classification. A key lesson is that better reconstruction quality does not always monotonically mean better downstream classification quality, so image size, steps, hidden width, and initialization should be tuned empirically.

Links:

- https://arxiv.org/abs/2312.10531
- https://fit-a-nef.github.io/

### Direct INR Classification Work

Gielisse and van Gemert, **End-to-End Implicit Neural Representations for Classification**, CVPR 2025.

This paper directly addresses INR-based image classification. It represents images via SIREN parameters and trains classification machinery over those representations. It also notes that using INR parameters for classification is nontrivial because fitted neural network parameters have symmetries and can be hard for downstream models to interpret. This helps explain why our simple INR descriptors work best as a boundary re-ranker rather than a standalone classifier.

Links:

- https://arxiv.org/abs/2503.18123
- https://openaccess.thecvf.com/content/CVPR2025/papers/Gielisse_End-to-End_Implicit_Neural_Representations_for_Classification_CVPR_2025_paper.pdf

### Fit Pixels, Get Labels

Vyas et al., **Fit Pixels, Get Labels: Meta-Learned Implicit Networks for Image Segmentation**, MICCAI 2025.

This is the paper that motivated the branch. MetaSeg fits pixels of an unseen image at test time and then decodes labels from a meta-learned INR. Our implementation is much simpler: we do not have pixel-level segmentation labels, so we do not train a segmentation head. Instead, we fit pixels, extract INR fitting statistics, and use them as image-level features for binary classification.

Links:

- https://arxiv.org/abs/2510.04021
- https://papers.miccai.org/miccai-2025/0340-Paper3113.html
- https://kushalvyas.github.io/metaseg.html

### Medical/Domain-Specific Functa Work

VidFuncta and MedFuncta are useful supporting examples showing INR/functa representations being used for downstream medical-imaging tasks.

Relevant examples:

- **VidFuncta: Towards Generalizable Neural Representations for Ultrasound Videos** uses INR/functa-style compact representations for downstream ultrasound tasks including classification.
  - https://arxiv.org/abs/2507.21863
- **MedFuncta: A Unified Framework for Learning Efficient Medical Neural Fields** evaluates neural field representations across medical datasets and downstream tasks.
  - https://proceedings.mlr.press/v315/friedrich26a.html

### How Our Method Differs From The Papers

Our method is deliberately simpler than Functa, Spatial Functa, MetaSeg, or end-to-end INR classification:

```text
No meta-learned INR initialization.
No learned modulation vector.
No transformer over raw SIREN weights.
No pixel-level segmentation decoder.
No end-to-end classification loss through INR fitting.
```

Instead, it uses a practical competition-oriented approximation:

```text
fit one SIREN per image from a shared initialization
extract robust scalar/vector statistics from the fitting process
concatenate/blend these descriptors with DINOv2 predictions
use INR mainly to re-rank top-k boundary samples
```

So the literature supports the core idea that fitted INRs/neural fields can serve as representations for downstream prediction, but our exact descriptor-based boundary reranking is an engineering adaptation for this competition.
