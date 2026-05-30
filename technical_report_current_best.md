# Technical Report: Current Best Meteorite Stage-2 Pipeline

## Current Best Route

As of 2026-05-30, the methods that are actually working are:

```text
original images
+ DINOv2 ViT-B/14 frozen features
+ multi-center views
+ lightweight feature ensemble
+ strong INR descriptor re-ranking
+ top105 submission
```

The strongest recorded public score is approximately:

```text
0.764
```

from:

```text
submission_original_inr_strong_w040_top105.csv
```

The current best command pattern is:

```bash
python train_feature_ensemble.py \
  --data-root . \
  --backend dinov2_vitb14 \
  --views full,center90,center80,center75,center70,center60 \
  --batch-size 8 \
  --output-dir foundation_checkpoints/ensemble_original

python predict_feature_ensemble.py \
  --checkpoints "foundation_checkpoints/ensemble_original/dinov2_vitb14_*_full-center90-center80-center75-center70-center60.pkl" \
  --data-root . \
  --detail-output ensemble_original_vitb_multicenter_detail.csv \
  --output submission_original_vitb_multicenter.csv \
  --ensemble-method weighted \
  --topk-list 105

python train_inr_classifier.py \
  --data-root . \
  --foundation-backend dinov2_vitb14 \
  --foundation-views full,center90,center80,center75,center70,center60 \
  --classifier logreg \
  --inr-image-size 64 \
  --inr-steps 160 \
  --inr-pixels-per-step 2048 \
  --inr-hidden-dim 64 \
  --inr-hidden-layers 3 \
  --batch-size 8 \
  --output-dir foundation_checkpoints/inr_strong

python predict_inr_classifier.py \
  --checkpoint foundation_checkpoints/inr_strong/dinov2_vitb14_full-center90-center80-center75-center70-center60_plus_inr_logreg.pkl \
  --data-root . \
  --external-prob-file ensemble_original_vitb_multicenter_detail.csv \
  --external-prob-col weighted_prob \
  --inr-weight 0.40 \
  --detail-output original_inr_strong_w040_detail.csv \
  --output submission_original_inr_strong_w040.csv \
  --topk-list 105
```

## What Worked

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
```

Interpretation:

```text
INR is not the primary classifier.
INR is useful for re-ranking boundary samples near top105.
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

The best current model is not solving the task by pure semantic classification alone. It benefits from two complementary signals:

```text
DINOv2 ViT-B/14:
  strong visual representation, object shape, context, global semantics

INR:
  texture, reconstruction dynamics, residual patterns, surface complexity
```

The likely path from `0.764` toward `0.80+` is:

```text
1. Keep original-image dinov2_vitb14 multi-center as the base.
2. Improve INR descriptor strength and boundary reranking.
3. Manually inspect rank 80-130 changed samples.
4. Avoid adding LP unless a future variant proves useful.
```

## References And Literature Support

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
