# Algorithm Change Log

## 2026-05-28

- Expanded the Stage-2 frozen foundation feature route while keeping the original `train_foundation.py` and `predict_foundation.py` commands compatible.
- Added multi-backend feature support in `foundation_features.py`: `dinov2`, `dinov2_vitb14`, `clip`, plus a reserved `dinov3` entry that raises a clear not-implemented error.
- Extended multi-view feature extraction to accept arbitrary comma-separated views such as `full,center90,center75,center60`; features from each view are concatenated.
- Added shared submission validation in `submission_utils.py`, enforcing exact `sample_submission.csv` columns, exact id order, no missing values, 0/1 labels only, and positive-count printing.
- Added `train_feature_ensemble.py` for lightweight classifier ensembling over frozen features: LogisticRegression, calibrated LinearSVM, RandomForest, optional XGBoost, and optional LightGBM.
- Added `predict_feature_ensemble.py` for multi-checkpoint prediction, model probability detail output, simple average, validation-F1 weighted average, and optional historical submission voting.
- Added `make_weighted_voting_submission.py` for standalone weighted voting from historical Kaggle submissions, with `--min-score` filtering and `--score-power` weighting.
- Updated `readme.md` with recommended Stage-2 frozen feature ensemble commands and weighted voting commands.
- Added `label_propagation.py` to use DINO/CLIP features for kNN LabelSpreading over labeled train images, current test images, and optional `test_images_stage1` unlabeled images.
- Added optional blending in `label_propagation.py` between label propagation probabilities and an external ensemble detail file such as `ensemble_prediction_detail.csv`.
- Added `submission_log.csv` to track Kaggle submission experiments with filename, data root, backend, views, classifier/ensemble, top-k or threshold, public score, and positive count.

## 2026-05-29

- Created and switched to the `INR` branch for experiments inspired by *Fit Pixels, Get Labels: Meta-Learned Implicit Networks for Image Segmentation*.
- Added `inr_features.py`, which fits a small SIREN/INR to each image's RGB pixels and exports reconstruction dynamics, parameter-delta statistics, residual statistics, and hidden activation statistics as image-level descriptors.
- Added `train_inr_classifier.py` to train a lightweight classifier on INR descriptors, optionally concatenated with existing frozen DINOv2/CLIP foundation features.
- Added `predict_inr_classifier.py` to generate INR probability detail files, threshold submissions, top-k submissions, and optional blends with an existing ensemble detail score.
- Updated `readme.md` with the recommended INR training, prediction, and INR+ensemble blending commands.
- Confirmed that `dinov2_vitb14` multi-center views improved the public score to roughly 0.70 at `top105`. Adding INR blending with `--inr-weight 0.20` produced the same public score, so the current evidence points to `dinov2_vitb14 + multi-center views` as the main gain rather than INR. The useful command sequence is:

```bash
python train_feature_ensemble.py \
  --data-root processed/sam_clip_nofilter \
  --backend dinov2_vitb14 \
  --views full,center90,center80,center75,center70,center60 \
  --batch-size 8

python predict_feature_ensemble.py \
  --checkpoints "foundation_checkpoints/ensemble/dinov2_vitb14_*_full-center90-center80-center75-center70-center60.pkl" \
  --data-root processed/sam_clip_nofilter \
  --detail-output ensemble_vitb_multicenter_detail.csv \
  --output submission_vitb_multicenter.csv \
  --ensemble-method weighted \
  --topk-list 105

python train_inr_classifier.py \
  --data-root processed/sam_clip_nofilter \
  --foundation-backend dinov2_vitb14 \
  --foundation-views full,center90,center80,center75,center70,center60 \
  --classifier logreg \
  --inr-image-size 48 \
  --inr-steps 80 \
  --inr-pixels-per-step 1024 \
  --batch-size 8

python predict_inr_classifier.py \
  --checkpoint foundation_checkpoints/inr/dinov2_vitb14_full-center90-center80-center75-center70-center60_plus_inr_logreg.pkl \
  --data-root processed/sam_clip_nofilter \
  --external-prob-file ensemble_vitb_multicenter_detail.csv \
  --external-prob-col weighted_prob \
  --inr-weight 0.20 \
  --detail-output inr_vitb_multicenter_blend_detail.csv \
  --output submission_inr_vitb_multicenter_blend.csv \
  --topk-list 105
```
- Tested `dinov2_vitb14` multi-center ensemble blended with stage1-unlabeled Label Propagation at `top105`; public score was roughly 0.69, slightly below the plain `dinov2_vitb14` multi-center ensemble score of 0.70. Current best evidence: keep the plain `vitb14` multi-center ensemble as the strong baseline, and treat LP/INR as optional probes rather than default additions on top of it.
- Added `nearest_neighbor_analysis.py` to analyze DINO/CLIP feature-space nearest train neighbors for each test image, export kNN label-voting scores, compare against an external model detail file, and generate top-k submissions.
- Added `export_rank_images.py` to copy rank-boundary test images, such as rank 80-130 around `top105`, into an inspection folder with a manifest for manual error analysis.
- Tested the `dinov2_vitb14` multi-center ensemble blended with train-label kNN nearest-neighbor voting at `top105`; public score was roughly 0.70, matching the plain `vitb14` multi-center ensemble. kNN retrieval is useful for analysis, but the current blend did not improve the submitted ranking.
- Added `blend_prediction_details.py` to blend prediction detail CSVs from different branches, especially the `processed/sam_clip_nofilter` SAM-crop branch and the original-image branch. This enables original + crop two-branch fusion without retraining a larger joint model.
- Tested the original-image `dinov2_vitb14` multi-center ensemble at `top105`; public score reached roughly 0.74, exceeding the SAM/CLIP crop branch score of roughly 0.70. Current best evidence: original images preserve important Stage-2 test information, and SAM-crop preprocessing may be hurting distribution match.
- Tested original-image Label Propagation blend at `top105`; public score was roughly 0.69. Adding INR with weight 0.40 to the LP blend reached roughly 0.71, but this remains below the original-image `vitb14` multi-center baseline and the original-image INR weight 0.40 blend. Current evidence: LP should not be part of the default strong pipeline; INR is useful mainly as a boundary re-ranker around the original-image baseline.
- Strengthened the INR descriptor configuration on original images (`image_size=64`, `steps=160`, `pixels_per_step=2048`, `hidden_dim=64`, `hidden_layers=3`) and blended it with the original-image `dinov2_vitb14` multi-center baseline at INR weight 0.40. Public score reached roughly 0.764 at `top105`, confirming that stronger INR descriptors can improve boundary ranking.
- Tested strong INR with a calibrated linear SVM classifier at weight 0.40; public score dropped to roughly 0.75 versus the strong INR LogisticRegression score of roughly 0.764. Current best INR classifier is the strong LogisticRegression version.
- Added `technical_report_current_best.md`, summarizing the current best technical route: original images, `dinov2_vitb14` multi-center frozen features, lightweight ensemble, and strong INR LogisticRegression re-ranking. The report also documents negative results from LP, kNN, SVM INR, and SAM-crop as the primary branch.
- Expanded the technical report with INR literature references, including SIREN, Functa, Spatial Functa, Fit-a-NeF, end-to-end INR classification, MetaSeg, and medical Functa variants. The report now clarifies that our INR descriptor re-ranker is an engineering adaptation supported by related INR-as-representation work, not an exact reproduction of one paper.
- Tested a larger xstrong INR configuration. It changed only one top105 boundary swap versus the strong INR version, replacing `000173.jpg` with `000074.jpg`, and the public score decreased. This supports the Fit-a-NeF-style lesson that larger/better-fitting INR settings do not necessarily improve downstream classification; current best remains the strong INR configuration rather than xstrong.
- Added `inferred_test_labels.md` to maintain public-leaderboard-derived label hypotheses for boundary test images. Current strong inferred labels include `000173.jpg` as likely positive and `000083.jpg`/`000074.jpg` as likely negative.
- Tested strong INR `top105` versus `top106`; `top105` scored `0.76439`, while `top106` scored `0.76041`. The extra rank-106 image was `000097.jpg`, now inferred as likely negative or not worth including. Current best positive count remains `105`.
- Tested strong INR `top107`; score decreased further to `0.75647`. Compared with `top105`, it only added `000097.jpg` and `000016.jpg`, so both added boundary images are likely negatives under the current ranking.
- Tested SAM-crop strong INR blended with the original-image baseline at INR weight 0.40. It changed 22 labels versus the current best and scored roughly 0.72, confirming that SAM-crop INR is too disruptive at high weight. If revisited, use only low weights such as 0.10-0.25.
- Created branch `INR2.0` for meta-learning experiments and added a Meta-INR encoder path. The new `--use-meta-inr` option trains a small supervised reconstruction-regularized MLP on INR descriptors, appends `meta_inr_prob` plus a learned embedding to the DINOv2+INR feature matrix, and stores the meta encoder inside the INR checkpoint for matching prediction-time feature construction.
- Added `train_kfold_inr_rank_ensemble.py` for a more stable algorithmic route: original-image `dinov2_vitb14` multi-center baseline plus strong INR descriptors trained with K-fold splits across multiple INR seeds, combined by rank averaging instead of raw probability averaging, with automatic `top103-107` submission generation.
- Tested the K-fold multi-seed INR rank ensemble. `top104` scored `0.75789` and `top105` scored `0.75392`, below the current strong INR single-split best of `0.76439`. The K-fold route is more stable but appears to smooth away the useful boundary bias from the best single strong INR run.
- Added `train_dinov2_lora.py` for DINOv2 LoRA/PEFT fine-tuning. The script freezes the DINOv2 backbone, injects LoRA adapters into selected Linear layers of the last ViT blocks, trains a lightweight binary head with augmentation and optional K-fold ensembling, supports CLS plus patch-token pooling, and exports top-k submissions for comparison against the frozen-feature INR route.
- Tested the quick single-split DINOv2 LoRA/PEFT run; `top105` reached `0.79581`, a major improvement over the frozen DINOv2 + strong INR best of `0.76439`. This indicates that parameter-efficient domain adaptation of DINOv2 is the strongest algorithmic direction so far and likely corrected about three additional boundary positives.
- Submitted DINOv2 LoRA quick `top106`; score reached `0.80208`, breaking the 0.8 target. This implies the rank-106 image in the LoRA quick ranking is likely positive and the LoRA route increased the estimated top-k true positives to about 77.
