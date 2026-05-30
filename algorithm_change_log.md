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
