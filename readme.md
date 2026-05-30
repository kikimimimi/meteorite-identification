# Meteorite Identification Pipeline

This project contains two main training routes:

- `train.py` / `predict.py`: conventional image model baseline.
- `train_foundation.py` / `predict_foundation.py`: frozen foundation features, currently the recommended route.

The current recommended pipeline is:

1. Use SAM to propose object masks on train images.
2. Use CLIP to rerank SAM candidates and select the most stone-like crop.
3. Copy test images unchanged, because the current test set is already mostly clean stone-only images.
4. Extract multi-view DINOv2 features.
5. Train a lightweight classifier.
6. Generate submission CSV files.

## Dataset Layout

The project root should contain:

```text
train_labels.csv
sample_submission.csv
train_images/
test_images/
```

Nested image directories such as `train_images/train_images` and `test_images/test_images` are also supported by `dataset.py`.

## 1. SAM + CLIP Preprocessing

Run this on the server from the project root, for example:

```bash
cd /data/meteorite-identification
```

Then generate the CLIP-selected SAM dataset:

```bash
python sam_preprocess.py \
  --checkpoint /data/models/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --output-root processed/sam_clip_nofilter \
  --mask-ranker clip \
  --mask-ranker-model /data/models/clip-vit-base-patch32 \
  --mask-ranker-max-candidates 30 \
  --mask-ranker-weight 3.0 \
  --debug-dir debug/sam_clip_nofilter \
  --debug-limit 300
```

By default, this only runs SAM + CLIP on the train images. Test images are copied unchanged into `processed/sam_clip_nofilter`.

Important outputs:

```text
processed/sam_clip_nofilter/train_images/
processed/sam_clip_nofilter/test_images/
processed/sam_clip_nofilter/train_labels.csv
processed/sam_clip_nofilter/sample_submission.csv
debug/sam_clip_nofilter/train/
```

Before training, inspect the debug images. The red box should usually cover the stone-like object rather than labels, rulers, hands, or tiny fragments.

If you explicitly want to process test images too, add:

```bash
--process-test
```

For the current second-stage test set, do not use `--process-test` unless you have checked that SAM improves the test images.

## 2. Train DINOv2 Multi-View Classifier

Train the recommended small DINOv2 model:

```bash
python train_foundation.py \
  --data-root processed/sam_clip_nofilter \
  --backend dinov2 \
  --views full,center75,center60 \
  --batch-size 16
```

Expected checkpoint:

```text
foundation_checkpoints/dinov2_sam_logreg_full-center75-center60.pkl
```

The progress bar denominator is the number of batches, not the number of images. For 5098 images and batch size 16, `319/319` is normal.

## 3. Predict and Generate Submission

Generate the main submission:

```bash
python predict_foundation.py \
  --checkpoint foundation_checkpoints/dinov2_sam_logreg_full-center75-center60.pkl \
  --data-root processed/sam_clip_nofilter \
  --tta \
  --output submission_samclip_multiview.csv \
  --prob-output probs_samclip_multiview.csv
```

This creates:

```text
submission_samclip_multiview.csv
probs_samclip_multiview.csv
```

`submission_samclip_multiview.csv` is the file to submit.

## 4. Optional Threshold and Top-K Submissions

To generate several candidate submissions in one run:

```bash
python predict_foundation.py \
  --checkpoint foundation_checkpoints/dinov2_sam_logreg_full-center75-center60.pkl \
  --data-root processed/sam_clip_nofilter \
  --tta \
  --output submission_samclip_multiview.csv \
  --prob-output probs_samclip_multiview.csv \
  --topk-list 70,80,90,100,110 \
  --threshold-list 0.35,0.40,0.45,0.50,0.55
```

This will create files like:

```text
submission_samclip_multiview_top80.csv
submission_samclip_multiview_th0.45.csv
```

Use these only as controlled experiments. The validation threshold is saved in the checkpoint, but the online test distribution may differ.

## 5. Feature Ensemble Route

The Stage-2 recommended route remains frozen foundation features plus lightweight classifiers. Train several classifiers on the same multi-view feature cache:

```bash
python train_feature_ensemble.py \
  --data-root processed/sam_clip_nofilter \
  --backend dinov2 \
  --views full,center90,center75,center60 \
  --batch-size 16
```

This writes checkpoints and validation probabilities under:

```text
foundation_checkpoints/ensemble/
```

Optional backends:

```bash
python train_feature_ensemble.py \
  --data-root processed/sam_clip_nofilter \
  --backend dinov2_vitb14 \
  --views full,center75,center60 \
  --batch-size 8

python train_feature_ensemble.py \
  --data-root processed/sam_clip_nofilter \
  --backend clip \
  --views full,center75,center60 \
  --batch-size 16
```

`dinov3` is reserved as a backend name, but it currently raises a clear error until a local model path/loading recipe is added.

Generate ensemble probabilities and F1-aware submissions:

```bash
python predict_feature_ensemble.py \
  --checkpoints "foundation_checkpoints/ensemble/dinov2_*_full-center90-center75-center60.pkl" \
  --data-root processed/sam_clip_nofilter \
  --detail-output ensemble_prediction_detail.csv \
  --output submission_ensemble.csv \
  --ensemble-method weighted \
  --topk-list 105,108,110 \
  --threshold-list 0.35,0.38,0.40,0.42,0.44,0.50
```

This creates files such as:

```text
submission_ensemble_top105.csv
submission_ensemble_top108.csv
submission_ensemble_top110.csv
submission_ensemble_th0.40.csv
```

Every generated submission is checked against `sample_submission.csv`: exact columns, exact id order, no missing values, and 0/1 labels only.

## 6. Historical Weighted Voting

Use previous leaderboard submissions as weighted 0/1 voters:

```bash
python make_weighted_voting_submission.py \
  --sample-submission processed/sam_clip_nofilter/sample_submission.csv \
  --submissions 0.6385.csv,0.638.csv,0.635.csv \
  --scores 0.6385,0.638,0.635 \
  --min-score 0.60 \
  --score-power 2 \
  --output submission_weighted_voting.csv \
  --topk-list 105,108,110 \
  --threshold-list 0.40,0.45,0.50,0.55,0.60
```

This writes `weighted_voting_detail.csv` with each historical label, `weighted_vote_percent`, `vote_count`, and `rank`, plus top-k and threshold submissions.

## 7. Label Propagation With Stage-1 Unlabeled Test

If train/test distribution shift is strong, use DINOv2 features to build a kNN graph over labeled train images, current test images, and optional stage-1 unlabeled test images:

```bash
python label_propagation.py \
  --data-root processed/sam_clip_nofilter \
  --stage1-unlabeled-root test_images_stage1 \
  --backend dinov2 \
  --views full,center75,center60 \
  --n-neighbors 15 \
  --alpha 0.2 \
  --batch-size 16 \
  --output submission_label_propagation.csv \
  --detail-output label_propagation_detail.csv \
  --topk-list 90,100,105,108,110,120 \
  --threshold-list 0.35,0.38,0.40,0.42,0.44,0.50
```

Blend label propagation with an existing model ensemble detail file:

```bash
python label_propagation.py \
  --data-root processed/sam_clip_nofilter \
  --stage1-unlabeled-root test_images_stage1 \
  --backend dinov2 \
  --views full,center75,center60 \
  --external-prob-file ensemble_prediction_detail.csv \
  --external-prob-col weighted_prob \
  --lp-weight 0.45 \
  --output submission_lp_blend.csv \
  --detail-output label_propagation_blend_detail.csv \
  --topk-list 90,100,105,108,110,120
```

Use `--no-stage1` to test whether stage-1 unlabeled images help or hurt.

## 8. INR Descriptor Branch

The `INR` branch adds a lightweight idea inspired by *Fit Pixels, Get Labels*: fit a small SIREN/INR to each image's pixels and use the fitting dynamics plus hidden statistics as extra descriptors. The current implementation does not require segmentation masks and does not replace DINOv2; it concatenates INR descriptors with frozen foundation features.

Train a DINOv2 + INR classifier:

```bash
python train_inr_classifier.py \
  --data-root processed/sam_clip_nofilter \
  --foundation-backend dinov2 \
  --foundation-views full,center75,center60 \
  --classifier logreg \
  --inr-image-size 48 \
  --inr-steps 80 \
  --inr-pixels-per-step 1024 \
  --batch-size 16
```

Predict and generate top-k submissions:

```bash
python predict_inr_classifier.py \
  --checkpoint foundation_checkpoints/inr/dinov2_full-center75-center60_plus_inr_logreg.pkl \
  --data-root processed/sam_clip_nofilter \
  --detail-output inr_prediction_detail.csv \
  --output submission_inr.csv \
  --topk-list 90,100,105,108,110,120 \
  --threshold-list 0.35,0.38,0.40,0.42,0.44,0.50
```

Blend INR probabilities with an existing ensemble detail file:

```bash
python predict_inr_classifier.py \
  --checkpoint foundation_checkpoints/inr/dinov2_full-center75-center60_plus_inr_logreg.pkl \
  --data-root processed/sam_clip_nofilter \
  --external-prob-file ensemble_prediction_detail.csv \
  --external-prob-col weighted_prob \
  --inr-weight 0.25 \
  --detail-output inr_blend_prediction_detail.csv \
  --output submission_inr_blend.csv \
  --topk-list 90,100,105,108,110,120
```

For a faster sanity check, reduce `--inr-steps` to `30`. For stronger descriptors, try `--inr-image-size 64 --inr-steps 120`, but it will be slower.

## 9. Nearest-Neighbor And Boundary Inspection

When top-k is already near the best positive count, use feature-space retrieval to find ranking errors and possible near-duplicates:

```bash
python nearest_neighbor_analysis.py \
  --data-root processed/sam_clip_nofilter \
  --backend dinov2_vitb14 \
  --views full,center90,center80,center75,center70,center60 \
  --batch-size 8 \
  --vote-k 7 \
  --sim-power 4 \
  --external-prob-file ensemble_vitb_multicenter_detail.csv \
  --external-prob-col weighted_prob \
  --external-weight 0.70 \
  --detail-output nearest_neighbor_vitb_detail.csv \
  --output submission_nearest_neighbor_vitb.csv \
  --topk-list 105
```

This writes nearest train neighbors, train labels, cosine similarities, optional stage-1 nearest neighbors, and a kNN-derived score.

Export rank-boundary images for manual review:

```bash
python export_rank_images.py \
  --detail-file ensemble_vitb_multicenter_detail.csv \
  --data-root processed/sam_clip_nofilter \
  --output-dir inspect_vitb_rank_80_130 \
  --rank-column rank \
  --score-column weighted_prob \
  --rank-from 80 \
  --rank-to 130
```

The most useful manual review region is usually around the chosen top-k boundary, for example rank 80-130 when submitting top105.

## 10. Original + SAM-Crop Two-Branch Blend

Train and predict the original-image branch with the same strong `dinov2_vitb14` multi-center setup:

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
```

Blend the original-image branch with the SAM/CLIP crop branch:

```bash
python blend_prediction_details.py \
  --sample-submission processed/sam_clip_nofilter/sample_submission.csv \
  --detail-files ensemble_vitb_multicenter_detail.csv,ensemble_original_vitb_multicenter_detail.csv \
  --score-cols weighted_prob,weighted_prob \
  --weights 0.65,0.35 \
  --aliases samcrop,original \
  --detail-output ensemble_sam_original_blend_detail.csv \
  --output submission_sam_original_blend.csv \
  --topk-list 105
```

If the original branch is strong, also try `--weights 0.50,0.50`. If it is weaker but complementary, keep the original branch at `0.20-0.35`.

## 11. Optional Larger DINOv2 Model

If GPU memory and time allow, try the larger DINOv2 base model:

```bash
python train_foundation.py \
  --data-root processed/sam_clip_nofilter \
  --backend dinov2_vitb14 \
  --views full,center75,center60 \
  --batch-size 8
```

Predict with:

```bash
python predict_foundation.py \
  --checkpoint foundation_checkpoints/dinov2_vitb14_sam_logreg_full-center75-center60.pkl \
  --data-root processed/sam_clip_nofilter \
  --tta \
  --output submission_samclip_vitb_multiview.csv \
  --prob-output probs_samclip_vitb_multiview.csv
```

## 12. CLIP Model Files

`sam_preprocess.py --mask-ranker clip` needs a local CLIP model folder if the server cannot access Hugging Face.

Recommended local path:

```text
/data/models/clip-vit-base-patch32
```

The folder should contain files such as:

```text
config.json
model.safetensors
preprocessor_config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
special_tokens_map.json
```

Then pass:

```bash
--mask-ranker-model /data/models/clip-vit-base-patch32
```

Do not pass only `model.safetensors`; `transformers` needs the full folder.

## 13. Recommended Submission Order

Try these first:

```text
submission_samclip_multiview.csv
submission_samclip_vitb_multiview.csv
```

If the larger model is too slow, prioritize:

```text
submission_samclip_multiview.csv
```

## Notes

- SAM + CLIP is used for train image cleanup and object selection.
- Test images are copied unchanged by default.
- DINOv2 is frozen; training is fast because only a lightweight classifier is trained.
- Multi-view features concatenate several DINOv2 views, for example `full + center75 + center60`.
- If SAM debug quality is poor, compare against the pure original-image route:

```bash
python train_foundation.py \
  --data-root . \
  --backend dinov2 \
  --views full,center75,center60 \
  --batch-size 16
```
