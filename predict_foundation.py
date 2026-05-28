import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import StoneDataset
from foundation_features import build_extractor, extract_with_tta, logit, parse_number_list, parse_views, sigmoid
from submission_utils import save_threshold_submission, save_topk_submission, split_output_name


def dataset_paths_or_fail(root):
    dataset = StoneDataset(root, split="test", transforms=None)
    missing = [image_id for image_id, path in zip(dataset.ids, dataset.samples) if path is None]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images, examples: {missing[:8]}")
    return dataset.ids, [Path(path) for path in dataset.samples]


def extract_features_for_checkpoint(extractor, backend, data_root, fusion_root, batch_size, use_tta, views=None):
    ids, paths = dataset_paths_or_fail(data_root)
    features = extract_with_tta(extractor, paths, backend, batch_size, use_tta, views=views)

    if fusion_root is not None:
        fusion_ids, fusion_paths = dataset_paths_or_fail(fusion_root)
        if fusion_ids != ids:
            raise RuntimeError("Primary data-root and fusion-root test ids do not match.")
        fusion_features = extract_with_tta(extractor, fusion_paths, backend, batch_size, use_tta, views=views)
        features = np.concatenate([features, fusion_features], axis=1)
        print(f"Using feature fusion: {data_root} + {fusion_root} -> dim={features.shape[1]}")

    return ids, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="foundation_checkpoints/dinov2_sam_logreg.pkl")
    parser.add_argument("--data-root", default="processed/sam")
    parser.add_argument("--fusion-root", default=None, help="Optional second image root. If omitted, uses checkpoint metadata.")
    parser.add_argument("--original-root", default=".")
    parser.add_argument("--output", default="submission_foundation_sam.csv")
    parser.add_argument("--guided-output", default="submission_foundation_guided.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--views", default=None, help="Override checkpoint DINOv2 views, e.g. full,center75,center60")
    parser.add_argument("--guidance-gamma", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold.")
    parser.add_argument("--threshold-list", default="", help="Comma-separated thresholds, e.g. 0.3,0.4,0.5")
    parser.add_argument("--topk-list", default="", help="Comma-separated positive counts, e.g. 60,70,80,90")
    parser.add_argument("--prob-output", default="prediction_probs.csv")
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)
    classifier = checkpoint["classifier"]
    threshold = float(checkpoint["threshold"])
    if args.threshold is not None:
        threshold = float(args.threshold)
    backend = checkpoint["backend"]
    views = parse_views(args.views if args.views is not None else checkpoint.get("views", "full"))
    fusion_root = args.fusion_root
    if fusion_root is None:
        fusion_root = checkpoint.get("fusion_root")

    template = pd.read_csv(os.path.join(args.data_root, "sample_submission.csv"))

    extractor = build_extractor(backend)
    ids, features = extract_features_for_checkpoint(
        extractor,
        backend,
        args.data_root,
        fusion_root,
        args.batch_size,
        args.tta,
        views=views,
    )
    probs = classifier.predict_proba(features)[:, 1]
    save_threshold_submission(template, ids, probs, threshold, args.output)

    prob_df = pd.DataFrame({"id": ids, "prob": probs}).sort_values("prob", ascending=False)
    prob_df.to_csv(args.prob_output, index=False)
    print(
        f"Saved probabilities: {args.prob_output} | "
        f"range={float(np.min(probs)):.4f}-{float(np.max(probs)):.4f} | "
        f"threshold={threshold:.3f}"
    )

    stem, ext = split_output_name(args.output)
    for value in parse_number_list(args.threshold_list, float):
        save_threshold_submission(template, ids, probs, value, f"{stem}_th{value:.2f}{ext}")
    for value in parse_number_list(args.topk_list, int):
        save_topk_submission(template, ids, probs, value, f"{stem}_top{value}{ext}")

    if args.guidance_gamma > 0:
        original_ids, original_paths = dataset_paths_or_fail(args.original_root)
        if original_ids != ids:
            raise RuntimeError("SAM-processed ids and original ids do not match.")
        original_features = extract_with_tta(extractor, original_paths, backend, args.batch_size, False, views=views)
        if original_features.shape[1] != features.shape[1]:
            raise RuntimeError("--guidance-gamma is not compatible with the current fused feature shape.")
        original_probs = classifier.predict_proba(original_features)[:, 1]
        guided_probs = sigmoid(logit(probs) + args.guidance_gamma * (logit(probs) - logit(original_probs)))
        save_threshold_submission(template, ids, guided_probs, threshold, args.guided_output)


if __name__ == "__main__":
    main()
