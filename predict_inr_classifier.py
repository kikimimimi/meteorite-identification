import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import StoneDataset
from foundation_features import load_or_extract_features, parse_number_list, parse_views
from inr_features import extract_inr_features
from submission_utils import save_threshold_submission, save_topk_submission, split_output_name


def load_test_dataset(root):
    dataset = StoneDataset(root, split="test", transforms=None)
    missing = [image_id for image_id, path in zip(dataset.ids, dataset.samples) if path is None]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images, examples: {missing[:8]}")
    return list(dataset.ids), [Path(path) for path in dataset.samples]


def load_external_scores(path, column, ids):
    if not path:
        return None
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{path} must contain an id column.")
    if column not in df.columns:
        raise ValueError(f"{path} does not contain score column {column}.")
    scores = pd.DataFrame({"id": [str(item) for item in ids]}).merge(
        df[["id", column]].assign(id=df["id"].astype(str)),
        on="id",
        how="left",
    )[column]
    if scores.isna().any():
        missing = [ids[index] for index in np.where(scores.isna().to_numpy())[0][:8]]
        raise RuntimeError(f"Missing external scores for ids: {missing}")
    return scores.to_numpy(dtype=np.float64)


def build_feature_matrix(checkpoint, paths, cache_dir, batch_size):
    inr_config = checkpoint["inr_config"]
    inr_features, _ = extract_inr_features(
        paths,
        cache_dir,
        prefix="inr_test",
        image_size=inr_config["image_size"],
        steps=inr_config["steps"],
        pixels_per_step=inr_config["pixels_per_step"],
        hidden_dim=inr_config["hidden_dim"],
        hidden_layers=inr_config["hidden_layers"],
        omega_0=inr_config["omega_0"],
        lr=inr_config["lr"],
        seed=inr_config["seed"],
    )
    matrices = [inr_features]
    if not checkpoint.get("no_foundation", False):
        backend = checkpoint["foundation_backend"]
        views = parse_views(checkpoint.get("foundation_views", "full"))
        foundation_features = load_or_extract_features(
            paths,
            backend,
            cache_dir,
            batch_size,
            prefix="inr_test_foundation",
            views=views,
        )
        matrices.insert(0, foundation_features)
    features = np.concatenate(matrices, axis=1).astype(np.float32)
    expected_dim = checkpoint.get("feature_dim")
    if expected_dim is not None and int(expected_dim) != features.shape[1]:
        raise RuntimeError(f"Checkpoint expects feature_dim={expected_dim}, got {features.shape[1]}.")
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="foundation_checkpoints/inr/dinov2_full-center75-center60_plus_inr_logreg.pkl")
    parser.add_argument("--data-root", default=None, help="If omitted, uses checkpoint data_root.")
    parser.add_argument("--cache-dir", default="features")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--detail-output", default="inr_prediction_detail.csv")
    parser.add_argument("--output", default="submission_inr.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-list", default="0.35,0.38,0.40,0.42,0.44,0.50")
    parser.add_argument("--topk-list", default="80,90,100,105,108,110,120,128")
    parser.add_argument("--external-prob-file", default="", help="Optional detail csv, e.g. ensemble_prediction_detail.csv.")
    parser.add_argument("--external-prob-col", default="weighted_prob")
    parser.add_argument("--inr-weight", type=float, default=1.0, help="Final score = inr_weight*inr + (1-inr_weight)*external.")
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)

    data_root = args.data_root or checkpoint.get("data_root", "processed/sam_clip_nofilter")
    template = pd.read_csv(os.path.join(data_root, "sample_submission.csv"))
    ids, paths = load_test_dataset(data_root)
    if template["id"].astype(str).tolist() != [str(item) for item in ids]:
        raise RuntimeError("Test ids do not match sample_submission.csv order.")

    features = build_feature_matrix(checkpoint, paths, args.cache_dir, args.batch_size)
    inr_prob = checkpoint["classifier"].predict_proba(features)[:, 1]
    final_score = inr_prob
    detail = pd.DataFrame({"id": ids, "inr_prob": inr_prob})

    external_prob = load_external_scores(args.external_prob_file, args.external_prob_col, ids)
    if external_prob is not None:
        inr_weight = float(args.inr_weight)
        if inr_weight < 0 or inr_weight > 1:
            raise ValueError("--inr-weight must be in [0, 1].")
        detail[args.external_prob_col] = external_prob
        final_score = inr_weight * inr_prob + (1.0 - inr_weight) * external_prob
        print(f"Blending INR with {args.external_prob_file}:{args.external_prob_col} | inr_weight={inr_weight:.2f}")

    detail["final_score"] = final_score
    detail["rank"] = detail["final_score"].rank(method="first", ascending=False).astype(int)
    detail.sort_values("rank").to_csv(args.detail_output, index=False)
    print(f"Saved detail: {args.detail_output}")

    threshold = float(args.threshold) if args.threshold is not None else float(checkpoint.get("threshold", 0.5))
    stem, ext = split_output_name(args.output)
    save_threshold_submission(template, ids, final_score, threshold, args.output)
    for value in parse_number_list(args.threshold_list, float):
        save_threshold_submission(template, ids, final_score, value, f"{stem}_th{value:.2f}{ext}")
    for value in parse_number_list(args.topk_list, int):
        save_topk_submission(template, ids, final_score, value, f"{stem}_top{value}{ext}")


if __name__ == "__main__":
    main()
