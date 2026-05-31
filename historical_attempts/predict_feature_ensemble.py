import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import StoneDataset
from foundation_features import (
    build_extractor,
    extract_with_tta,
    load_or_extract_features,
    parse_number_list,
    parse_views,
)
from submission_utils import save_threshold_submission, save_topk_submission, split_output_name


def dataset_paths_or_fail(root):
    dataset = StoneDataset(root, split="test", transforms=None)
    missing = [image_id for image_id, path in zip(dataset.ids, dataset.samples) if path is None]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images, examples: {missing[:8]}")
    return dataset.ids, [Path(path) for path in dataset.samples]


def parse_checkpoint_paths(value):
    paths = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        matches = sorted(glob.glob(item))
        paths.extend(matches or [item])
    if not paths:
        raise ValueError("No checkpoints provided.")
    return paths


def load_history_votes(template, submission_files, scores):
    if not submission_files:
        return None
    files = [item.strip() for item in submission_files.split(",") if item.strip()]
    score_values = parse_number_list(scores, float)
    if len(files) != len(score_values):
        raise ValueError("--history-submissions and --history-scores must have the same length.")

    template_ids = template["id"].astype(str).tolist()
    labels = []
    for path in files:
        df = pd.read_csv(path)
        if list(df.columns) != list(template.columns):
            raise ValueError(f"{path} columns do not match sample_submission.csv.")
        if df["id"].astype(str).tolist() != template_ids:
            raise ValueError(f"{path} id order does not match sample_submission.csv.")
        labels.append(df["label"].astype(int).to_numpy())
    weights = np.asarray(score_values, dtype=np.float64)
    label_matrix = np.vstack(labels).T
    vote_percent = (label_matrix * weights).sum(axis=1) / weights.sum()
    return vote_percent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint paths or glob patterns.")
    parser.add_argument("--data-root", default="processed/sam_clip_nofilter")
    parser.add_argument("--cache-dir", default="features")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--detail-output", default="ensemble_prediction_detail.csv")
    parser.add_argument("--output", default="submission_ensemble.csv")
    parser.add_argument("--ensemble-method", default="weighted", choices=["average", "weighted", "history-vote"])
    parser.add_argument("--threshold-list", default="0.35,0.38,0.40,0.42,0.44,0.50")
    parser.add_argument("--topk-list", default="80,90,100,105,108,110,120,128")
    parser.add_argument("--history-submissions", default="")
    parser.add_argument("--history-scores", default="")
    args = parser.parse_args()

    template = pd.read_csv(os.path.join(args.data_root, "sample_submission.csv"))
    ids, paths = dataset_paths_or_fail(args.data_root)
    if template["id"].astype(str).tolist() != [str(item) for item in ids]:
        raise RuntimeError("Test ids do not match sample_submission.csv order.")

    checkpoints = []
    for checkpoint_path in parse_checkpoint_paths(args.checkpoints):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        checkpoint["path"] = checkpoint_path
        checkpoints.append(checkpoint)

    feature_cache = {}
    prob_columns = []
    detail = pd.DataFrame({"id": ids})
    for index, checkpoint in enumerate(checkpoints, start=1):
        backend = checkpoint["backend"]
        views = parse_views(checkpoint.get("views", "full"))
        key = (backend, tuple(views))
        if key not in feature_cache:
            if args.tta:
                extractor = build_extractor(backend)
                features = extract_with_tta(
                    extractor,
                    paths,
                    backend,
                    args.batch_size,
                    use_tta=True,
                    views=views,
                    desc_prefix="Ensemble predict",
                )
            else:
                features = load_or_extract_features(
                    paths,
                    backend,
                    args.cache_dir,
                    args.batch_size,
                    prefix="test_ensemble",
                    views=views,
                )
            feature_cache[key] = features
        features = feature_cache[key]
        expected_dim = checkpoint.get("feature_dim")
        if expected_dim is not None and int(expected_dim) != features.shape[1]:
            raise RuntimeError(
                f"{checkpoint['path']} expects feature_dim={expected_dim}, got {features.shape[1]} "
                f"for backend={backend} views={views}."
            )

        name = checkpoint.get("classifier_name") or Path(checkpoint["path"]).stem
        col = f"prob_model_{index}"
        probs = checkpoint["classifier"].predict_proba(features)[:, 1]
        detail[col] = probs
        prob_columns.append(col)
        print(
            f"{col} | backend={backend} | views={','.join(views)} | "
            f"val_f1={float(checkpoint.get('val_f1', 0.0)):.4f}"
        )

    prob_matrix = detail[prob_columns].to_numpy()
    detail["avg_prob"] = prob_matrix.mean(axis=1)
    val_f1_weights = np.asarray([max(float(item.get("val_f1", 0.0)), 1e-6) for item in checkpoints])
    detail["weighted_prob"] = (prob_matrix * val_f1_weights).sum(axis=1) / val_f1_weights.sum()

    history_vote = load_history_votes(template, args.history_submissions, args.history_scores)
    if history_vote is not None:
        detail["history_vote_percent"] = history_vote

    if args.ensemble_method == "average":
        final_score = detail["avg_prob"].to_numpy()
    elif args.ensemble_method == "weighted":
        final_score = detail["weighted_prob"].to_numpy()
    else:
        if "history_vote_percent" not in detail:
            raise ValueError("--ensemble-method history-vote requires --history-submissions and --history-scores.")
        final_score = detail["history_vote_percent"].to_numpy()

    detail["final_score"] = final_score
    detail["rank"] = detail["final_score"].rank(method="first", ascending=False).astype(int)
    detail.sort_values("rank").to_csv(args.detail_output, index=False)
    print(f"Saved detail: {args.detail_output}")

    stem, ext = split_output_name(args.output)
    default_threshold = 0.5 if args.ensemble_method == "history-vote" else float(np.median([c.get("threshold", 0.5) for c in checkpoints]))
    save_threshold_submission(template, ids, final_score, default_threshold, args.output)
    for threshold in parse_number_list(args.threshold_list, float):
        save_threshold_submission(template, ids, final_score, threshold, f"{stem}_th{threshold:.2f}{ext}")
    for topk in parse_number_list(args.topk_list, int):
        save_topk_submission(template, ids, final_score, topk, f"{stem}_top{topk}{ext}")


if __name__ == "__main__":
    main()
