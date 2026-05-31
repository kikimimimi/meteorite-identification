import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import IMAGE_EXTENSIONS, StoneDataset
from foundation_features import load_or_extract_features, parse_number_list, parse_views
from submission_utils import save_threshold_submission, save_topk_submission, split_output_name


def normalize_rows(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def load_train_dataset(root):
    dataset = StoneDataset(root, split="train", transforms=None)
    return list(dataset.ids), np.asarray(dataset.labels, dtype=int), [Path(path) for path in dataset.samples]


def load_test_dataset(root):
    dataset = StoneDataset(root, split="test", transforms=None)
    missing = [image_id for image_id, path in zip(dataset.ids, dataset.samples) if path is None]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images, examples: {missing[:8]}")
    return list(dataset.ids), [Path(path) for path in dataset.samples]


def list_unlabeled_images(root):
    root = Path(root)
    candidates = [root / "test_images" / "test_images", root / "test_images", root]
    image_dir = next((path for path in candidates if path.exists()), None)
    if image_dir is None:
        raise FileNotFoundError(f"Unlabeled root not found: {root}")
    paths = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    ids = [f"{root.name}:{path.name}" for path in paths]
    return ids, paths


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


def weighted_neighbor_score(similarities, labels, sim_power):
    weights = np.maximum(similarities, 0.0) ** float(sim_power)
    denom = weights.sum(axis=1)
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    return (weights * labels.reshape(1, -1)).sum(axis=1) / denom


def build_neighbor_detail(
    test_ids,
    train_ids,
    train_labels,
    test_features,
    train_features,
    nearest_k,
    vote_k,
    sim_power,
):
    sims = normalize_rows(test_features) @ normalize_rows(train_features).T
    order = np.argsort(-sims, axis=1)
    nearest_idx = order[:, 0]
    vote_idx = order[:, :vote_k]
    vote_sims = np.take_along_axis(sims, vote_idx, axis=1)
    vote_labels = train_labels[vote_idx]
    weights = np.maximum(vote_sims, 0.0) ** float(sim_power)
    weight_sum = np.maximum(weights.sum(axis=1), 1e-12)
    nn_score = (weights * vote_labels).sum(axis=1) / weight_sum

    detail = pd.DataFrame(
        {
            "id": test_ids,
            "nn_score": nn_score,
            "nearest_train_id": [train_ids[index] for index in nearest_idx],
            "nearest_train_label": train_labels[nearest_idx],
            "nearest_train_similarity": sims[np.arange(len(test_ids)), nearest_idx],
        }
    )
    for rank in range(min(nearest_k, vote_k)):
        current_idx = vote_idx[:, rank]
        detail[f"neighbor_{rank + 1}_id"] = [train_ids[index] for index in current_idx]
        detail[f"neighbor_{rank + 1}_label"] = train_labels[current_idx]
        detail[f"neighbor_{rank + 1}_similarity"] = vote_sims[:, rank]
    detail["nn_rank"] = detail["nn_score"].rank(method="first", ascending=False).astype(int)
    return detail


def add_stage1_context(detail, test_features, stage1_ids, stage1_features, nearest_k):
    if stage1_features is None or len(stage1_ids) == 0:
        return detail
    sims = normalize_rows(test_features) @ normalize_rows(stage1_features).T
    order = np.argsort(-sims, axis=1)
    nearest_idx = order[:, 0]
    detail["nearest_stage1_id"] = [stage1_ids[index] for index in nearest_idx]
    detail["nearest_stage1_similarity"] = sims[np.arange(len(detail)), nearest_idx]
    for rank in range(min(nearest_k, order.shape[1])):
        current_idx = order[:, rank]
        detail[f"stage1_neighbor_{rank + 1}_id"] = [stage1_ids[index] for index in current_idx]
        detail[f"stage1_neighbor_{rank + 1}_similarity"] = sims[np.arange(len(detail)), current_idx]
    return detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="processed/sam_clip_nofilter")
    parser.add_argument("--backend", default="dinov2_vitb14")
    parser.add_argument("--views", default="full,center90,center80,center75,center70,center60")
    parser.add_argument("--cache-dir", default="features")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vote-k", type=int, default=7)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--sim-power", type=float, default=4.0)
    parser.add_argument("--stage1-unlabeled-root", default="test_images_stage1")
    parser.add_argument("--no-stage1", action="store_true")
    parser.add_argument("--external-prob-file", default="", help="Optional model detail file to compare with kNN.")
    parser.add_argument("--external-prob-col", default="weighted_prob")
    parser.add_argument("--external-weight", type=float, default=0.0, help="Blend: external_weight*external + (1-external_weight)*nn_score.")
    parser.add_argument("--detail-output", default="nearest_neighbor_detail.csv")
    parser.add_argument("--output", default="submission_nearest_neighbor.csv")
    parser.add_argument("--topk-list", default="105")
    parser.add_argument("--threshold-list", default="")
    args = parser.parse_args()

    views = parse_views(args.views)
    template = pd.read_csv(os.path.join(args.data_root, "sample_submission.csv"))
    train_ids, train_labels, train_paths = load_train_dataset(args.data_root)
    test_ids, test_paths = load_test_dataset(args.data_root)
    if template["id"].astype(str).tolist() != [str(item) for item in test_ids]:
        raise RuntimeError("Test ids do not match sample_submission.csv order.")

    train_features = load_or_extract_features(
        train_paths, args.backend, args.cache_dir, args.batch_size, prefix="nn_train", views=views
    )
    test_features = load_or_extract_features(
        test_paths, args.backend, args.cache_dir, args.batch_size, prefix="nn_test", views=views
    )
    detail = build_neighbor_detail(
        test_ids,
        train_ids,
        train_labels,
        test_features,
        train_features,
        args.nearest_k,
        args.vote_k,
        args.sim_power,
    )

    if not args.no_stage1:
        stage1_ids, stage1_paths = list_unlabeled_images(args.stage1_unlabeled_root)
        stage1_features = load_or_extract_features(
            stage1_paths,
            args.backend,
            args.cache_dir,
            args.batch_size,
            prefix="nn_stage1_unlabeled",
            views=views,
        )
        detail = add_stage1_context(detail, test_features, stage1_ids, stage1_features, args.nearest_k)
        print(f"Added stage1 nearest-neighbor context: {len(stage1_ids)} images")

    final_score = detail["nn_score"].to_numpy(dtype=np.float64)
    external_score = load_external_scores(args.external_prob_file, args.external_prob_col, test_ids)
    if external_score is not None:
        external_weight = float(args.external_weight)
        if external_weight < 0 or external_weight > 1:
            raise ValueError("--external-weight must be in [0, 1].")
        detail[args.external_prob_col] = external_score
        final_score = external_weight * external_score + (1.0 - external_weight) * final_score
        print(
            f"Blending nearest-neighbor score with {args.external_prob_file}:{args.external_prob_col} | "
            f"external_weight={external_weight:.2f}"
        )
    detail["final_score"] = final_score
    detail["final_rank"] = detail["final_score"].rank(method="first", ascending=False).astype(int)
    detail.sort_values("final_rank").to_csv(args.detail_output, index=False)
    print(f"Saved nearest-neighbor detail: {args.detail_output}")

    stem, ext = split_output_name(args.output)
    save_threshold_submission(template, test_ids, final_score, 0.5, args.output)
    for threshold in parse_number_list(args.threshold_list, float):
        save_threshold_submission(template, test_ids, final_score, threshold, f"{stem}_th{threshold:.2f}{ext}")
    for topk in parse_number_list(args.topk_list, int):
        save_topk_submission(template, test_ids, final_score, topk, f"{stem}_top{topk}{ext}")


if __name__ == "__main__":
    main()
