import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading

from dataset import IMAGE_EXTENSIONS, StoneDataset
from foundation_features import load_or_extract_features, parse_number_list, parse_views
from submission_utils import save_threshold_submission, save_topk_submission, split_output_name


def normalize_rows(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def load_train_dataset(root):
    dataset = StoneDataset(root, split="train", transforms=None)
    return list(dataset.ids), np.array(dataset.labels), [Path(path) for path in dataset.samples]


def load_test_dataset(root):
    dataset = StoneDataset(root, split="test", transforms=None)
    missing = [image_id for image_id, path in zip(dataset.ids, dataset.samples) if path is None]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images, examples: {missing[:8]}")
    return list(dataset.ids), [Path(path) for path in dataset.samples]


def list_unlabeled_images(root):
    root = Path(root)
    candidates = [
        root / "test_images" / "test_images",
        root / "test_images",
        root,
    ]
    image_dir = next((path for path in candidates if path.exists()), None)
    if image_dir is None:
        raise FileNotFoundError(f"Unlabeled root not found: {root}")

    paths = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No unlabeled images found under {image_dir}")
    ids = [f"{root.name}:{path.name}" for path in paths]
    return ids, paths


def find_best_threshold(labels, probs):
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.95, 0.005):
        f1 = f1_score(labels, (probs > threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold), float(best_f1)


def semi_supervised_val_score(features, labels, n_neighbors, alpha, max_iter, val_size, seed):
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=val_size, stratify=labels, random_state=seed)
    y = np.full(len(labels), -1, dtype=int)
    y[train_idx] = labels[train_idx]
    model = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter, n_jobs=-1)
    model.fit(features, y)
    positive_index = int(np.where(model.classes_ == 1)[0][0])
    probs = model.label_distributions_[val_idx, positive_index]
    threshold, val_f1 = find_best_threshold(labels[val_idx], probs)
    return threshold, val_f1, int((probs > threshold).sum())


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="processed/sam_clip_nofilter")
    parser.add_argument("--stage1-unlabeled-root", default="test_images_stage1")
    parser.add_argument("--no-stage1", action="store_true", help="Use only train + current test in the graph.")
    parser.add_argument("--backend", default="dinov2")
    parser.add_argument("--views", default="full,center75,center60")
    parser.add_argument("--cache-dir", default="features")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-prob-file", default="", help="Optional detail csv, e.g. ensemble_prediction_detail.csv.")
    parser.add_argument("--external-prob-col", default="weighted_prob")
    parser.add_argument("--lp-weight", type=float, default=1.0, help="Final score = lp_weight*lp + (1-lp_weight)*external.")
    parser.add_argument("--detail-output", default="label_propagation_detail.csv")
    parser.add_argument("--output", default="submission_label_propagation.csv")
    parser.add_argument("--threshold-list", default="0.35,0.38,0.40,0.42,0.44,0.50")
    parser.add_argument("--topk-list", default="80,90,100,105,108,110,120,128")
    args = parser.parse_args()

    views = parse_views(args.views)
    template = pd.read_csv(os.path.join(args.data_root, "sample_submission.csv"))

    train_ids, train_labels, train_paths = load_train_dataset(args.data_root)
    test_ids, test_paths = load_test_dataset(args.data_root)
    if template["id"].astype(str).tolist() != [str(item) for item in test_ids]:
        raise RuntimeError("Test ids do not match sample_submission.csv order.")

    train_features = load_or_extract_features(
        train_paths, args.backend, args.cache_dir, args.batch_size, prefix="lp_train", views=views
    )
    test_features = load_or_extract_features(
        test_paths, args.backend, args.cache_dir, args.batch_size, prefix="lp_test", views=views
    )

    extra_ids, extra_paths, extra_features = [], [], np.empty((0, train_features.shape[1]), dtype=train_features.dtype)
    if not args.no_stage1:
        extra_ids, extra_paths = list_unlabeled_images(args.stage1_unlabeled_root)
        extra_features = load_or_extract_features(
            extra_paths, args.backend, args.cache_dir, args.batch_size, prefix="lp_stage1_unlabeled", views=views
        )
        print(f"Using stage1 unlabeled images: {len(extra_paths)}")

    graph_features = normalize_rows(np.concatenate([train_features, test_features, extra_features], axis=0))
    graph_labels = np.concatenate(
        [
            train_labels.astype(int),
            np.full(len(test_paths) + len(extra_paths), -1, dtype=int),
        ]
    )

    val_threshold, val_f1, val_positive_count = semi_supervised_val_score(
        normalize_rows(train_features),
        train_labels,
        args.n_neighbors,
        args.alpha,
        args.max_iter,
        args.val_size,
        args.seed,
    )
    print(
        f"LabelSpreading val_f1={val_f1:.4f} | best_threshold={val_threshold:.3f} | "
        f"val_positive_count={val_positive_count}"
    )

    model = LabelSpreading(
        kernel="knn",
        n_neighbors=args.n_neighbors,
        alpha=args.alpha,
        max_iter=args.max_iter,
        n_jobs=-1,
    )
    model.fit(graph_features, graph_labels)
    positive_index = int(np.where(model.classes_ == 1)[0][0])
    test_start = len(train_paths)
    test_end = test_start + len(test_paths)
    lp_prob = model.label_distributions_[test_start:test_end, positive_index]

    external_prob = load_external_scores(args.external_prob_file, args.external_prob_col, test_ids)
    final_score = lp_prob
    detail = pd.DataFrame({"id": test_ids, "lp_prob": lp_prob})
    if external_prob is not None:
        lp_weight = float(args.lp_weight)
        if lp_weight < 0 or lp_weight > 1:
            raise ValueError("--lp-weight must be in [0, 1].")
        detail[args.external_prob_col] = external_prob
        final_score = lp_weight * lp_prob + (1.0 - lp_weight) * external_prob
        print(f"Blending LP with {args.external_prob_file}:{args.external_prob_col} | lp_weight={lp_weight:.2f}")

    detail["final_score"] = final_score
    detail["rank"] = detail["final_score"].rank(method="first", ascending=False).astype(int)
    detail.sort_values("rank").to_csv(args.detail_output, index=False)
    print(f"Saved detail: {args.detail_output}")

    stem, ext = split_output_name(args.output)
    save_threshold_submission(template, test_ids, final_score, val_threshold, args.output)
    for threshold in parse_number_list(args.threshold_list, float):
        save_threshold_submission(template, test_ids, final_score, threshold, f"{stem}_th{threshold:.2f}{ext}")
    for topk in parse_number_list(args.topk_list, int):
        save_topk_submission(template, test_ids, final_score, topk, f"{stem}_top{topk}{ext}")


if __name__ == "__main__":
    main()
