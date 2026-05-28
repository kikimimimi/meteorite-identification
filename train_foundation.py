import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import StoneDataset
from foundation_features import load_or_extract_features, parse_views


def find_best_threshold(labels, probs):
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(labels, (probs > threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold), float(best_f1)


def normalize_rows(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def load_train_dataset(root):
    dataset = StoneDataset(root, split="train", transforms=None)
    labels = np.array(dataset.labels)
    ids = list(dataset.ids)
    paths = [Path(path) for path in dataset.samples]
    return dataset, ids, labels, paths


def fit_classifier(x_train, y_train, sample_weight=None):
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
        ),
    )
    classifier.fit(x_train, y_train, logisticregression__sample_weight=sample_weight)
    return classifier


def build_hard_negative_weights(x_train, y_train, x_val, y_val, val_probs, threshold, hard_weight):
    weights = np.ones(len(y_train), dtype=np.float32)
    false_positive_features = x_val[(y_val == 0) & (val_probs > threshold)]
    if len(false_positive_features) == 0:
        print("No validation false positives found; hard-negative mining skipped.")
        return weights

    train_neg = y_train == 0
    x_neg_norm = normalize_rows(x_train[train_neg])
    fp_norm = normalize_rows(false_positive_features)
    max_similarity = np.maximum(0.0, x_neg_norm @ fp_norm.T).max(axis=1)
    weights[train_neg] += hard_weight * max_similarity
    print(
        f"Hard-negative mining: {len(false_positive_features)} validation false positives, "
        f"negative weight range {weights[train_neg].min():.2f}-{weights[train_neg].max():.2f}"
    )
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="processed/sam")
    parser.add_argument("--fusion-root", default=None, help="Optional second image root, e.g. . for original images.")
    parser.add_argument("--backend", default="dinov2", help="dinov2, dinov2_vits14, dinov2_vitb14, or clip")
    parser.add_argument(
        "--views",
        default="full",
        help="Comma-separated DINOv2 views, e.g. full,center75,center60",
    )
    parser.add_argument("--output-dir", default="foundation_checkpoints")
    parser.add_argument("--cache-dir", default="features")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-weight", type=float, default=2.5)
    args = parser.parse_args()

    if not Path(args.data_root).exists():
        raise FileNotFoundError(
            f"{args.data_root} does not exist. Run sam_preprocess.py first, "
            "or pass --data-root . to test features on original images."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    views = parse_views(args.views)
    _, ids, labels, paths = load_train_dataset(args.data_root)
    features = load_or_extract_features(
        paths,
        args.backend,
        args.cache_dir,
        args.batch_size,
        prefix="train_primary",
        views=views,
    )

    if args.fusion_root is not None:
        _, fusion_ids, fusion_labels, fusion_paths = load_train_dataset(args.fusion_root)
        if fusion_ids != ids:
            raise RuntimeError("Primary data-root and fusion-root train ids do not match.")
        if not np.array_equal(fusion_labels, labels):
            raise RuntimeError("Primary data-root and fusion-root labels do not match.")
        fusion_features = load_or_extract_features(
            fusion_paths,
            args.backend,
            args.cache_dir,
            args.batch_size,
            prefix="train_fusion",
            views=views,
        )
        features = np.concatenate([features, fusion_features], axis=1)
        print(f"Using feature fusion: {args.data_root} + {args.fusion_root} -> dim={features.shape[1]}")

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_size,
        stratify=labels,
        random_state=args.seed,
    )

    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]

    initial_clf = fit_classifier(x_train, y_train)
    initial_probs = initial_clf.predict_proba(x_val)[:, 1]
    initial_threshold, initial_f1 = find_best_threshold(y_val, initial_probs)
    print(f"Initial val F1: {initial_f1:.4f} @ {initial_threshold:.2f}")

    weights = build_hard_negative_weights(
        x_train,
        y_train,
        x_val,
        y_val,
        initial_probs,
        initial_threshold,
        hard_weight=args.hard_weight,
    )
    final_clf = fit_classifier(x_train, y_train, sample_weight=weights)
    final_probs = final_clf.predict_proba(x_val)[:, 1]
    final_threshold, final_f1 = find_best_threshold(y_val, final_probs)
    print(f"Final val F1: {final_f1:.4f} @ {final_threshold:.2f}")

    suffix = "fusion_logreg" if args.fusion_root is not None else "sam_logreg"
    if views != ["full"]:
        suffix = f"{suffix}_{'-'.join(views)}"
    output_path = Path(args.output_dir) / f"{args.backend}_{suffix}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump({
            "classifier": final_clf,
            "threshold": final_threshold,
            "val_f1": final_f1,
            "initial_threshold": initial_threshold,
            "initial_val_f1": initial_f1,
            "backend": args.backend,
            "views": views,
            "data_root": args.data_root,
            "fusion_root": args.fusion_root,
            "feature_dim": int(features.shape[1]),
        }, f)
    print(f"Saved foundation classifier: {output_path}")


if __name__ == "__main__":
    main()
