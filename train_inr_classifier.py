import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from dataset import StoneDataset
from foundation_features import load_or_extract_features, parse_views
from inr_features import extract_inr_features
from meta_inr import meta_feature_names, train_meta_inr_encoder, transform_meta_inr_features


def find_best_threshold(labels, probs):
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.95, 0.005):
        f1 = f1_score(labels, (probs > threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold), float(best_f1)


def load_train_dataset(root):
    dataset = StoneDataset(root, split="train", transforms=None)
    labels = np.array(dataset.labels)
    ids = list(dataset.ids)
    paths = [Path(path) for path in dataset.samples]
    return ids, labels, paths


def build_classifier(name, seed):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0, solver="lbfgs"),
        )
    if name == "linear_svm":
        linear_svc = LinearSVC(class_weight="balanced", C=0.5, random_state=seed, max_iter=8000)
        try:
            calibrated = CalibratedClassifierCV(linear_svc, method="sigmoid", cv=3)
        except TypeError:
            calibrated = CalibratedClassifierCV(base_estimator=linear_svc, method="sigmoid", cv=3)
        return make_pipeline(StandardScaler(), calibrated)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier: {name}")


def build_feature_matrix(args, paths):
    inr_features, inr_names = extract_inr_features(
        paths,
        args.cache_dir,
        prefix="inr_train",
        image_size=args.inr_image_size,
        steps=args.inr_steps,
        pixels_per_step=args.inr_pixels_per_step,
        hidden_dim=args.inr_hidden_dim,
        hidden_layers=args.inr_hidden_layers,
        omega_0=args.inr_omega_0,
        lr=args.inr_lr,
        seed=args.seed,
    )
    matrices = [inr_features]
    feature_parts = [f"INR({inr_features.shape[1]})"]

    views = parse_views(args.foundation_views)
    if not args.no_foundation:
        foundation_features = load_or_extract_features(
            paths,
            args.foundation_backend,
            args.cache_dir,
            args.batch_size,
            prefix="inr_train_foundation",
            views=views,
        )
        matrices.insert(0, foundation_features)
        feature_parts.insert(0, f"{args.foundation_backend}:{'-'.join(views)}({foundation_features.shape[1]})")

    features = np.concatenate(matrices, axis=1).astype(np.float32)
    print(f"Feature matrix: {' + '.join(feature_parts)} -> dim={features.shape[1]}")
    return features, inr_features, inr_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="processed/sam_clip_nofilter")
    parser.add_argument("--output-dir", default="foundation_checkpoints/inr")
    parser.add_argument("--cache-dir", default="features")
    parser.add_argument("--classifier", default="logreg", choices=["logreg", "linear_svm", "random_forest"])
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-foundation", action="store_true", help="Train on INR descriptors only.")
    parser.add_argument("--foundation-backend", default="dinov2")
    parser.add_argument("--foundation-views", default="full,center75,center60")
    parser.add_argument("--inr-image-size", type=int, default=48)
    parser.add_argument("--inr-steps", type=int, default=80)
    parser.add_argument("--inr-pixels-per-step", type=int, default=1024)
    parser.add_argument("--inr-hidden-dim", type=int, default=32)
    parser.add_argument("--inr-hidden-layers", type=int, default=2)
    parser.add_argument("--inr-omega-0", type=float, default=30.0)
    parser.add_argument("--inr-lr", type=float, default=1e-3)
    parser.add_argument("--use-meta-inr", action="store_true", help="Learn a nonlinear meta encoder from INR descriptors.")
    parser.add_argument("--meta-hidden-dim", type=int, default=256)
    parser.add_argument("--meta-embedding-dim", type=int, default=64)
    parser.add_argument("--meta-epochs", type=int, default=250)
    parser.add_argument("--meta-batch-size", type=int, default=64)
    parser.add_argument("--meta-lr", type=float, default=1e-3)
    parser.add_argument("--meta-weight-decay", type=float, default=1e-4)
    parser.add_argument("--meta-recon-weight", type=float, default=0.05)
    parser.add_argument("--meta-dropout", type=float, default=0.1)
    parser.add_argument("--meta-patience", type=int, default=35)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ids, labels, paths = load_train_dataset(args.data_root)
    features, inr_features, inr_names = build_feature_matrix(args, paths)

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_size,
        stratify=labels,
        random_state=args.seed,
    )
    meta_checkpoint = None
    meta_names = []
    if args.use_meta_inr:
        meta_config = {
            "hidden_dim": args.meta_hidden_dim,
            "embedding_dim": args.meta_embedding_dim,
            "epochs": args.meta_epochs,
            "batch_size": args.meta_batch_size,
            "lr": args.meta_lr,
            "weight_decay": args.meta_weight_decay,
            "recon_weight": args.meta_recon_weight,
            "dropout": args.meta_dropout,
            "patience": args.meta_patience,
        }
        meta_checkpoint = train_meta_inr_encoder(
            inr_features[train_idx],
            labels[train_idx],
            inr_features[val_idx],
            labels[val_idx],
            meta_config,
            seed=args.seed,
        )
        meta_features = transform_meta_inr_features(meta_checkpoint, inr_features)
        meta_names = meta_feature_names(args.meta_embedding_dim)
        features = np.concatenate([features, meta_features], axis=1).astype(np.float32)
        print(f"Added Meta-INR features: dim={meta_features.shape[1]} -> total_dim={features.shape[1]}")

    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]

    classifier = build_classifier(args.classifier, args.seed)
    classifier.fit(x_train, y_train)
    val_probs = classifier.predict_proba(x_val)[:, 1]
    threshold, val_f1 = find_best_threshold(y_val, val_probs)
    val_positive_count = int((val_probs > threshold).sum())
    print(
        f"INR classifier={args.classifier} | val_f1={val_f1:.4f} | "
        f"best_threshold={threshold:.3f} | val_positive_count={val_positive_count}"
    )

    foundation_views = parse_views(args.foundation_views)
    feature_tag = "inr_only" if args.no_foundation else f"{args.foundation_backend}_{'-'.join(foundation_views)}_plus_inr"
    if args.use_meta_inr:
        feature_tag = f"{feature_tag}_meta"
    checkpoint_path = Path(args.output_dir) / f"{feature_tag}_{args.classifier}.pkl"
    checkpoint = {
        "classifier": classifier,
        "classifier_name": args.classifier,
        "threshold": threshold,
        "val_f1": val_f1,
        "data_root": args.data_root,
        "feature_dim": int(features.shape[1]),
        "no_foundation": bool(args.no_foundation),
        "foundation_backend": args.foundation_backend,
        "foundation_views": foundation_views,
        "use_meta_inr": bool(args.use_meta_inr),
        "meta_inr_checkpoint": meta_checkpoint,
        "meta_inr_feature_names": meta_names,
        "inr_config": {
            "image_size": args.inr_image_size,
            "steps": args.inr_steps,
            "pixels_per_step": args.inr_pixels_per_step,
            "hidden_dim": args.inr_hidden_dim,
            "hidden_layers": args.inr_hidden_layers,
            "omega_0": args.inr_omega_0,
            "lr": args.inr_lr,
            "seed": args.seed,
        },
        "inr_feature_names": inr_names,
        "val_ids": [ids[index] for index in val_idx],
        "val_labels": y_val,
        "val_probs": val_probs,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    val_prob_path = checkpoint_path.with_name(checkpoint_path.stem + "_val_probs.csv")
    pd.DataFrame({"id": checkpoint["val_ids"], "label": y_val, "prob": val_probs}).to_csv(val_prob_path, index=False)
    print(f"Saved INR checkpoint: {checkpoint_path}")
    print(f"Saved validation probabilities: {val_prob_path}")


if __name__ == "__main__":
    main()
