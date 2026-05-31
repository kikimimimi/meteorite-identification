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


def build_classifiers(seed):
    linear_svc = LinearSVC(class_weight="balanced", C=0.5, random_state=seed, max_iter=8000)
    try:
        calibrated_svm = CalibratedClassifierCV(linear_svc, method="sigmoid", cv=3)
    except TypeError:
        calibrated_svm = CalibratedClassifierCV(base_estimator=linear_svc, method="sigmoid", cv=3)

    classifiers = [
        (
            "logreg",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0, solver="lbfgs"),
            ),
        ),
        (
            "linear_svm_calibrated",
            make_pipeline(
                StandardScaler(),
                calibrated_svm,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ]

    try:
        from xgboost import XGBClassifier

        classifiers.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        )
    except Exception as exc:
        print(f"Skipping xgboost: {exc}")

    try:
        from lightgbm import LGBMClassifier

        classifiers.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=600,
                    max_depth=-1,
                    learning_rate=0.03,
                    class_weight="balanced",
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        )
    except Exception as exc:
        print(f"Skipping lightgbm: {exc}")

    return classifiers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="processed/sam_clip_nofilter")
    parser.add_argument("--backend", default="dinov2", help="dinov2, dinov2_vitb14, clip, or reserved dinov3")
    parser.add_argument("--views", default="full,center75,center60")
    parser.add_argument("--output-dir", default="foundation_checkpoints/ensemble")
    parser.add_argument("--cache-dir", default="features")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    views = parse_views(args.views)
    ids, labels, paths = load_train_dataset(args.data_root)
    features = load_or_extract_features(
        paths,
        args.backend,
        args.cache_dir,
        args.batch_size,
        prefix="train_ensemble",
        views=views,
    )

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_size,
        stratify=labels,
        random_state=args.seed,
    )
    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]
    val_ids = [ids[index] for index in val_idx]

    summary_rows = []
    for name, classifier in build_classifiers(args.seed):
        print(f"Training {name}...")
        classifier.fit(x_train, y_train)
        probs = classifier.predict_proba(x_val)[:, 1]
        threshold, val_f1 = find_best_threshold(y_val, probs)
        positive_count = int((probs > threshold).sum())

        checkpoint_name = f"{args.backend}_{name}_{'-'.join(views)}.pkl"
        checkpoint_path = Path(args.output_dir) / checkpoint_name
        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {
                    "classifier": classifier,
                    "classifier_name": name,
                    "threshold": threshold,
                    "val_f1": val_f1,
                    "backend": args.backend,
                    "views": views,
                    "data_root": args.data_root,
                    "feature_dim": int(features.shape[1]),
                    "val_ids": val_ids,
                    "val_labels": y_val,
                    "val_probs": probs,
                },
                f,
            )

        val_prob_path = Path(args.output_dir) / f"{args.backend}_{name}_{'-'.join(views)}_val_probs.csv"
        pd.DataFrame({"id": val_ids, "label": y_val, "prob": probs}).to_csv(val_prob_path, index=False)
        row = {
            "classifier": name,
            "backend": args.backend,
            "views": ",".join(views),
            "val_f1": val_f1,
            "best_threshold": threshold,
            "val_positive_count": positive_count,
            "checkpoint": str(checkpoint_path),
        }
        summary_rows.append(row)
        print(
            f"{name} | backend={args.backend} | views={','.join(views)} | "
            f"val_f1={val_f1:.4f} | best_threshold={threshold:.3f} | "
            f"val_positive_count={positive_count}"
        )

    summary = pd.DataFrame(summary_rows).sort_values("val_f1", ascending=False)
    summary_path = Path(args.output_dir) / f"{args.backend}_{'-'.join(views)}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved ensemble summary: {summary_path}")


if __name__ == "__main__":
    main()
