import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from dataset import StoneDataset
from foundation_features import load_or_extract_features, parse_number_list, parse_views
from inr_features import extract_inr_features
from submission_utils import save_topk_submission, split_output_name


def load_train_dataset(root):
    dataset = StoneDataset(root, split="train", transforms=None)
    return list(dataset.ids), np.asarray(dataset.labels, dtype=int), [Path(path) for path in dataset.samples]


def load_test_dataset(root):
    dataset = StoneDataset(root, split="test", transforms=None)
    missing = [image_id for image_id, path in zip(dataset.ids, dataset.samples) if path is None]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images, examples: {missing[:8]}")
    return list(dataset.ids), [Path(path) for path in dataset.samples]


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
            n_estimators=700,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier: {name}")


def find_best_threshold(labels, probs):
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.95, 0.005):
        f1 = f1_score(labels, (probs > threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold), float(best_f1)


def load_external_scores(path, score_col, ids):
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{path} must contain an id column.")
    if score_col not in df.columns:
        raise ValueError(f"{path} does not contain score column {score_col}.")
    merged = pd.DataFrame({"id": [str(item) for item in ids]}).merge(
        df[["id", score_col]].assign(id=df["id"].astype(str)),
        on="id",
        how="left",
    )
    if merged[score_col].isna().any():
        missing = merged.loc[merged[score_col].isna(), "id"].head(8).tolist()
        raise RuntimeError(f"Missing external scores from {path}: {missing}")
    return merged[score_col].to_numpy(dtype=np.float64)


def rank_score(scores):
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (len(scores) - ranks + 1.0) / len(scores)


def extract_seed_features(args, train_paths, test_paths, seed, foundation_train, foundation_test):
    config = {
        "image_size": args.inr_image_size,
        "steps": args.inr_steps,
        "pixels_per_step": args.inr_pixels_per_step,
        "hidden_dim": args.inr_hidden_dim,
        "hidden_layers": args.inr_hidden_layers,
        "omega_0": args.inr_omega_0,
        "lr": args.inr_lr,
        "seed": seed,
    }
    train_inr, _ = extract_inr_features(train_paths, args.cache_dir, prefix=f"kfold_inr_train_seed{seed}", **config)
    test_inr, _ = extract_inr_features(test_paths, args.cache_dir, prefix=f"kfold_inr_test_seed{seed}", **config)
    train_features = np.concatenate([foundation_train, train_inr], axis=1).astype(np.float32)
    test_features = np.concatenate([foundation_test, test_inr], axis=1).astype(np.float32)
    return train_features, test_features


def predict_kfold(args, features, labels, test_features, seed):
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(labels), dtype=np.float64)
    test_fold_probs = []
    fold_rows = []
    base_classifier = build_classifier(args.classifier, seed)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(features, labels), start=1):
        classifier = clone(base_classifier)
        classifier.fit(features[train_idx], labels[train_idx])
        val_probs = classifier.predict_proba(features[val_idx])[:, 1]
        test_probs = classifier.predict_proba(test_features)[:, 1]
        oof[val_idx] = val_probs
        threshold, fold_f1 = find_best_threshold(labels[val_idx], val_probs)
        fold_rows.append(
            {
                "seed": seed,
                "fold": fold,
                "val_f1": fold_f1,
                "threshold": threshold,
                "val_positive_count": int((val_probs > threshold).sum()),
            }
        )
        test_fold_probs.append(test_probs)
        print(
            f"seed={seed} fold={fold}/{args.folds} | "
            f"val_f1={fold_f1:.4f} | threshold={threshold:.3f}"
        )

    threshold, oof_f1 = find_best_threshold(labels, oof)
    print(f"seed={seed} OOF | f1={oof_f1:.4f} | threshold={threshold:.3f}")
    return oof, np.mean(test_fold_probs, axis=0), fold_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--sample-submission", default="", help="Defaults to <data-root>/sample_submission.csv.")
    parser.add_argument("--cache-dir", default="features")
    parser.add_argument("--foundation-backend", default="dinov2_vitb14")
    parser.add_argument("--foundation-views", default="full,center90,center80,center75,center70,center60")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--classifier", default="logreg", choices=["logreg", "linear_svm", "random_forest"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed-list", default="42,101,202,303")
    parser.add_argument("--baseline-detail", required=True)
    parser.add_argument("--baseline-score-col", default="weighted_prob")
    parser.add_argument("--baseline-weight", type=float, default=0.60)
    parser.add_argument("--inr-weight", type=float, default=0.40)
    parser.add_argument("--detail-output", default="kfold_seed_rank_ensemble_detail.csv")
    parser.add_argument("--fold-output", default="kfold_seed_rank_ensemble_folds.csv")
    parser.add_argument("--output", default="submission_kfold_seed_rank_ensemble.csv")
    parser.add_argument("--topk-list", default="103,104,105,106,107")
    parser.add_argument("--inr-image-size", type=int, default=64)
    parser.add_argument("--inr-steps", type=int, default=160)
    parser.add_argument("--inr-pixels-per-step", type=int, default=2048)
    parser.add_argument("--inr-hidden-dim", type=int, default=64)
    parser.add_argument("--inr-hidden-layers", type=int, default=3)
    parser.add_argument("--inr-omega-0", type=float, default=30.0)
    parser.add_argument("--inr-lr", type=float, default=1e-3)
    args = parser.parse_args()

    seeds = parse_number_list(args.seed_list, int)
    if not seeds:
        raise ValueError("--seed-list must contain at least one seed.")
    if args.baseline_weight < 0 or args.inr_weight < 0 or args.baseline_weight + args.inr_weight <= 0:
        raise ValueError("--baseline-weight and --inr-weight must be non-negative and sum to > 0.")

    train_ids, labels, train_paths = load_train_dataset(args.data_root)
    test_ids, test_paths = load_test_dataset(args.data_root)
    sample_submission = args.sample_submission or os.path.join(args.data_root, "sample_submission.csv")
    template = pd.read_csv(sample_submission)
    if template["id"].astype(str).tolist() != [str(item) for item in test_ids]:
        raise RuntimeError("Test ids do not match sample_submission.csv order.")

    views = parse_views(args.foundation_views)
    foundation_train = load_or_extract_features(
        train_paths,
        args.foundation_backend,
        args.cache_dir,
        args.batch_size,
        prefix="kfold_inr_train_foundation",
        views=views,
    )
    foundation_test = load_or_extract_features(
        test_paths,
        args.foundation_backend,
        args.cache_dir,
        args.batch_size,
        prefix="kfold_inr_test_foundation",
        views=views,
    )

    detail = pd.DataFrame({"id": test_ids})
    fold_rows = []
    seed_rank_scores = []
    seed_probs = []
    for seed in seeds:
        train_features, test_features = extract_seed_features(
            args,
            train_paths,
            test_paths,
            seed,
            foundation_train,
            foundation_test,
        )
        oof, test_probs, current_fold_rows = predict_kfold(args, train_features, labels, test_features, seed)
        oof_threshold, oof_f1 = find_best_threshold(labels, oof)
        detail[f"seed{seed}_prob"] = test_probs
        detail[f"seed{seed}_rank_score"] = rank_score(test_probs)
        seed_rank_scores.append(detail[f"seed{seed}_rank_score"].to_numpy(dtype=np.float64))
        seed_probs.append(test_probs)
        fold_rows.extend(current_fold_rows)
        print(f"seed={seed} complete | oof_f1={oof_f1:.4f} | oof_threshold={oof_threshold:.3f}")

    baseline_score = load_external_scores(args.baseline_detail, args.baseline_score_col, test_ids)
    baseline_rank_score = rank_score(baseline_score)
    inr_rank_score = np.mean(np.vstack(seed_rank_scores), axis=0)
    final_score = (
        args.baseline_weight * baseline_rank_score + args.inr_weight * inr_rank_score
    ) / (args.baseline_weight + args.inr_weight)

    detail["baseline_score"] = baseline_score
    detail["baseline_rank_score"] = baseline_rank_score
    detail["inr_rank_score"] = inr_rank_score
    detail["mean_seed_prob"] = np.mean(np.vstack(seed_probs), axis=0)
    detail["final_score"] = final_score
    detail["rank"] = detail["final_score"].rank(method="first", ascending=False).astype(int)
    detail.sort_values("rank").to_csv(args.detail_output, index=False)
    pd.DataFrame(fold_rows).to_csv(args.fold_output, index=False)
    print(f"Saved detail: {args.detail_output}")
    print(f"Saved fold summary: {args.fold_output}")

    stem, ext = split_output_name(args.output)
    for topk in parse_number_list(args.topk_list, int):
        save_topk_submission(template, test_ids, final_score, topk, f"{stem}_top{topk}{ext}")


if __name__ == "__main__":
    main()
