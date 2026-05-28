import os

import numpy as np
import pandas as pd


def format_threshold(value):
    return f"{float(value):.2f}"


def validate_submission(template, submission):
    if list(submission.columns) != list(template.columns):
        raise ValueError(
            f"Submission columns {list(submission.columns)} do not match sample columns {list(template.columns)}."
        )
    if len(submission) != len(template):
        raise ValueError(f"Submission length {len(submission)} does not match sample length {len(template)}.")
    if submission["id"].astype(str).tolist() != template["id"].astype(str).tolist():
        raise ValueError("Submission id order does not match sample_submission.csv.")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values.")
    labels = set(submission["label"].astype(int).unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Submission labels must be 0/1, got {sorted(labels)}.")
    return submission


def make_submission(template, ids, labels):
    result = pd.DataFrame({"id": [str(item) for item in ids], "label": np.asarray(labels, dtype=int)})
    submission = template.copy()
    template_ids = template["id"].astype(str)
    merged = pd.DataFrame({"id": template_ids}).merge(result, on="id", how="left")
    submission["label"] = merged["label"]
    if submission["label"].isna().any():
        missing = submission.loc[submission["label"].isna(), "id"].head(8).tolist()
        raise RuntimeError(f"Missing predictions for ids: {missing}")
    submission["label"] = submission["label"].astype(int)
    return validate_submission(template, submission)


def labels_from_threshold(probs, threshold):
    return (np.asarray(probs) > float(threshold)).astype(int)


def labels_from_topk(probs, topk):
    probs = np.asarray(probs)
    topk = int(topk)
    if topk <= 0 or topk > len(probs):
        raise ValueError(f"Invalid topk={topk}; must be in [1, {len(probs)}].")
    labels = np.zeros(len(probs), dtype=int)
    order = np.argsort(-probs)
    labels[order[:topk]] = 1
    return labels


def save_submission_file(template, ids, labels, output_name):
    submission = make_submission(template, ids, labels)
    submission.to_csv(output_name, index=False)
    positive_count = int(submission["label"].sum())
    total_count = len(submission)
    print(f"{output_name} | positive_count={positive_count} | total_count={total_count}")
    return submission


def save_threshold_submission(template, ids, probs, threshold, output_name):
    return save_submission_file(template, ids, labels_from_threshold(probs, threshold), output_name)


def save_topk_submission(template, ids, probs, topk, output_name):
    return save_submission_file(template, ids, labels_from_topk(probs, topk), output_name)


def split_output_name(output):
    stem, ext = os.path.splitext(output)
    return stem, ext or ".csv"
