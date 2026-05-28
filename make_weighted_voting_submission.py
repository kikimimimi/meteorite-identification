import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from submission_utils import save_threshold_submission, save_topk_submission, split_output_name


def parse_number_list(value, cast_type=float):
    if value is None or value == "":
        return []
    return [cast_type(item.strip()) for item in str(value).split(",") if item.strip()]


def sanitize_name(path, index):
    stem = Path(path).stem
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    return safe or f"submission_{index}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-submission", default="sample_submission.csv")
    parser.add_argument("--submissions", required=True, help="Comma-separated historical submission csv files.")
    parser.add_argument("--scores", required=True, help="Comma-separated leaderboard scores matching --submissions.")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--score-power", type=float, default=1.0)
    parser.add_argument("--detail-output", default="weighted_voting_detail.csv")
    parser.add_argument("--output", default="submission_weighted_voting.csv")
    parser.add_argument("--threshold-list", default="0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--topk-list", default="80,90,100,105,108,110,120,128")
    args = parser.parse_args()

    template = pd.read_csv(args.sample_submission)
    files = [item.strip() for item in args.submissions.split(",") if item.strip()]
    scores = parse_number_list(args.scores, float)
    if len(files) != len(scores):
        raise ValueError("--submissions and --scores must have the same length.")

    selected = []
    for path, score in zip(files, scores):
        if args.min_score is not None and score < args.min_score:
            print(f"Skipping {path}: score {score:.4f} < min_score {args.min_score:.4f}")
            continue
        selected.append((path, score))
    if not selected:
        raise ValueError("No submissions left after --min-score filtering.")

    template_ids = template["id"].astype(str).tolist()
    detail = pd.DataFrame({"id": template["id"]})
    label_columns = []
    weights = []
    for index, (path, score) in enumerate(selected, start=1):
        df = pd.read_csv(path)
        if list(df.columns) != list(template.columns):
            raise ValueError(f"{path} columns do not match sample_submission.csv.")
        if df["id"].astype(str).tolist() != template_ids:
            raise ValueError(f"{path} id order does not match sample_submission.csv.")
        labels = df["label"].astype(int)
        if not set(labels.unique().tolist()).issubset({0, 1}):
            raise ValueError(f"{path} contains labels outside 0/1.")
        col = f"label_{index}_{sanitize_name(path, index)}"
        detail[col] = labels
        label_columns.append(col)
        weights.append(float(score) ** float(args.score_power))
        print(f"Using {path} | score={score:.4f} | weight={weights[-1]:.6f} | positives={int(labels.sum())}")

    label_matrix = detail[label_columns].to_numpy(dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    detail["weighted_vote_percent"] = (label_matrix * weights).sum(axis=1) / weights.sum()
    detail["vote_count"] = label_matrix.sum(axis=1).astype(int)
    detail["rank"] = detail["weighted_vote_percent"].rank(method="first", ascending=False).astype(int)
    detail.sort_values("rank").to_csv(args.detail_output, index=False)
    print(f"Saved detail: {args.detail_output}")

    score = detail["weighted_vote_percent"].to_numpy()
    stem, ext = split_output_name(args.output)
    save_threshold_submission(template, template["id"].tolist(), score, 0.5, args.output)
    for threshold in parse_number_list(args.threshold_list, float):
        save_threshold_submission(template, template["id"].tolist(), score, threshold, f"{stem}_th{threshold:.2f}{ext}")
    for topk in parse_number_list(args.topk_list, int):
        save_topk_submission(template, template["id"].tolist(), score, topk, f"{stem}_top{topk}{ext}")


if __name__ == "__main__":
    main()
