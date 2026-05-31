import argparse
import os

import numpy as np
import pandas as pd

from submission_utils import save_threshold_submission, save_topk_submission, split_output_name


def parse_csv_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_number_list(value, cast_type=float):
    if value is None or value == "":
        return []
    return [cast_type(item.strip()) for item in str(value).split(",") if item.strip()]


def load_score_file(path, score_col, template_ids, alias):
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{path} must contain an id column.")
    if score_col not in df.columns:
        raise ValueError(f"{path} does not contain score column {score_col}.")
    merged = pd.DataFrame({"id": template_ids}).merge(
        df[["id", score_col]].assign(id=df["id"].astype(str)),
        on="id",
        how="left",
    )
    if merged[score_col].isna().any():
        missing = merged.loc[merged[score_col].isna(), "id"].head(8).tolist()
        raise RuntimeError(f"Missing scores from {path} for ids: {missing}")
    return merged[score_col].to_numpy(dtype=np.float64), alias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-submission", default="processed/sam_clip_nofilter/sample_submission.csv")
    parser.add_argument("--detail-files", required=True, help="Comma-separated detail csv files.")
    parser.add_argument("--score-cols", default="weighted_prob", help="Comma-separated score columns, or one column reused.")
    parser.add_argument("--weights", default="", help="Comma-separated weights. Defaults to equal weights.")
    parser.add_argument("--aliases", default="", help="Optional comma-separated short names for output columns.")
    parser.add_argument("--detail-output", default="blended_prediction_detail.csv")
    parser.add_argument("--output", default="submission_blended.csv")
    parser.add_argument("--topk-list", default="105")
    parser.add_argument("--threshold-list", default="")
    args = parser.parse_args()

    template = pd.read_csv(args.sample_submission)
    template_ids = template["id"].astype(str).tolist()
    detail_files = parse_csv_list(args.detail_files)
    if not detail_files:
        raise ValueError("--detail-files must contain at least one file.")

    score_cols = parse_csv_list(args.score_cols)
    if len(score_cols) == 1 and len(detail_files) > 1:
        score_cols = score_cols * len(detail_files)
    if len(score_cols) != len(detail_files):
        raise ValueError("--score-cols must have length 1 or match --detail-files.")

    weights = parse_number_list(args.weights, float)
    if not weights:
        weights = [1.0] * len(detail_files)
    if len(weights) != len(detail_files):
        raise ValueError("--weights must match --detail-files.")
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("--weights must be non-negative and sum to > 0.")

    aliases = parse_csv_list(args.aliases)
    if not aliases:
        aliases = [f"score_{index}" for index in range(1, len(detail_files) + 1)]
    if len(aliases) != len(detail_files):
        raise ValueError("--aliases must match --detail-files.")

    detail = pd.DataFrame({"id": template["id"]})
    scores = []
    for path, score_col, alias in zip(detail_files, score_cols, aliases):
        score, alias = load_score_file(path, score_col, template_ids, alias)
        detail[alias] = score
        scores.append(score)
        print(f"Loaded {path}:{score_col} as {alias} | range={score.min():.4f}-{score.max():.4f}")

    matrix = np.vstack(scores).T
    final_score = (matrix * weights.reshape(1, -1)).sum(axis=1) / weights.sum()
    detail["final_score"] = final_score
    detail["rank"] = detail["final_score"].rank(method="first", ascending=False).astype(int)
    detail.sort_values("rank").to_csv(args.detail_output, index=False)
    print(f"Saved blended detail: {args.detail_output}")

    stem, ext = split_output_name(args.output)
    save_threshold_submission(template, template["id"].tolist(), final_score, 0.5, args.output)
    for threshold in parse_number_list(args.threshold_list, float):
        save_threshold_submission(template, template["id"].tolist(), final_score, threshold, f"{stem}_th{threshold:.2f}{ext}")
    for topk in parse_number_list(args.topk_list, int):
        save_topk_submission(template, template["id"].tolist(), final_score, topk, f"{stem}_top{topk}{ext}")


if __name__ == "__main__":
    main()
