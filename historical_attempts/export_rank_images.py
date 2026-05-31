import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def resolve_test_image_dir(root):
    root = Path(root)
    candidates = [
        root / "test_images" / "test_images",
        root / "test_images",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find test_images under {root}")


def resolve_image_path(image_dir, image_id):
    path = image_dir / str(image_id)
    if path.exists():
        return path
    stem = path.stem
    for ext in IMAGE_EXTENSIONS:
        candidate = path.with_name(stem + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image for id {image_id} not found in {image_dir}")


def pick_rank_column(df, preferred):
    if preferred in df.columns:
        return preferred
    for candidate in ["rank", "final_rank", "nn_rank"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No rank column found. Tried {preferred}, rank, final_rank, nn_rank.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail-file", required=True)
    parser.add_argument("--data-root", default="processed/sam_clip_nofilter")
    parser.add_argument("--output-dir", default="inspect_rank_80_130")
    parser.add_argument("--rank-column", default="rank")
    parser.add_argument("--score-column", default="final_score")
    parser.add_argument("--rank-from", type=int, default=80)
    parser.add_argument("--rank-to", type=int, default=130)
    args = parser.parse_args()

    df = pd.read_csv(args.detail_file)
    if "id" not in df.columns:
        raise ValueError(f"{args.detail_file} must contain an id column.")
    rank_column = pick_rank_column(df, args.rank_column)
    if args.score_column not in df.columns:
        score_candidates = [column for column in ["final_score", "weighted_prob", "avg_prob", "nn_score"] if column in df.columns]
        args.score_column = score_candidates[0] if score_candidates else rank_column

    selected = df[df[rank_column].between(args.rank_from, args.rank_to)].copy()
    selected = selected.sort_values(rank_column)
    if selected.empty:
        raise ValueError(f"No rows found for {rank_column} in [{args.rank_from}, {args.rank_to}].")

    image_dir = resolve_test_image_dir(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for _, row in selected.iterrows():
        image_id = str(row["id"])
        src = resolve_image_path(image_dir, image_id)
        rank = int(row[rank_column])
        score = float(row[args.score_column]) if args.score_column in row else 0.0
        dst_name = f"rank_{rank:03d}_score_{score:.5f}_{image_id}"
        dst = output_dir / dst_name
        shutil.copy2(src, dst)
        manifest_rows.append(
            {
                "copied_file": dst_name,
                "id": image_id,
                "rank": rank,
                "score": score,
                "source_path": str(src),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Copied {len(manifest)} images to {output_dir}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
