import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import StoneDataset
from model import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_transform(image_size, rotation=0, hflip=False, vflip=False, brightness=1.0, contrast=1.0):
    ops = [transforms.Resize((image_size, image_size))]
    if hflip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    if vflip:
        ops.append(transforms.RandomVerticalFlip(p=1.0))
    if rotation != 0:
        ops.append(transforms.RandomRotation((rotation, rotation), fill=255))
    if brightness != 1.0 or contrast != 1.0:
        ops.append(transforms.ColorJitter(brightness=(brightness, brightness), contrast=(contrast, contrast)))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)


def build_tta_transforms(image_size):
    return [
        make_transform(image_size),
        make_transform(image_size, hflip=True),
        make_transform(image_size, vflip=True),
        make_transform(image_size, rotation=8),
        make_transform(image_size, rotation=-8),
        make_transform(image_size, brightness=1.05, contrast=1.04),
        make_transform(image_size, brightness=0.95, contrast=0.96),
    ]


def parse_number_list(value, cast_type=int):
    if value is None or value == "":
        return []
    return [cast_type(item.strip()) for item in value.split(",") if item.strip()]


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    arch = checkpoint.get("arch", "efficientnet_b0") if isinstance(checkpoint, dict) else "efficientnet_b0"
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    threshold = float(checkpoint.get("threshold", 0.5)) if isinstance(checkpoint, dict) else 0.5
    image_size = int(checkpoint.get("image_size", 384)) if isinstance(checkpoint, dict) else 384
    model = get_model(arch=arch).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {checkpoint_path} | arch={arch} | image_size={image_size} | threshold={threshold:.3f}")
    return model, threshold, image_size


def predict_with_transform(model, transform, data_root, batch_size):
    dataset = StoneDataset(root=data_root, split="test", transforms=transform)
    missing = [image_id for image_id, path in zip(dataset.ids, dataset.samples) if path is None]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images, examples: {missing[:8]}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ids, probs = [], []
    with torch.no_grad():
        for images, batch_ids in tqdm(loader, desc="Inference", leave=False):
            images = images.to(DEVICE)
            batch_probs = torch.softmax(model(images), dim=1)[:, 1]
            probs.extend(batch_probs.cpu().numpy())
            ids.extend(batch_ids)
    return ids, np.array(probs)


def save_submission(template, ids, probs, threshold, output_name):
    result = pd.DataFrame({"id": ids, "prob": probs})
    result["label"] = (result["prob"] > threshold).astype(int)
    submission = template[["id"]].merge(result[["id", "label"]], on="id", how="left")
    if submission["label"].isna().any():
        missing = submission.loc[submission["label"].isna(), "id"].head(8).tolist()
        raise RuntimeError(f"Missing predictions for ids: {missing}")
    submission["label"] = submission["label"].astype(int)
    submission.to_csv(output_name, index=False)
    print(f"Saved {output_name} | positives={int(submission['label'].sum())}/{len(submission)}")


def save_topk_submission(template, ids, probs, topk, output_name):
    df = pd.DataFrame({"id": ids, "prob": probs}).sort_values("prob", ascending=False).reset_index(drop=True)
    df["label"] = 0
    df.loc[:topk - 1, "label"] = 1
    submission = template[["id"]].merge(df[["id", "label"]], on="id", how="left")
    submission["label"] = submission["label"].astype(int)
    submission.to_csv(output_name, index=False)
    print(f"Saved {output_name} | topk={topk} | positives={int(submission['label'].sum())}/{len(submission)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--checkpoint", default="checkpoints/efficientnet_b0_best_model.pth")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--output", default="submission_baseline.csv")
    parser.add_argument("--prob-output", default="probs_baseline.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--topk-list", default="")
    args = parser.parse_args()

    model, threshold, image_size = load_model(args.checkpoint)
    if args.threshold is not None:
        threshold = float(args.threshold)
    tta_transforms = build_tta_transforms(image_size)
    all_probs = []
    ids = None
    for index, transform in enumerate(tta_transforms, start=1):
        current_ids, current_probs = predict_with_transform(model, transform, args.data_root, args.batch_size)
        if ids is None:
            ids = current_ids
        elif current_ids != ids:
            raise RuntimeError("TTA id order mismatch.")
        all_probs.append(current_probs)
        print(f"TTA {index}/{len(tta_transforms)} done.")

    probs = np.mean(all_probs, axis=0)
    template = pd.read_csv(os.path.join(args.data_root, "sample_submission.csv"))
    save_submission(template, ids, probs, threshold, args.output)

    pd.DataFrame({"id": ids, "prob": probs}).sort_values("prob", ascending=False).to_csv(args.prob_output, index=False)
    print(f"Saved probabilities: {args.prob_output} | threshold={threshold:.3f}")

    stem, ext = os.path.splitext(args.output)
    for topk in parse_number_list(args.topk_list, int):
        save_topk_submission(template, ids, probs, topk, f"{stem}_top{topk}{ext or '.csv'}")


if __name__ == "__main__":
    main()
