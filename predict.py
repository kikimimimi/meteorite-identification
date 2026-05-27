import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import StoneDataset
from model import get_model


DATA_ROOT = "."
CHECKPOINT_PATH = "checkpoints/baseline_best_model.pth"
BATCH_SIZE = 24
OUTPUT_NAME = "submission_baseline.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_transform(rotation=0, hflip=False, vflip=False, brightness=1.0, contrast=1.0):
    ops = [transforms.Resize((384, 384))]
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


TTA_TRANSFORMS = [
    make_transform(),
    make_transform(hflip=True),
    make_transform(vflip=True),
    make_transform(rotation=8),
    make_transform(rotation=-8),
    make_transform(brightness=1.05, contrast=1.04),
    make_transform(brightness=0.95, contrast=0.96),
]


def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    arch = checkpoint.get("arch", "efficientnet_b0") if isinstance(checkpoint, dict) else "efficientnet_b0"
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    threshold = float(checkpoint.get("threshold", 0.5)) if isinstance(checkpoint, dict) else 0.5
    model = get_model(arch=arch).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {CHECKPOINT_PATH} | arch={arch} | threshold={threshold:.3f}")
    return model, threshold


def predict_with_transform(model, transform):
    dataset = StoneDataset(root=DATA_ROOT, split="test", transforms=transform)
    missing = [image_id for image_id, path in zip(dataset.ids, dataset.samples) if path is None]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images, examples: {missing[:8]}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    ids, probs = [], []
    with torch.no_grad():
        for images, batch_ids in tqdm(loader, desc="Inference", leave=False):
            images = images.to(DEVICE)
            batch_probs = torch.softmax(model(images), dim=1)[:, 1]
            probs.extend(batch_probs.cpu().numpy())
            ids.extend(batch_ids)
    return ids, np.array(probs)


def main():
    model, threshold = load_model()
    all_probs = []
    ids = None
    for index, transform in enumerate(TTA_TRANSFORMS, start=1):
        current_ids, current_probs = predict_with_transform(model, transform)
        if ids is None:
            ids = current_ids
        elif current_ids != ids:
            raise RuntimeError("TTA id order mismatch.")
        all_probs.append(current_probs)
        print(f"TTA {index}/{len(TTA_TRANSFORMS)} done.")

    probs = np.mean(all_probs, axis=0)
    result = pd.DataFrame({"id": ids, "prob": probs})
    result["label"] = (result["prob"] > threshold).astype(int)
    template = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
    submission = template[["id"]].merge(result[["id", "label"]], on="id", how="left")
    if submission["label"].isna().any():
        missing = submission.loc[submission["label"].isna(), "id"].head(8).tolist()
        raise RuntimeError(f"Missing predictions for ids: {missing}")
    submission["label"] = submission["label"].astype(int)
    submission.to_csv(OUTPUT_NAME, index=False)
    print(f"Saved {OUTPUT_NAME} | positives={int(submission['label'].sum())}/{len(submission)}")


if __name__ == "__main__":
    main()
