import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return path


def _load_csv(path, required_columns):
    df = pd.read_csv(_require_file(path))
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    return df


def _resolve_image_dir(root, dirname):
    base = Path(root) / dirname
    nested = base / dirname
    if nested.exists():
        return nested
    return base


def _resolve_image_path(image_dir, image_id):
    path = Path(image_dir) / str(image_id)
    if path.exists():
        return path

    stem = path.stem
    for ext in IMAGE_EXTENSIONS:
        candidate = path.with_name(stem + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image for id {image_id} not found in {image_dir}.")


class StoneDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split: {split}.")

        self.root = root
        self.split = split
        self.transforms = transforms
        self.samples = []
        self.labels = []
        self.ids = []

        if split == "train":
            image_dir = _resolve_image_dir(root, "train_images")
            csv_path = os.path.join(root, "train_labels.csv")
            df = _load_csv(csv_path, required_columns=["id", "label"])
            for _, row in df.iterrows():
                image_id = str(row["id"])
                self.samples.append(_resolve_image_path(image_dir, image_id))
                self.ids.append(image_id)
                self.labels.append(int(row["label"]))
        else:
            image_dir = _resolve_image_dir(root, "test_images")
            csv_path = os.path.join(root, "sample_submission.csv")
            df = _load_csv(csv_path, required_columns=["id"])
            for _, row in df.iterrows():
                image_id = str(row["id"])
                try:
                    self.samples.append(_resolve_image_path(image_dir, image_id))
                except FileNotFoundError:
                    self.samples.append(None)
                self.ids.append(image_id)
                self.labels.append(0)

    def __getitem__(self, index):
        img_path = self.samples[index]
        image_id = self.ids[index]

        if img_path is None:
            raise FileNotFoundError(f"Missing test image for id {image_id}.")

        image = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)

        if self.split == "test":
            return image, image_id
        return image, self.labels[index]

    def __len__(self):
        return len(self.samples)
