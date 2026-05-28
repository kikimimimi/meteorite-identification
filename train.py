import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from tqdm import tqdm

from dataset import StoneDataset
from model import get_model


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transforms(image_size):
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.RandomResizedCrop((image_size, image_size), scale=(0.82, 1.0), ratio=(0.85, 1.15), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=12),
        v2.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def find_best_threshold(labels, probs):
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(labels, (probs > threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold), float(best_f1)


def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            probs = torch.softmax(model(images), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def train_one_epoch(model, loader, criterion, optimizer, desc):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc=desc):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def get_classifier_head(model):
    if hasattr(model, "head"):
        return model.head
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "fc"):
        return model.fc
    raise AttributeError("Cannot find classifier head on model.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "efficientnet_b3", "swin_v2_t"])
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Fine-tuning image model | arch={args.arch} | data_root={args.data_root}")

    train_transform, val_transform = build_transforms(args.image_size)
    full_dataset = StoneDataset(root=args.data_root, split="train", transforms=None)
    labels = np.array(full_dataset.labels)
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_size,
        stratify=labels,
        random_state=args.seed,
    )

    train_ds = StoneDataset(root=args.data_root, split="train", transforms=train_transform)
    val_ds = StoneDataset(root=args.data_root, split="train", transforms=val_transform)
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = get_model(arch=args.arch).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    print("Warmup classifier head...")
    for param in model.parameters():
        param.requires_grad = False
    for param in get_classifier_head(model).parameters():
        param.requires_grad = True

    optimizer_head = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.head_lr)
    for epoch in range(args.warmup_epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer_head, f"Warmup {epoch + 1}")
        print(f"Warmup {epoch + 1} | Loss: {loss:.4f}")

    print("Fine-tuning full model...")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    best_threshold = 0.5
    checkpoint_name = args.checkpoint_name or f"{args.arch}_best_model.pth"
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, f"Epoch {epoch + 1}")
        probs, val_labels = evaluate(model, val_loader)
        threshold, f1 = find_best_threshold(val_labels, probs)
        print(f"Epoch {epoch + 1} | Loss: {train_loss:.4f} | Val F1: {f1:.4f} @ {threshold:.2f}")
        scheduler.step()

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            path = os.path.join(args.output_dir, checkpoint_name)
            torch.save({
                "model": model.state_dict(),
                "threshold": float(best_threshold),
                "f1": float(best_f1),
                "arch": args.arch,
                "image_size": int(args.image_size),
                "data_root": args.data_root,
                "split": "single_stratified",
            }, path)
            print(f"Saved {path} | F1: {best_f1:.4f}")

    print(f"Baseline complete | best val F1={best_f1:.4f} threshold={best_threshold:.3f}")


if __name__ == "__main__":
    main()
