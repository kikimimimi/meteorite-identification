import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def stable_cache_name(prefix, paths, backend):
    digest = hashlib.md5()
    digest.update(backend.encode("utf-8"))
    for path in paths:
        digest.update(str(path).encode("utf-8"))
    return f"{prefix}_{backend}_{digest.hexdigest()[:12]}.npz"


def parse_views(value):
    if value is None:
        return ["full"]
    if isinstance(value, (list, tuple)):
        views = list(value)
    else:
        views = [item.strip() for item in str(value).split(",") if item.strip()]
    return views or ["full"]


def parse_number_list(value, cast_type=float):
    if value is None or value == "":
        return []
    return [cast_type(item.strip()) for item in str(value).split(",") if item.strip()]


class RelativeSquareCrop:
    def __init__(self, scale=1.0, position="center"):
        self.scale = float(scale)
        self.position = position

    def __call__(self, image):
        width, height = image.size
        side = max(1, int(round(min(width, height) * self.scale)))
        side = min(side, width, height)

        if "left" in self.position:
            left = 0
        elif "right" in self.position:
            left = width - side
        else:
            left = (width - side) // 2

        if "top" in self.position:
            top = 0
        elif "bottom" in self.position:
            top = height - side
        else:
            top = (height - side) // 2

        return image.crop((left, top, left + side, top + side))


def view_to_crop(view):
    if view in ("full", "default"):
        return None

    presets = {
        "center90": (0.90, "center"),
        "center80": (0.80, "center"),
        "center75": (0.75, "center"),
        "center70": (0.70, "center"),
        "center60": (0.60, "center"),
        "top_left75": (0.75, "top_left"),
        "top_right75": (0.75, "top_right"),
        "bottom_left75": (0.75, "bottom_left"),
        "bottom_right75": (0.75, "bottom_right"),
    }
    if view in presets:
        return presets[view]

    if view.startswith("center"):
        try:
            return float(view.replace("center", "")) / 100.0, "center"
        except ValueError as exc:
            raise ValueError(f"Invalid view: {view}") from exc

    raise ValueError(f"Unknown feature view: {view}")


def apply_view(image, view="full"):
    crop = view_to_crop(view)
    if crop is None:
        return image
    scale, position = crop
    return RelativeSquareCrop(scale=scale, position=position)(image)


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = [Path(path) for path in paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class DinoV2Extractor:
    def __init__(self, model_name="dinov2_vits14"):
        self.model_name = model_name
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        except Exception as exc:
            raise RuntimeError(
                "Could not load DINOv2 through torch.hub. "
                "Make sure the machine has internet for the first run or has the torch hub cache ready."
            ) from exc
        self.model.to(DEVICE)
        self.model.eval()

    def make_transform(
        self,
        hflip=False,
        vflip=False,
        rotation=0,
        brightness=1.0,
        contrast=1.0,
        view="full",
    ):
        ops = []
        crop = view_to_crop(view)
        if crop is not None:
            scale, position = crop
            ops.append(RelativeSquareCrop(scale=scale, position=position))
        ops.extend([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
        ])
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

    @torch.no_grad()
    def encode_paths(self, paths, batch_size=16, num_workers=2, desc="DINOv2", transform=None):
        dataset = ImagePathDataset(paths, transform or self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        features = []
        for images in tqdm(loader, desc=desc):
            images = images.to(DEVICE)
            output = self.model(images)
            if isinstance(output, dict):
                output = output.get("x_norm_clstoken", output.get("last_hidden_state"))
            if output.ndim == 3:
                output = output[:, 0]
            features.append(output.detach().cpu().float().numpy())
        return np.concatenate(features, axis=0)

    def encode_paths_views(self, paths, views=None, batch_size=16, num_workers=2, desc="DINOv2"):
        views = parse_views(views)
        all_features = []
        for view in views:
            features = self.encode_paths(
                paths,
                batch_size=batch_size,
                num_workers=num_workers,
                desc=f"{desc} view={view}",
                transform=self.make_transform(view=view),
            )
            all_features.append(features)
        if len(all_features) == 1:
            return all_features[0]
        return np.concatenate(all_features, axis=1)


class ClipExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            raise RuntimeError("CLIP backend requires transformers: pip install transformers") from exc

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def make_transform(
        self,
        hflip=False,
        vflip=False,
        rotation=0,
        brightness=1.0,
        contrast=1.0,
        view="full",
    ):
        ops = []
        crop = view_to_crop(view)
        if crop is not None:
            scale, position = crop
            ops.append(RelativeSquareCrop(scale=scale, position=position))
        if hflip:
            ops.append(transforms.RandomHorizontalFlip(p=1.0))
        if vflip:
            ops.append(transforms.RandomVerticalFlip(p=1.0))
        if rotation != 0:
            ops.append(transforms.RandomRotation((rotation, rotation), fill=255))
        if brightness != 1.0 or contrast != 1.0:
            ops.append(transforms.ColorJitter(brightness=(brightness, brightness), contrast=(contrast, contrast)))
        return transforms.Compose(ops) if ops else None

    @torch.no_grad()
    def encode_paths(self, paths, batch_size=16, num_workers=0, desc="CLIP", transform=None):
        all_features = []
        for start in tqdm(range(0, len(paths), batch_size), desc=desc):
            batch_paths = paths[start:start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            if transform is not None:
                images = [transform(image) for image in images]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
            features = self.model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=1)
            all_features.append(features.detach().cpu().float().numpy())
        return np.concatenate(all_features, axis=0)

    def encode_paths_views(self, paths, views=None, batch_size=16, num_workers=0, desc="CLIP"):
        views = parse_views(views)
        all_features = []
        for view in views:
            all_features.append(
                self.encode_paths(
                    paths,
                    batch_size=batch_size,
                    desc=f"{desc} view={view}",
                    transform=self.make_transform(view=view),
                )
            )
        if len(all_features) == 1:
            return all_features[0]
        return np.concatenate(all_features, axis=1)


def build_extractor(backend):
    if backend == "dinov3" or backend.startswith("dinov3"):
        raise NotImplementedError(
            "DINOv3 backend is reserved but not wired yet. "
            "Please provide a local model path/loading recipe before using --backend dinov3."
        )
    if backend.startswith("dinov2"):
        model_name = backend if backend != "dinov2" else "dinov2_vits14"
        return DinoV2Extractor(model_name=model_name)
    if backend == "clip":
        return ClipExtractor()
    raise ValueError(f"Unknown feature backend: {backend}")


def load_or_extract_features(paths, backend, cache_dir, batch_size, prefix="train", views=None, num_workers=2):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    views = parse_views(views)
    cache_backend = f"{backend}_views-{'-'.join(views)}"
    cache_path = cache_dir / stable_cache_name(prefix, paths, cache_backend)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["features"]

    extractor = build_extractor(backend)
    if hasattr(extractor, "encode_paths_views"):
        features = extractor.encode_paths_views(
            paths,
            views=views,
            batch_size=batch_size,
            num_workers=num_workers,
            desc=f"Extract {prefix} {backend}",
        )
    elif views != ["full"]:
        raise ValueError(f"--views is not supported for backend={backend}")
    else:
        features = extractor.encode_paths(
            paths,
            batch_size=batch_size,
            num_workers=num_workers,
            desc=f"Extract {prefix} {backend}",
        )
    np.savez_compressed(cache_path, features=features, paths=np.array([str(path) for path in paths]))
    print(f"Cached features: {cache_path}")
    return features


def extract_with_tta(extractor, paths, backend, batch_size, use_tta=False, views=None, num_workers=2, desc_prefix="Predict"):
    views = parse_views(views)
    view_features = []
    for view in views:
        if not use_tta:
            view_features.append(
                extractor.encode_paths(
                    paths,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    desc=f"{desc_prefix} {backend} view={view}",
                    transform=extractor.make_transform(view=view) if hasattr(extractor, "make_transform") else None,
                )
            )
            continue

        if not hasattr(extractor, "make_transform"):
            raise ValueError(f"TTA is not supported for backend={backend}")
        transforms_for_tta = [
            extractor.make_transform(view=view),
            extractor.make_transform(view=view, hflip=True),
            extractor.make_transform(view=view, vflip=True),
            extractor.make_transform(view=view, rotation=8),
            extractor.make_transform(view=view, rotation=-8),
            extractor.make_transform(view=view, brightness=1.05, contrast=1.04),
            extractor.make_transform(view=view, brightness=0.95, contrast=0.96),
        ]
        features = []
        for index, transform in enumerate(transforms_for_tta, start=1):
            features.append(
                extractor.encode_paths(
                    paths,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    desc=f"{desc_prefix} {backend} {view} TTA {index}/{len(transforms_for_tta)}",
                    transform=transform,
                )
            )
        view_features.append(np.mean(features, axis=0))

    if len(view_features) == 1:
        return view_features[0]
    return np.concatenate(view_features, axis=1)
