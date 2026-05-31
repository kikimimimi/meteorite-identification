import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from dataset import StoneDataset


def expand_bbox(bbox, width, height, margin_ratio):
    x, y, w, h = [int(round(value)) for value in bbox]
    margin = int(max(w, h) * margin_ratio)
    return (
        max(0, x - margin),
        max(0, y - margin),
        min(width, x + w + margin),
        min(height, y + h + margin),
    )


def crop_mask_to_white(image, mask, margin_ratio):
    image = image.convert("RGB")
    width, height = image.size
    bbox = expand_bbox(mask["bbox"], width, height, margin_ratio)
    mask_image = Image.fromarray((mask["segmentation"].astype(np.uint8) * 255), mode="L")
    crop = image.crop(bbox)
    crop_mask = mask_image.crop(bbox).filter(ImageFilter.GaussianBlur(1.5))
    background = Image.new("RGB", crop.size, (255, 255, 255))
    return Image.composite(crop, background, crop_mask)


def mask_stats(mask, image):
    segmentation = mask["segmentation"]
    pixels = image[segmentation]
    if len(pixels) == 0:
        return {
            "mean": 255.0,
            "std": 0.0,
            "saturation": 0.0,
            "texture": 0.0,
        }

    pixels_f = pixels.astype(np.float32)
    gray = pixels_f.mean(axis=1)
    saturation = pixels_f.max(axis=1) - pixels_f.min(axis=1)
    red, green, blue = pixels_f[:, 0], pixels_f[:, 1], pixels_f[:, 2]
    white_ratio = np.mean((red > 225) & (green > 225) & (blue > 225))
    dark_ratio = np.mean((red < 70) & (green < 70) & (blue < 70))
    skin_ratio = np.mean(
        (red > 95)
        & (green > 40)
        & (blue > 20)
        & ((red - green) > 12)
        & ((red - blue) > 25)
        & ((np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])) > 15)
    )
    x, y, w, h = [int(round(value)) for value in mask["bbox"]]
    crop = image[y:y + h, x:x + w]
    crop_gray = crop.mean(axis=2).astype(np.float32)
    if crop_gray.shape[0] > 1 and crop_gray.shape[1] > 1:
        gx = np.abs(crop_gray[:, 1:] - crop_gray[:, :-1]).mean()
        gy = np.abs(crop_gray[1:, :] - crop_gray[:-1, :]).mean()
        texture = float(gx + gy)
    else:
        texture = 0.0

    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "saturation": float(saturation.mean()),
        "texture": texture,
        "white_ratio": float(white_ratio),
        "dark_ratio": float(dark_ratio),
        "skin_ratio": float(skin_ratio),
    }


def looks_like_background_or_marker(mask, image, width, height):
    x, y, w, h = [int(round(value)) for value in mask["bbox"]]
    area_ratio = mask["area"] / float(width * height)
    bbox_area_ratio = (w * h) / float(width * height)
    rectangularity = mask["area"] / max(1.0, float(w * h))
    aspect = w / max(1.0, h)
    touches_left = x <= 3
    touches_top = y <= 3
    touches_right = x + w >= width - 3
    touches_bottom = y + h >= height - 3
    border_touches = sum([touches_left, touches_top, touches_right, touches_bottom])
    stats = mask_stats(mask, image)

    large_uniform_background = (
        area_ratio > 0.72
        and stats["white_ratio"] > 0.55
        and stats["std"] < 35
        and stats["texture"] < 24
    )
    large_rectangular_plate = (
        bbox_area_ratio > 0.55
        and rectangularity > 0.72
        and stats["white_ratio"] > 0.45
        and stats["texture"] < 30
    )
    border_background = (
        border_touches >= 2
        and area_ratio > 0.25
        and stats["white_ratio"] > 0.45
        and stats["texture"] < 30
    )
    if large_uniform_background or large_rectangular_plate or border_background:
        return True

    very_uniform_white = (
        stats["mean"] > 228
        and stats["std"] < 24
        and stats["saturation"] < 18
        and stats["texture"] < 18
    )
    rectangular_white_marker = (
        0.60 <= aspect <= 1.65
        and rectangularity > 0.72
        and stats["mean"] > 210
        and stats["std"] < 35
        and stats["texture"] < 24
    )
    thin_text_or_label = (
        (h / height) < 0.14
        and aspect > 2.2
        and stats["mean"] < 235
    )
    ruler_or_scale = (
        rectangularity > 0.80
        and (aspect > 2.0 or aspect < 0.50)
        and area_ratio < 0.25
        and stats["texture"] < 45
    )
    hand_or_finger = (
        stats["skin_ratio"] > 0.20
        or (stats["skin_ratio"] > 0.08 and border_touches >= 1 and area_ratio > 0.04)
    )
    return very_uniform_white or rectangular_white_marker or thin_text_or_label or ruler_or_scale or hand_or_finger


def score_sam_masks(masks, image):
    height, width = image.shape[:2]
    image_area = width * height
    min_side = max(24, int(round(min(width, height) * 0.015)))
    min_area = max(80, int(round(image_area * 0.0005)))
    candidates = []
    for mask in masks:
        area_ratio = mask["area"] / image_area
        if mask["area"] <= 0:
            continue

        x, y, w, h = [int(round(value)) for value in mask["bbox"]]
        if w < min_side or h < min_side or mask["area"] < min_area:
            continue

        cx = (x + w / 2) / width
        cy = (y + h / 2) / height

        center_score = 1.0 - min(1.0, ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5)
        quality = mask.get("predicted_iou", 0.0) + mask.get("stability_score", 0.0)
        area_score = 1.0 - min(abs(area_ratio - 0.30), 0.70)
        score = (
            quality
            + 0.15 * center_score
            + 0.10 * area_score
        )
        candidates.append((score, mask))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


class SemanticMaskRanker:
    def __init__(
        self,
        backend,
        model_name=None,
        device="cuda",
        max_candidates=8,
        semantic_weight=1.2,
    ):
        self.backend = backend
        self.max_candidates = max_candidates
        self.semantic_weight = semantic_weight
        try:
            import torch
            from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
        except Exception as exc:
            raise RuntimeError(
                "Semantic mask ranking requires transformers and torch. "
                "Install them first, e.g. pip install transformers."
            ) from exc

        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.positive_prompts = [
            "a photo of a meteorite",
            "a photo of a rough stone",
            "a photo of a rock specimen",
            "a close-up photo of a mineral or stone",
        ]
        self.negative_prompts = [
            "a photo of a human hand",
            "a photo of a ruler or measuring scale",
            "printed text or a label",
            "a white background or tray",
            "a piece of paper or card",
        ]
        self.prompts = self.positive_prompts + self.negative_prompts

        if backend == "clip":
            model_name = model_name or "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        elif backend == "siglip":
            model_name = model_name or "google/siglip-base-patch16-224"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        else:
            raise ValueError(f"Unknown semantic mask ranker: {backend}")
        self.model.eval()

    def score_crops(self, crops):
        if not crops:
            return np.array([], dtype=np.float32)
        with self.torch.no_grad():
            inputs = self.processor(
                text=self.prompts,
                images=crops,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image.float()
            pos_count = len(self.positive_prompts)
            pos = logits[:, :pos_count].max(dim=1).values
            neg = logits[:, pos_count:].max(dim=1).values
            scores = self.torch.tanh((pos - neg) / 5.0)
        return scores.detach().cpu().numpy()

    def choose(self, candidates, pil_image, margin_ratio):
        if not candidates:
            return None
        shortlisted = candidates[:self.max_candidates]
        crops = [crop_mask_to_white(pil_image, mask, margin_ratio) for _, mask in shortlisted]
        semantic_scores = self.score_crops(crops)
        combined = []
        for (heuristic_score, mask), semantic_score in zip(shortlisted, semantic_scores):
            combined_score = heuristic_score + self.semantic_weight * float(semantic_score)
            mask["_semantic_score"] = float(semantic_score)
            mask["_combined_score"] = float(combined_score)
            combined.append((combined_score, mask))
        combined.sort(key=lambda item: item[0], reverse=True)
        return combined[0][1]


def choose_sam_mask(masks, image, pil_image=None, mask_ranker=None, margin_ratio=0.10):
    valid = score_sam_masks(masks, image)
    if mask_ranker is not None and pil_image is not None:
        return mask_ranker.choose(valid, pil_image, margin_ratio)
    if not valid:
        return None
    return valid[0][1]


def sam_crop_to_white(image, mask_generator, margin_ratio=0.10, mask_ranker=None):
    image = image.convert("RGB")
    np_image = np.asarray(image)
    masks = mask_generator.generate(np_image)
    chosen = choose_sam_mask(
        masks,
        np_image,
        pil_image=image,
        mask_ranker=mask_ranker,
        margin_ratio=margin_ratio,
    )
    if chosen is None:
        return image

    return crop_mask_to_white(image, chosen, margin_ratio)


def draw_debug_image(image, chosen, output_path):
    debug = image.convert("RGB").copy()
    draw = ImageDraw.Draw(debug)
    if chosen is not None:
        x, y, w, h = [int(round(value)) for value in chosen["bbox"]]
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=max(2, min(debug.size) // 180))
    debug.save(output_path)


def load_sam(checkpoint, model_type, device):
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except Exception as exc:
        raise RuntimeError(
            "SAM preprocessing requires segment-anything. Install it and provide a checkpoint, "
            "for example: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=24,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.88,
        crop_n_layers=1,
        min_mask_region_area=200,
    )


def save_processed_split(
    dataset,
    output_dir,
    split,
    mask_generator,
    margin_ratio,
    debug_dir=None,
    debug_limit=0,
    mask_ranker=None,
):
    image_dir = output_dir / f"{split}_images" / f"{split}_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for index, (path, image_id) in enumerate(tqdm(list(zip(dataset.samples, dataset.ids)), desc=f"SAM {split}")):
        if path is None:
            continue
        image = Image.open(path).convert("RGB")
        try:
            width, height = image.size
            np_image = np.asarray(image)
            masks = mask_generator.generate(np_image)
            chosen = choose_sam_mask(
                masks,
                np_image,
                pil_image=image,
                mask_ranker=mask_ranker,
                margin_ratio=margin_ratio,
            )
            if chosen is None:
                processed = image
            else:
                processed = crop_mask_to_white(image, chosen, margin_ratio)
            if debug_dir is not None and index < debug_limit:
                draw_debug_image(image, chosen, debug_dir / image_id)
        except Exception as exc:
            print(f"Warning: SAM failed for {image_id}, saving original image. Error: {exc}")
            processed = image
        processed.save(image_dir / image_id)


def save_original_split(dataset, output_dir, split):
    image_dir = output_dir / f"{split}_images" / f"{split}_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for path, image_id in tqdm(list(zip(dataset.samples, dataset.ids)), desc=f"Copy {split}"):
        if path is None:
            continue
        image = Image.open(path).convert("RGB")
        image.save(image_dir / image_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=".")
    parser.add_argument("--output-root", default="processed/sam")
    parser.add_argument("--checkpoint", default=os.environ.get("SAM_CHECKPOINT"))
    parser.add_argument("--model-type", default=os.environ.get("SAM_MODEL_TYPE", "vit_b"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--debug-limit", type=int, default=0)
    parser.add_argument(
        "--mask-ranker",
        choices=["none", "clip", "siglip"],
        default="none",
        help="Use CLIP/SigLIP to rerank SAM mask candidates.",
    )
    parser.add_argument("--mask-ranker-model", default=None)
    parser.add_argument("--mask-ranker-max-candidates", type=int, default=8)
    parser.add_argument("--mask-ranker-weight", type=float, default=1.2)
    parser.add_argument(
        "--process-test",
        action="store_true",
        help="Also run SAM on test images. By default test images are copied unchanged.",
    )
    args = parser.parse_args()

    if not args.checkpoint:
        raise ValueError("Please pass --checkpoint or set SAM_CHECKPOINT.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(args.input_root) / "train_labels.csv", output_root / "train_labels.csv")
    shutil.copy2(Path(args.input_root) / "sample_submission.csv", output_root / "sample_submission.csv")

    mask_generator = load_sam(args.checkpoint, args.model_type, args.device)
    mask_ranker = None
    if args.mask_ranker != "none":
        mask_ranker = SemanticMaskRanker(
            args.mask_ranker,
            model_name=args.mask_ranker_model,
            device=args.device,
            max_candidates=args.mask_ranker_max_candidates,
            semantic_weight=args.mask_ranker_weight,
        )
    train_ds = StoneDataset(args.input_root, split="train", transforms=None)
    test_ds = StoneDataset(args.input_root, split="test", transforms=None)
    debug_root = Path(args.debug_dir) if args.debug_dir else None
    save_processed_split(
        train_ds,
        output_root,
        "train",
        mask_generator,
        args.margin,
        debug_dir=(debug_root / "train") if debug_root else None,
        debug_limit=args.debug_limit,
        mask_ranker=mask_ranker,
    )
    if args.process_test:
        save_processed_split(
            test_ds,
            output_root,
            "test",
            mask_generator,
            args.margin,
            debug_dir=(debug_root / "test") if debug_root else None,
            debug_limit=args.debug_limit,
            mask_ranker=mask_ranker,
        )
    else:
        save_original_split(test_ds, output_root, "test")
    print(f"SAM-processed dataset saved to {output_root}")


if __name__ == "__main__":
    main()
