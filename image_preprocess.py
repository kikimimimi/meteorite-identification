import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def _expand_bbox(bbox, width, height, margin_ratio):
    left, top, right, bottom = bbox
    box_w = right - left
    box_h = bottom - top
    margin = int(max(box_w, box_h) * margin_ratio)
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(width, right + margin),
        min(height, bottom + margin),
    )


def _foreground_mask(image):
    arr = np.asarray(image.convert("RGB")).astype(np.int16)
    height, width = arr.shape[:2]

    border = max(6, min(width, height) // 24)
    border_pixels = np.concatenate([
        arr[:border].reshape(-1, 3),
        arr[-border:].reshape(-1, 3),
        arr[:, :border].reshape(-1, 3),
        arr[:, -border:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(border_pixels, axis=0)

    color_dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = max_channel - min_channel

    # White-background test images are handled by color distance from the border.
    # Natural-background training images get extra help from saturation/texture cues.
    threshold = max(24, np.percentile(color_dist[:border], 90) + 8)
    mask = (color_dist > threshold) | ((saturation > 28) & (max_channel < 248))

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.filter(ImageFilter.MedianFilter(5))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(9))
    mask_img = mask_img.filter(ImageFilter.MinFilter(5))
    return mask_img


class ForegroundCropWhite:
    """Crop the dominant stone-like foreground and place it on a clean background."""

    def __init__(
        self,
        p=1.0,
        margin_range=(0.08, 0.18),
        background_colors=((255, 255, 255), (248, 248, 248), (240, 240, 240)),
        max_side=768,
        fallback_center_crop=True,
    ):
        self.p = p
        self.margin_range = margin_range
        self.background_colors = background_colors
        self.max_side = max_side
        self.fallback_center_crop = fallback_center_crop

    def __call__(self, image):
        if random.random() > self.p:
            return image.convert("RGB")

        image = ImageOps.exif_transpose(image).convert("RGB")
        work = image.copy()
        work.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
        width, height = work.size

        mask = _foreground_mask(work)
        bbox = mask.getbbox()
        if bbox is None:
            return self._fallback(work)

        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / float(width * height)
        if bbox_area < 0.02:
            return self._fallback(work)
        if bbox_area > 0.96:
            return self._fallback(work)

        margin = random.uniform(*self.margin_range)
        bbox = _expand_bbox(bbox, width, height, margin)
        crop = work.crop(bbox)
        crop_mask = mask.crop(bbox).filter(ImageFilter.GaussianBlur(2))
        background = Image.new("RGB", crop.size, random.choice(self.background_colors))
        return Image.composite(crop, background, crop_mask)

    def _fallback(self, image):
        image = image.convert("RGB")
        if not self.fallback_center_crop:
            return image
        width, height = image.size
        side = int(min(width, height) * 0.88)
        left = max(0, (width - side) // 2)
        top = max(0, (height - side) // 2)
        return image.crop((left, top, left + side, top + side))
