from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np

from roadsign_assist.baseline.models import UInt8Image


def preprocess_bgr(image: UInt8Image, config: dict[str, Any]) -> UInt8Image:
    settings = config["preprocessing"]
    result = image.copy()
    blur_size = int(settings.get("gaussian_blur_size", 0))
    if blur_size >= 3:
        if blur_size % 2 == 0:
            blur_size += 1
        result = cv2.GaussianBlur(result, (blur_size, blur_size), 0)
    if bool(settings.get("clahe_enabled", False)):
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        luminance, a_channel, b_channel = cv2.split(lab)
        grid = int(settings.get("clahe_grid_size", 8))
        clahe = cv2.createCLAHE(
            clipLimit=float(settings.get("clahe_clip_limit", 2.0)),
            tileGridSize=(grid, grid),
        )
        enhanced = clahe.apply(luminance)
        result = cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    return cast(UInt8Image, result)


def _morphology(mask: UInt8Image, config: dict[str, Any]) -> UInt8Image:
    settings = config["morphology"]
    iterations = int(settings.get("iterations", 1))
    open_size = max(1, int(settings.get("open_kernel", 3)))
    close_size = max(1, int(settings.get("close_kernel", 7)))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=iterations)
    result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)
    return cast(UInt8Image, result)


def _refine_giant_background_mask(
    mask: UInt8Image,
    saturation: UInt8Image,
    config: dict[str, Any],
) -> UInt8Image:
    settings = config["morphology"].get("background_refinement", {})
    if not bool(settings.get("enabled", True)):
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = mask.shape[0] * mask.shape[1]
    largest_ratio = (
        max((float(cv2.contourArea(contour)) for contour in contours), default=0.0) / image_area
    )
    if largest_ratio <= float(settings.get("giant_contour_area_ratio", 0.70)):
        return mask
    floor = int(settings.get("saturation_floor", 130))
    _, saturated = cv2.threshold(saturation, floor - 1, 255, cv2.THRESH_BINARY)
    refined = cv2.bitwise_and(mask, saturated)
    return _morphology(cast(UInt8Image, refined), config)


def segment_colors(image: UInt8Image, config: dict[str, Any]) -> dict[str, UInt8Image]:
    processed = preprocess_bgr(image, config)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    saturation = cast(UInt8Image, hsv[:, :, 1])
    masks: dict[str, UInt8Image] = {}
    for color, settings in config["colors"].items():
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for color_range in settings["ranges"]:
            lower = np.asarray(color_range["lower"], dtype=np.uint8)
            upper = np.asarray(color_range["upper"], dtype=np.uint8)
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))
        cleaned = _morphology(cast(UInt8Image, combined), config)
        masks[str(color)] = _refine_giant_background_mask(cleaned, saturation, config)
    return masks
