from __future__ import annotations

import math
from typing import Any, cast

import cv2
import numpy as np

from roadsign_assist.baseline.models import BoundingBox, Candidate, Contour, UInt8Image


def classify_shape(vertices: int, circularity: float, aspect_ratio: float) -> str:
    if circularity >= 0.76 and 0.72 <= aspect_ratio <= 1.38:
        return "circle"
    if vertices == 3:
        return "triangle"
    if vertices == 4:
        return "square_or_rectangle"
    if 7 <= vertices <= 10:
        return "octagon"
    return "other"


def _candidate_score(
    *,
    solidity: float,
    extent: float,
    circularity: float,
    aspect_ratio: float,
    shape_label: str,
    area_ratio: float,
) -> float:
    shape_bonus = {
        "circle": 0.15,
        "triangle": 0.15,
        "square_or_rectangle": 0.10,
        "octagon": 0.18,
        "other": 0.0,
    }[shape_label]
    aspect_score = max(0.0, 1.0 - abs(math.log(max(aspect_ratio, 1e-6))) / 2.0)
    area_score = min(1.0, area_ratio / 0.02)
    return float(
        np.clip(
            0.26 * solidity
            + 0.20 * extent
            + 0.18 * circularity
            + 0.12 * aspect_score
            + 0.09 * area_score
            + shape_bonus,
            0.0,
            1.0,
        )
    )


def extract_candidates(
    masks: dict[str, UInt8Image],
    image_shape: tuple[int, ...],
    config: dict[str, Any],
) -> list[Candidate]:
    settings = config["candidates"]
    image_height, image_width = image_shape[:2]
    image_area = image_width * image_height
    candidates: list[Candidate] = []

    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for raw_contour in contours:
            contour = cast(Contour, raw_contour)
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue
            area_ratio = area / image_area
            x, y, width, height = cv2.boundingRect(contour)
            if (
                area_ratio < float(settings["min_area_ratio"])
                or area_ratio > float(settings["max_area_ratio"])
                or width < int(settings["min_width"])
                or height < int(settings["min_height"])
            ):
                continue

            bbox_area = max(1, width * height)
            extent = area / bbox_area
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area else 0.0
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0
            approximation = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            vertices = len(approximation)
            aspect_ratio = width / max(1, height)
            if (
                extent < float(settings["min_extent"])
                or solidity < float(settings["min_solidity"])
                or aspect_ratio > float(settings["max_aspect_ratio"])
                or aspect_ratio < 1.0 / float(settings["max_aspect_ratio"])
            ):
                continue

            shape_label = classify_shape(vertices, circularity, aspect_ratio)
            score = _candidate_score(
                solidity=solidity,
                extent=extent,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                shape_label=shape_label,
                area_ratio=area_ratio,
            )
            candidates.append(
                Candidate(
                    color=color,
                    bbox=BoundingBox(x=x, y=y, width=width, height=height),
                    contour=contour,
                    area=area,
                    area_ratio=area_ratio,
                    aspect_ratio=aspect_ratio,
                    extent=extent,
                    solidity=solidity,
                    circularity=circularity,
                    polygon_vertices=vertices,
                    shape_label=shape_label,
                    score=score,
                )
            )

    return non_maximum_suppression(candidates)


def non_maximum_suppression(
    candidates: list[Candidate],
    iou_threshold: float = 0.45,
) -> list[Candidate]:
    selected: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(
            candidate.bbox.intersection_over_union(existing.bbox) < iou_threshold
            for existing in selected
        ):
            selected.append(candidate)
    return selected
