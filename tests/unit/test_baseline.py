from typing import Any

import cv2
import numpy as np

from roadsign_assist.baseline.benchmark import extract_crop_feature_sets
from roadsign_assist.baseline.pipeline import process_image
from roadsign_assist.config import load_yaml


def _synthetic_signs() -> np.ndarray[Any, np.dtype[np.uint8]]:
    image = np.full((360, 640, 3), 230, dtype=np.uint8)
    cv2.circle(image, (130, 150), 65, (0, 0, 230), thickness=16)
    cv2.rectangle(image, (280, 80), (400, 200), (220, 80, 20), thickness=-1)
    diamond = np.asarray([[500, 70], [580, 150], [500, 230], [420, 150]], dtype=np.int32)
    cv2.fillConvexPoly(image, diamond, (20, 210, 230))
    return image


def test_baseline_finds_multiple_colors_without_expected_labels() -> None:
    image = _synthetic_signs()
    result = process_image(
        image,
        image_id="synthetic",
        image_path="renamed-anything.png",
        config=load_yaml("configs/baseline/default.yaml"),
    )
    colors = {candidate.color for candidate in result.candidates}
    assert {"red", "blue", "yellow"}.issubset(colors)
    assert result.runtime_ms < 2000


def test_crop_features_depend_only_on_pixels() -> None:
    image = _synthetic_signs()
    first = extract_crop_feature_sets(image)
    second = extract_crop_feature_sets(image.copy())
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
