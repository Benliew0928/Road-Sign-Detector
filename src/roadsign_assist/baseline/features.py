from __future__ import annotations

# pyright: reportUnknownVariableType=false
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.feature import hog

from roadsign_assist.baseline.models import Candidate, UInt8Image

FloatFeatures = NDArray[np.float32]


def normalized_crop(image: UInt8Image, candidate: Candidate, size: int = 96) -> UInt8Image:
    bbox = candidate.bbox
    crop = image[bbox.y : bbox.y2, bbox.x : bbox.x2]
    if crop.size == 0:
        raise ValueError("Candidate crop is empty")
    height, width = crop.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(
        crop,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def extract_hog(crop: UInt8Image) -> FloatFeatures:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    values = hog(  # pyright: ignore[reportUnknownVariableType]
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return np.asarray(cast(object, values), dtype=np.float32)


def extract_hsv_histogram(crop: UInt8Image) -> FloatFeatures:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram.flatten().astype(np.float32)


def extract_hu_moments(candidate: Candidate) -> FloatFeatures:
    moments = cv2.moments(candidate.contour)
    values = cv2.HuMoments(moments).flatten()
    transformed = -np.sign(values) * np.log10(np.abs(values) + 1e-12)
    return transformed.astype(np.float32)


def extract_geometry(candidate: Candidate) -> FloatFeatures:
    return np.asarray(
        [
            candidate.area_ratio,
            candidate.aspect_ratio,
            candidate.extent,
            candidate.solidity,
            candidate.circularity,
            candidate.polygon_vertices / 10.0,
            candidate.score,
        ],
        dtype=np.float32,
    )


def extract_feature_sets(
    image: UInt8Image,
    candidate: Candidate,
) -> dict[str, FloatFeatures]:
    crop = normalized_crop(image, candidate)
    hog_features = extract_hog(crop)
    hsv_features = extract_hsv_histogram(crop)
    hu_features = extract_hu_moments(candidate)
    geometry = extract_geometry(candidate)
    return {
        "hog": hog_features,
        "hog_hsv": np.concatenate([hog_features, hsv_features]).astype(np.float32),
        "all_handcrafted": np.concatenate(
            [hog_features, hsv_features, hu_features, geometry]
        ).astype(np.float32),
    }
