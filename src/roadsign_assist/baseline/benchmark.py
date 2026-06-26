from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from roadsign_assist.baseline.features import (
    FloatFeatures,
    extract_hog,
    extract_hsv_histogram,
)
from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.baseline.training import train_and_evaluate
from roadsign_assist.paths import project_path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
FEATURE_SET_NAMES = ("hog", "hog_hsv", "all_handcrafted")
CLASSIFIER_NAMES = ("svm", "random_forest")


def _shape_features(crop: UInt8Image) -> FloatFeatures:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea, default=None)
    if contour is None:
        return np.zeros(13, dtype=np.float32)

    area = float(cv2.contourArea(contour))
    x, y, width, height = cv2.boundingRect(contour)
    del x, y
    bbox_area = max(1, width * height)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    perimeter = float(cv2.arcLength(contour, True))
    approximation = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    transformed_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    geometry = np.asarray(
        [
            area / max(1, crop.shape[0] * crop.shape[1]),
            width / max(1, height),
            area / bbox_area,
            area / hull_area if hull_area else 0.0,
            4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0,
            len(approximation) / 10.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([geometry, transformed_hu.astype(np.float32)])


def extract_crop_feature_sets(image: UInt8Image, size: int = 96) -> dict[str, FloatFeatures]:
    crop = cast(
        UInt8Image,
        cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA),
    )
    hog = extract_hog(crop)
    hsv = extract_hsv_histogram(crop)
    shape = _shape_features(crop)
    return {
        "hog": hog,
        "hog_hsv": np.concatenate([hog, hsv]).astype(np.float32),
        "all_handcrafted": np.concatenate([hog, hsv, shape]).astype(np.float32),
    }


def _validate_dataset(root: Path, *, allow_unreviewed_experiment: bool) -> dict[str, Any]:
    metadata_path = root / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("coursework_images_included", -1)) != 0:
        raise ValueError("Coursework acceptance images cannot be used for baseline training")
    status = str(metadata.get("annotation_status", ""))
    if status != "approved" and not allow_unreviewed_experiment:
        raise ValueError(
            f"Baseline data status is {status!r}; use the experimental override for unreviewed data"
        )
    return metadata


def _load_split(
    root: Path,
    split: str,
) -> tuple[dict[str, NDArray[np.float32]], NDArray[np.str_], list[str]]:
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    feature_rows: dict[str, list[FloatFeatures]] = {name: [] for name in FEATURE_SET_NAMES}
    labels: list[str] = []
    source_paths: list[str] = []
    for label_directory in sorted(path for path in split_root.iterdir() if path.is_dir()):
        for path in sorted(label_directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cast(UInt8Image | None, cv2.imread(str(path), cv2.IMREAD_COLOR))
            if image is None:
                raise ValueError(f"Unable to decode classification crop: {path}")
            values = extract_crop_feature_sets(image)
            for name in FEATURE_SET_NAMES:
                feature_rows[name].append(values[name])
            labels.append(label_directory.name)
            source_paths.append(path.relative_to(root).as_posix())
    if not labels:
        raise ValueError(f"No images found in split: {split_root}")
    matrices = {name: np.stack(rows).astype(np.float32) for name, rows in feature_rows.items()}
    return matrices, np.asarray(labels, dtype=np.str_), source_paths


def run_baseline_classifier_benchmark(
    data_root: str | Path = "data/processed/emtd_classification",
    output_root: str | Path = "outputs/evaluation/baseline_classifiers",
    model_root: str | Path = "models/baseline",
    *,
    allow_unreviewed_experiment: bool = False,
) -> dict[str, Any]:
    data = project_path(data_root)
    output = project_path(output_root)
    models = project_path(model_root)
    metadata = _validate_dataset(
        data,
        allow_unreviewed_experiment=allow_unreviewed_experiment,
    )
    extraction_started = time.perf_counter()
    train_features, train_labels, train_paths = _load_split(data, "train")
    test_features, test_labels, test_paths = _load_split(data, "test")
    extraction_ms = (time.perf_counter() - extraction_started) * 1000

    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for classifier_name in CLASSIFIER_NAMES:
        for feature_set in FEATURE_SET_NAMES:
            run_name = f"{classifier_name}__{feature_set}"
            metrics = train_and_evaluate(
                classifier_name=classifier_name,
                feature_set=feature_set,
                x_train=train_features[feature_set],
                y_train=train_labels,
                x_test=test_features[feature_set],
                y_test=test_labels,
                model_path=models / f"{run_name}.joblib",
                metrics_path=output / f"{run_name}.json",
            )
            rows.append(
                {
                    **asdict(metrics),
                    "feature_count": int(train_features[feature_set].shape[1]),
                    "model_path": (models / f"{run_name}.joblib")
                    .relative_to(project_path("."))
                    .as_posix(),
                }
            )

    rows.sort(key=lambda row: cast(float, row["macro_f1"]), reverse=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "experimental": str(metadata.get("annotation_status")) != "approved",
        "annotation_status": metadata.get("annotation_status"),
        "coursework_images_included": metadata.get("coursework_images_included"),
        "split_source": "frozen_emtd_classification_folders",
        "train_count": len(train_labels),
        "test_count": len(test_labels),
        "feature_extraction_ms": extraction_ms,
        "feature_source": "image_pixels_only",
        "path_or_filename_features_used": False,
        "train_paths_sha256": _paths_digest(train_paths),
        "test_paths_sha256": _paths_digest(test_paths),
        "best_run": rows[0],
        "runs": rows,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def _paths_digest(paths: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in paths:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
