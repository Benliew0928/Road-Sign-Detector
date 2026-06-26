from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import json
import statistics
import time
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from roadsign_assist.paths import project_path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _dataset_images(data_yaml: Path, split: str) -> list[Path]:
    payload: dict[str, Any] = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(str(payload["path"]))
    split_key = "val" if split == "validation" else split
    relative = payload.get(split_key)
    if relative is None:
        raise ValueError(f"Dataset YAML has no {split_key!r} split")
    images_root = dataset_root / str(relative)
    images = sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found for {split!r}: {images_root}")
    return images


def percentile(values: list[float], percentile_value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def benchmark_detector_runtime(
    *,
    model_path: str | Path,
    data_yaml: str | Path,
    output_path: str | Path,
    split: str = "test",
    image_size: int = 512,
    confidence: float = 0.25,
    device: str = "cpu",
    limit: int | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model_file = project_path(model_path)
    data_file = project_path(data_yaml)
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    images = _dataset_images(data_file, split)
    if limit is not None:
        images = images[: max(1, limit)]

    model = YOLO(str(model_file), task="segment")
    model.predict(
        source=str(images[0]),
        imgsz=image_size,
        conf=confidence,
        device=device,
        verbose=False,
    )
    wall_times: list[float] = []
    inference_times: list[float] = []
    postprocess_times: list[float] = []
    images_with_masks = 0
    for image in images:
        started = time.perf_counter()
        results = model.predict(
            source=str(image),
            imgsz=image_size,
            conf=confidence,
            device=device,
            verbose=False,
        )
        wall_times.append((time.perf_counter() - started) * 1000)
        result = results[0]
        inference = result.speed.get("inference")
        postprocess = result.speed.get("postprocess")
        inference_times.append(float(inference if inference is not None else 0.0))
        postprocess_times.append(float(postprocess if postprocess is not None else 0.0))
        if result.masks is not None and len(result.masks) > 0:
            images_with_masks += 1

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "model": str(model_file.relative_to(project_path("."))),
        "dataset": str(data_file.relative_to(project_path("."))),
        "split": split,
        "device": device,
        "image_size": image_size,
        "confidence": confidence,
        "images": len(images),
        "images_with_masks": images_with_masks,
        "wall_latency_ms": {
            "mean": statistics.fmean(wall_times),
            "median": statistics.median(wall_times),
            "p95": percentile(wall_times, 95),
            "maximum": max(wall_times),
        },
        "model_inference_ms": {
            "mean": statistics.fmean(inference_times),
            "p95": percentile(inference_times, 95),
            "maximum": max(inference_times),
        },
        "postprocess_ms": {
            "mean": statistics.fmean(postprocess_times),
            "p95": percentile(postprocess_times, 95),
            "maximum": max(postprocess_times),
        },
    }
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def tune_detector_thresholds(
    *,
    model_path: str | Path,
    data_yaml: str | Path,
    output_path: str | Path,
    thresholds: tuple[float, ...] = (0.10, 0.20, 0.25, 0.35, 0.50),
    split: str = "val",
    image_size: int = 512,
    device: str = "0",
) -> dict[str, Any]:
    from ultralytics import YOLO

    model_file = project_path(model_path)
    data_file = project_path(data_yaml)
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    model = YOLO(str(model_file), task="segment")
    rows: list[dict[str, float]] = []
    evaluation_root = project_path(output_path).parent / "threshold_runs"
    for threshold in thresholds:
        metrics = model.val(
            data=str(data_file),
            split=split,
            imgsz=image_size,
            conf=threshold,
            device=device,
            project=str(evaluation_root),
            name=f"conf_{threshold:.2f}".replace(".", "_"),
            exist_ok=True,
            plots=False,
            verbose=False,
        )
        values = {
            str(key): float(value)
            for key, value in dict(metrics.results_dict).items()
            if isinstance(value, Real)
        }
        mask_precision = values.get("metrics/precision(M)", 0.0)
        mask_recall = values.get("metrics/recall(M)", 0.0)
        mask_f1 = (
            2.0 * mask_precision * mask_recall / (mask_precision + mask_recall)
            if mask_precision + mask_recall
            else 0.0
        )
        rows.append(
            {
                "confidence": threshold,
                "box_precision": values.get("metrics/precision(B)", 0.0),
                "box_recall": values.get("metrics/recall(B)", 0.0),
                "box_map50": values.get("metrics/mAP50(B)", 0.0),
                "mask_precision": mask_precision,
                "mask_recall": mask_recall,
                "mask_map50": values.get("metrics/mAP50(M)", 0.0),
                "mask_f1": mask_f1,
            }
        )
    selected = max(rows, key=lambda row: (row["mask_f1"], row["mask_recall"]))
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "model": str(model_file.relative_to(project_path("."))),
        "dataset": str(data_file.relative_to(project_path("."))),
        "split": split,
        "device": device,
        "image_size": image_size,
        "selection_rule": "maximum mask F1, then mask recall",
        "selected_confidence": selected["confidence"],
        "selected": selected,
        "runs": rows,
    }
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_detector_recall_slices(
    *,
    model_path: str | Path,
    data_yaml: str | Path,
    output_path: str | Path,
    split: str = "test",
    image_size: int = 640,
    confidence: float = 0.25,
    device: str = "0",
    match_iou: float = 0.50,
    small_area_ratio: float = 0.01,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model_file = project_path(model_path)
    data_file = project_path(data_yaml)
    images = _dataset_images(data_file, split)
    model = YOLO(str(model_file), task="segment")
    slice_counts = {
        "all": {"ground_truth": 0, "matched": 0},
        "small": {"ground_truth": 0, "matched": 0},
        "non_small": {"ground_truth": 0, "matched": 0},
    }

    for image_path in images:
        with Image.open(image_path) as source:
            width, height = source.size
        label_path = _label_path(image_path)
        ground_truth = _read_ground_truth_boxes(label_path, width, height)
        results = model.predict(
            source=str(image_path),
            imgsz=image_size,
            conf=confidence,
            device=device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None:
            predicted = np.empty((0, 4), dtype=np.float64)
        else:
            coordinates: Any = boxes.xyxy
            if hasattr(coordinates, "cpu"):
                coordinates = coordinates.cpu()
            if hasattr(coordinates, "numpy"):
                coordinates = coordinates.numpy()
            predicted = np.asarray(coordinates, dtype=np.float64)
        matches = greedy_box_matches(ground_truth, predicted, match_iou)
        for index, box in enumerate(ground_truth):
            area_ratio = _box_area(box) / max(1.0, width * height)
            bucket = "small" if area_ratio <= small_area_ratio else "non_small"
            slice_counts["all"]["ground_truth"] += 1
            slice_counts[bucket]["ground_truth"] += 1
            if index in matches:
                slice_counts["all"]["matched"] += 1
                slice_counts[bucket]["matched"] += 1

    slices = {
        name: {
            **counts,
            "recall_at_iou": (
                counts["matched"] / counts["ground_truth"] if counts["ground_truth"] else None
            ),
        }
        for name, counts in slice_counts.items()
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "model": str(model_file.relative_to(project_path("."))),
        "dataset": str(data_file.relative_to(project_path("."))),
        "split": split,
        "device": device,
        "image_size": image_size,
        "confidence": confidence,
        "match_iou": match_iou,
        "small_definition": f"ground-truth box area <= {small_area_ratio:.4f} of image area",
        "images": len(images),
        "slices": slices,
    }
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain an images directory: {image_path}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _read_ground_truth_boxes(
    label_path: Path,
    width: int,
    height: int,
) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
    boxes: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        coordinates = values[1:]
        if len(coordinates) < 6 or len(coordinates) % 2:
            continue
        x_values = np.asarray(coordinates[0::2], dtype=np.float64) * width
        y_values = np.asarray(coordinates[1::2], dtype=np.float64) * height
        boxes.append(
            np.asarray(
                [x_values.min(), y_values.min(), x_values.max(), y_values.max()],
                dtype=np.float64,
            )
        )
    return boxes


def _box_area(box: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _box_iou(
    first: np.ndarray[Any, np.dtype[np.float64]],
    second: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union else 0.0


def greedy_box_matches(
    ground_truth: list[np.ndarray[Any, np.dtype[np.float64]]],
    predicted: np.ndarray[Any, np.dtype[np.float64]],
    threshold: float,
) -> set[int]:
    candidates = sorted(
        (
            (
                _box_iou(ground_truth[ground_truth_index], predicted[prediction_index]),
                ground_truth_index,
                prediction_index,
            )
            for ground_truth_index in range(len(ground_truth))
            for prediction_index in range(len(predicted))
        ),
        reverse=True,
    )
    matched_ground_truth: set[int] = set()
    matched_predictions: set[int] = set()
    for iou, ground_truth_index, prediction_index in candidates:
        if iou < threshold:
            break
        if ground_truth_index in matched_ground_truth or prediction_index in matched_predictions:
            continue
        matched_ground_truth.add(ground_truth_index)
        matched_predictions.add(prediction_index)
    return matched_ground_truth
