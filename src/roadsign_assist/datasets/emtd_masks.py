from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from roadsign_assist.paths import project_path


@dataclass(frozen=True)
class MaskGenerationConfig:
    dataset_manifest: Path = Path("data/manifests/dataset.csv")
    annotation_csv: Path = Path("data/annotations/emtd_boxes.csv")
    split_root: Path = Path("data/splits")
    output_root: Path = Path("data/processed/emtd_segmentation")
    model: str = "models/pretrained/sam2.1_s.pt"
    device: str = "0"
    limit: int | None = None
    minimum_box_iou: float = 0.20
    minimum_area_ratio: float = 0.04
    maximum_area_ratio: float = 2.50
    accepted_preview_limit: int = 40


def polygon_area(points: np.ndarray[Any, np.dtype[np.float32]]) -> float:
    if len(points) < 3:
        return 0.0
    x_values = points[:, 0]
    y_values = points[:, 1]
    return float(
        0.5 * abs(np.dot(x_values, np.roll(y_values, 1)) - np.dot(y_values, np.roll(x_values, 1)))
    )


def polygon_box_iou(
    points: np.ndarray[Any, np.dtype[np.float32]],
    box: tuple[float, float, float, float],
) -> float:
    if not len(points):
        return 0.0
    polygon_box = (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )
    left = max(polygon_box[0], box[0])
    top = max(polygon_box[1], box[1])
    right = min(polygon_box[2], box[2])
    bottom = min(polygon_box[3], box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    polygon_box_area = max(0.0, polygon_box[2] - polygon_box[0]) * max(
        0.0, polygon_box[3] - polygon_box[1]
    )
    source_box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    union = polygon_box_area + source_box_area - intersection
    return intersection / union if union else 0.0


def validate_mask_polygon(
    points: np.ndarray[Any, np.dtype[np.float32]],
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    minimum_box_iou: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
) -> dict[str, float]:
    if len(points) < 3:
        raise ValueError("Mask polygon has fewer than three points")
    if not np.isfinite(points).all():
        raise ValueError("Mask polygon contains non-finite coordinates")
    if (
        (points[:, 0] < 0).any()
        or (points[:, 1] < 0).any()
        or (points[:, 0] > width).any()
        or (points[:, 1] > height).any()
    ):
        raise ValueError("Mask polygon exceeds image bounds")
    area = polygon_area(points)
    box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    area_ratio = area / box_area
    box_iou = polygon_box_iou(points, box)
    if area_ratio < minimum_area_ratio or area_ratio > maximum_area_ratio:
        raise ValueError(f"Mask area ratio {area_ratio:.3f} is outside QA bounds")
    if box_iou < minimum_box_iou:
        raise ValueError(f"Mask bounding-box IoU {box_iou:.3f} is below QA threshold")
    return {
        "polygon_area": area,
        "source_box_area": box_area,
        "area_ratio": area_ratio,
        "box_iou": box_iou,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _copy_processed_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_overlay(
    source: Path,
    destination: Path,
    boxes: list[list[float]],
    polygons: list[np.ndarray[Any, np.dtype[np.float32]]],
) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 300))
    for box in boxes:
        draw.rectangle(tuple(box), outline=(255, 196, 0), width=line_width)
    for polygon in polygons:
        points = [(float(point[0]), float(point[1])) for point in polygon]
        if len(points) >= 3:
            draw.line([*points, points[0]], fill=(255, 45, 85), width=line_width)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((1600, 1600))
    image.save(destination, quality=90)


def generate_emtd_masks(config: MaskGenerationConfig) -> dict[str, Any]:
    from ultralytics import SAM

    manifest_rows = _read_csv(project_path(config.dataset_manifest))
    annotations = _read_csv(project_path(config.annotation_csv))
    if config.limit is not None and config.limit < 1:
        raise ValueError("Mask generation limit must be positive")
    annotations_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in annotations:
        annotations_by_sample[row["sample_id"]].append(row)
    split_by_sample: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for row in _read_csv(project_path(config.split_root) / f"{split}.csv"):
            split_by_sample[row["sample_id"]] = split

    selected = sorted(manifest_rows, key=lambda row: row["sample_id"])
    if config.limit is not None:
        selected = selected[: config.limit]
    output_root = project_path(config.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    model = SAM(config.model)
    qa_rows: list[dict[str, object]] = []
    accepted_images = 0
    accepted_instances = 0
    failed_images = 0
    for completed, row in enumerate(selected, start=1):
        sample_id = row["sample_id"]
        source = project_path(row["path"])
        split = split_by_sample[sample_id]
        sample_annotations = annotations_by_sample[sample_id]
        boxes = [
            [
                float(annotation["xmin"]),
                float(annotation["ymin"]),
                float(annotation["xmax"]),
                float(annotation["ymax"]),
            ]
            for annotation in sample_annotations
        ]
        with Image.open(source) as image:
            width, height = image.size
        results = model.predict(
            source=str(source),
            bboxes=boxes,
            device=config.device,
            verbose=False,
            save=False,
        )
        masks = results[0].masks if results else None
        polygons = (
            [np.asarray(polygon, dtype=np.float32) for polygon in masks.xy]
            if masks is not None
            else []
        )
        if len(polygons) != len(boxes):
            failed_images += 1
            _write_overlay(
                source,
                output_root / "review" / "failed" / f"{sample_id}.jpg",
                boxes,
                polygons,
            )
            qa_rows.append(
                {
                    "sample_id": sample_id,
                    "instance_id": "",
                    "status": "failed",
                    "reason": (f"expected {len(boxes)} masks, received {len(polygons)}"),
                    "area_ratio": "",
                    "box_iou": "",
                }
            )
            continue

        label_lines: list[str] = []
        image_qa: list[dict[str, object]] = []
        try:
            for annotation, box_values, polygon in zip(
                sample_annotations,
                boxes,
                polygons,
                strict=True,
            ):
                points = polygon
                box = (
                    box_values[0],
                    box_values[1],
                    box_values[2],
                    box_values[3],
                )
                metrics = validate_mask_polygon(
                    points,
                    box,
                    width,
                    height,
                    minimum_box_iou=config.minimum_box_iou,
                    minimum_area_ratio=config.minimum_area_ratio,
                    maximum_area_ratio=config.maximum_area_ratio,
                )
                normalized = points.copy()
                normalized[:, 0] /= width
                normalized[:, 1] /= height
                coordinates = " ".join(f"{value:.8f}" for value in normalized.reshape(-1))
                label_lines.append(f"0 {coordinates}")
                image_qa.append(
                    {
                        "sample_id": sample_id,
                        "instance_id": annotation["instance_id"],
                        "status": "accepted_unreviewed",
                        "reason": "",
                        "area_ratio": metrics["area_ratio"],
                        "box_iou": metrics["box_iou"],
                    }
                )
        except ValueError as exc:
            failed_images += 1
            _write_overlay(
                source,
                output_root / "review" / "failed" / f"{sample_id}.jpg",
                boxes,
                polygons,
            )
            qa_rows.append(
                {
                    "sample_id": sample_id,
                    "instance_id": "",
                    "status": "failed",
                    "reason": str(exc),
                    "area_ratio": "",
                    "box_iou": "",
                }
            )
            continue

        destination_name = f"{sample_id}{source.suffix.lower()}"
        _copy_processed_image(
            source,
            output_root / "images" / split / destination_name,
        )
        label_path = output_root / "labels" / split / f"{sample_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(label_lines) + "\n", encoding="ascii")
        qa_rows.extend(image_qa)
        if accepted_images < config.accepted_preview_limit:
            _write_overlay(
                source,
                output_root / "review" / "accepted" / f"{sample_id}.jpg",
                boxes,
                polygons,
            )
        accepted_images += 1
        accepted_instances += len(label_lines)
        print(f"[{completed}/{len(selected)}] {sample_id}: {len(label_lines)} masks accepted")

    qa_path = output_root / "mask_qa.csv"
    with qa_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_id",
            "instance_id",
            "status",
            "reason",
            "area_ratio",
            "box_iou",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qa_rows)
    data_yaml = {
        "path": output_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "names": {0: "traffic_sign"},
    }
    (output_root / "data.yaml").write_text(
        json.dumps(data_yaml, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "1.0",
        "source_id": "emtd_zenodo_1217105",
        "mask_generator": config.model,
        "annotation_status": "sam2_box_prompt_unreviewed",
        "training_scope": "experimental_only",
        "coursework_images_included": 0,
        "selected_images": len(selected),
        "accepted_images": accepted_images,
        "failed_images": failed_images,
        "accepted_instances": accepted_instances,
        "minimum_box_iou": config.minimum_box_iou,
        "minimum_area_ratio": config.minimum_area_ratio,
        "maximum_area_ratio": config.maximum_area_ratio,
        "accepted_preview_limit": config.accepted_preview_limit,
        "manual_review_required": True,
    }
    (output_root / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata
