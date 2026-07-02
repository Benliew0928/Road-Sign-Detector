from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    detector: str = "ground_truth"

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Profile:
    name: str
    confidence: float
    fallback: bool
    fallback_max: int = 0


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def project_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain an images directory: {image_path}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def dataset_images(data_yaml: Path, split: str) -> list[Path]:
    payload: dict[str, Any] = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(payload["path"]))
    split_key = "val" if split == "validation" else split
    relative = payload.get(split_key)
    if not relative:
        raise ValueError(f"Dataset YAML has no split {split!r}")
    return sorted(
        path
        for path in (root / str(relative)).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_ground_truth_boxes(image_path: Path) -> list[Box]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    height, width = image.shape[:2]
    label_path = label_path_for(image_path)
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        coordinates = values[1:]
        if len(coordinates) < 6 or len(coordinates) % 2:
            continue
        xs = [coordinates[index] * width for index in range(0, len(coordinates), 2)]
        ys = [coordinates[index] * height for index in range(1, len(coordinates), 2)]
        boxes.append(Box(min(xs), min(ys), max(xs), max(ys)))
    return boxes


def read_negative_manifest(path: Path) -> list[Path]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [project_path(row["image_path"]) for row in csv.DictReader(handle)]


def box_iou(first: Box, second: Box) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.area + second.area - intersection
    return intersection / union if union else 0.0


def greedy_matches(ground_truth: list[Box], predictions: list[Box], iou_threshold: float) -> set[int]:
    candidates = sorted(
        (
            (
                box_iou(ground_truth[ground_truth_index], predictions[prediction_index]),
                ground_truth_index,
                prediction_index,
            )
            for ground_truth_index in range(len(ground_truth))
            for prediction_index in range(len(predictions))
        ),
        reverse=True,
    )
    matched_ground_truth: set[int] = set()
    matched_predictions: set[int] = set()
    for iou, ground_truth_index, prediction_index in candidates:
        if iou < iou_threshold:
            break
        if ground_truth_index in matched_ground_truth or prediction_index in matched_predictions:
            continue
        matched_ground_truth.add(ground_truth_index)
        matched_predictions.add(prediction_index)
    return matched_ground_truth


def parse_yolo_boxes(result: Any) -> list[Box]:
    boxes = result.boxes
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confidence = boxes.conf.cpu().numpy()
    parsed: list[Box] = []
    for coordinates, score in zip(xyxy, confidence, strict=True):
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        parsed.append(
            Box(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(score),
                detector="deep",
            )
        )
    return parsed


def baseline_boxes(image_path: Path) -> list[Box]:
    from roadsign_assist.detection.baseline_backend import BaselineSignDetector

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    detections = BaselineSignDetector().detect(image)
    return [
        Box(
            x1=detection.bbox.x1,
            y1=detection.bbox.y1,
            x2=detection.bbox.x2,
            y2=detection.bbox.y2,
            confidence=detection.confidence,
            detector=detection.detector,
        )
        for detection in detections
    ]


def profile_predictions(
    *,
    deep_boxes: list[Box],
    fallback_boxes: list[Box],
    profile: Profile,
) -> list[Box]:
    if deep_boxes or not profile.fallback:
        return deep_boxes
    return sorted(fallback_boxes, key=lambda box: box.confidence, reverse=True)[
        : profile.fallback_max
    ]


def summarize_positive(
    *,
    image_paths: list[Path],
    ground_truth_by_image: dict[Path, list[Box]],
    deep_by_threshold: dict[float, dict[Path, list[Box]]],
    baseline_by_image: dict[Path, list[Box]],
    profile: Profile,
    iou_threshold: float,
) -> dict[str, Any]:
    ground_truth_total = 0
    predicted_total = 0
    matched_total = 0
    images_with_detection = 0
    images_without_detection: list[str] = []
    for image_path in image_paths:
        ground_truth = ground_truth_by_image[image_path]
        predictions = profile_predictions(
            deep_boxes=deep_by_threshold[profile.confidence][image_path],
            fallback_boxes=baseline_by_image.get(image_path, []),
            profile=profile,
        )
        matches = greedy_matches(ground_truth, predictions, iou_threshold)
        ground_truth_total += len(ground_truth)
        predicted_total += len(predictions)
        matched_total += len(matches)
        if predictions:
            images_with_detection += 1
        else:
            images_without_detection.append(project_rel(image_path))
    false_positive_total = max(0, predicted_total - matched_total)
    return {
        "images": len(image_paths),
        "ground_truth_boxes": ground_truth_total,
        "predicted_boxes": predicted_total,
        "matched_boxes": matched_total,
        "recall_at_iou": matched_total / ground_truth_total if ground_truth_total else 0.0,
        "precision_at_iou": matched_total / predicted_total if predicted_total else 0.0,
        "false_positive_boxes": false_positive_total,
        "false_positive_boxes_per_image": false_positive_total / max(1, len(image_paths)),
        "images_with_detection": images_with_detection,
        "images_without_detection": images_without_detection,
    }


def summarize_negative(
    *,
    image_paths: list[Path],
    deep_by_threshold: dict[float, dict[Path, list[Box]]],
    baseline_by_image: dict[Path, list[Box]],
    profile: Profile,
) -> dict[str, Any]:
    false_boxes = 0
    images_with_false_boxes: list[dict[str, Any]] = []
    max_false_boxes = 0
    for image_path in image_paths:
        predictions = profile_predictions(
            deep_boxes=deep_by_threshold[profile.confidence][image_path],
            fallback_boxes=baseline_by_image.get(image_path, []),
            profile=profile,
        )
        count = len(predictions)
        false_boxes += count
        max_false_boxes = max(max_false_boxes, count)
        if count:
            images_with_false_boxes.append(
                {
                    "image_path": project_rel(image_path),
                    "false_boxes": count,
                    "detectors": sorted({box.detector for box in predictions}),
                    "max_confidence": max(box.confidence for box in predictions),
                }
            )
    return {
        "images": len(image_paths),
        "false_boxes": false_boxes,
        "false_boxes_per_100_images": false_boxes * 100.0 / max(1, len(image_paths)),
        "images_with_false_boxes": len(images_with_false_boxes),
        "image_false_positive_rate": len(images_with_false_boxes) / max(1, len(image_paths)),
        "max_false_boxes_on_one_image": max_false_boxes,
        "examples": sorted(
            images_with_false_boxes,
            key=lambda row: (row["false_boxes"], row["max_confidence"]),
            reverse=True,
        )[:20],
    }


def select_profile(rows: list[dict[str, Any]], current_name: str) -> str:
    current = next(row for row in rows if row["profile"] == current_name)
    minimum_recall = current["positive"]["recall_at_iou"] * 0.95
    eligible = [
        row
        for row in rows
        if row["positive"]["recall_at_iou"] >= minimum_recall
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["negative"]["false_boxes_per_100_images"],
            -row["positive"]["precision_at_iou"],
            -row["positive"]["recall_at_iou"],
        ),
    )
    return str(selected["profile"])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    model_path = project_path(args.model)
    data_yaml = project_path(args.data)
    negative_manifest = project_path(args.negative_manifest)
    output_path = project_path(args.output)
    csv_path = output_path.with_suffix(".csv")

    positive_images = dataset_images(data_yaml, args.split)
    negative_images = read_negative_manifest(negative_manifest)
    if args.limit_positive:
        positive_images = positive_images[: args.limit_positive]
    if args.limit_negative:
        negative_images = negative_images[: args.limit_negative]

    thresholds = tuple(float(value) for value in args.thresholds)
    profiles: list[Profile] = [
        Profile(name=f"deep_conf_{threshold:.2f}", confidence=threshold, fallback=False)
        for threshold in thresholds
    ]
    profiles.extend(
        [
            Profile(name="current_hybrid_conf_0.25_fallback3", confidence=0.25, fallback=True, fallback_max=3),
            Profile(name="hybrid_conf_0.35_fallback1", confidence=0.35, fallback=True, fallback_max=1),
        ]
    )
    threshold_values = sorted({profile.confidence for profile in profiles})
    all_images = sorted({*positive_images, *negative_images})

    model = YOLO(str(model_path), task="segment")
    deep_by_threshold: dict[float, dict[Path, list[Box]]] = {
        threshold: {} for threshold in threshold_values
    }
    for threshold in threshold_values:
        for image_path in all_images:
            results = model.predict(
                source=str(image_path),
                imgsz=args.image_size,
                conf=threshold,
                iou=args.nms_iou,
                device=args.device,
                verbose=False,
            )
            deep_by_threshold[threshold][image_path] = (
                parse_yolo_boxes(results[0]) if results else []
            )

    baseline_by_image = {image_path: baseline_boxes(image_path) for image_path in all_images}
    ground_truth_by_image = {
        image_path: read_ground_truth_boxes(image_path)
        for image_path in positive_images
    }

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        rows.append(
            {
                "profile": profile.name,
                "confidence": profile.confidence,
                "fallback": profile.fallback,
                "fallback_max": profile.fallback_max,
                "positive": summarize_positive(
                    image_paths=positive_images,
                    ground_truth_by_image=ground_truth_by_image,
                    deep_by_threshold=deep_by_threshold,
                    baseline_by_image=baseline_by_image,
                    profile=profile,
                    iou_threshold=args.match_iou,
                ),
                "negative": summarize_negative(
                    image_paths=negative_images,
                    deep_by_threshold=deep_by_threshold,
                    baseline_by_image=baseline_by_image,
                    profile=profile,
                ),
            }
        )

    selected_profile = select_profile(rows, "current_hybrid_conf_0.25_fallback3")
    report = {
        "schema_version": "1.0",
        "model": project_rel(model_path),
        "data": project_rel(data_yaml),
        "negative_manifest": project_rel(negative_manifest),
        "split": args.split,
        "image_size": args.image_size,
        "device": args.device,
        "match_iou": args.match_iou,
        "selection_rule": (
            "minimum negative false boxes among profiles retaining at least 95% "
            "of current hybrid positive recall"
        ),
        "selected_profile": selected_profile,
        "profiles": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "profile",
                "confidence",
                "fallback",
                "fallback_max",
                "positive_recall",
                "positive_precision",
                "positive_false_positive_boxes",
                "negative_false_boxes_per_100_images",
                "negative_image_false_positive_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "profile": row["profile"],
                    "confidence": row["confidence"],
                    "fallback": row["fallback"],
                    "fallback_max": row["fallback_max"],
                    "positive_recall": row["positive"]["recall_at_iou"],
                    "positive_precision": row["positive"]["precision_at_iou"],
                    "positive_false_positive_boxes": row["positive"]["false_positive_boxes"],
                    "negative_false_boxes_per_100_images": row["negative"][
                        "false_boxes_per_100_images"
                    ],
                    "negative_image_false_positive_rate": row["negative"][
                        "image_false_positive_rate"
                    ],
                }
            )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage F detector precision profiles.")
    parser.add_argument("--model", default="models/exported/experimental/emtd_segmenter_s30.onnx")
    parser.add_argument("--data", default="data/processed/emtd_segmentation/data.yaml")
    parser.add_argument(
        "--negative-manifest",
        default="data/processed/stage_f_negative_eval/manifest.csv",
    )
    parser.add_argument("--output", default="outputs/evaluation/stage_f_detector/profile_report.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.25, 0.35, 0.45, 0.55, 0.65],
    )
    parser.add_argument("--limit-positive", type=int)
    parser.add_argument("--limit-negative", type=int)
    return parser


def main() -> int:
    report = evaluate(build_parser().parse_args())
    selected = next(
        row for row in report["profiles"] if row["profile"] == report["selected_profile"]
    )
    print(
        "Stage F detector profiles complete: "
        f"selected={report['selected_profile']}, "
        f"positive_recall={selected['positive']['recall_at_iou']:.3f}, "
        f"negative_fp_per_100={selected['negative']['false_boxes_per_100_images']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
