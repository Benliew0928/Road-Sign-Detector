from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, cast

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from roadsign_assist.baseline.models import UInt8Image  # noqa: E402
from roadsign_assist.config import load_yaml  # noqa: E402
from roadsign_assist.inference.engine import InferenceEngine  # noqa: E402
from roadsign_assist.paths import project_path  # noqa: E402

COMMON_DEMO_LABELS = (
    "maximum_speed",
    "stop",
    "give_way",
    "no_entry",
    "pedestrian_crossing",
    "school_zone",
    "children_crossing",
    "no_left_turn",
    "no_right_turn",
    "no_u_turn",
    "roundabout_mandatory",
    "straight_ahead",
    "turn_left",
    "turn_right",
    "height_restriction",
    "width_restriction",
    "no_parking",
    "no_stopping",
)


def project_rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
    value = Path(path)
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    try:
        return value.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(value)


def read_json(path: str | Path) -> dict[str, Any] | None:
    value = project_path(path)
    if not value.is_file():
        return None
    return json.loads(value.read_text(encoding="utf-8"))


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(quantile * (len(ordered) - 1)))
    return ordered[index]


def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    with project_path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def choose_primary_event(events: list[Any]) -> Any | None:
    if not events:
        return None
    return sorted(
        events,
        key=lambda event: (bool(event.stable), float(event.confidence)),
        reverse=True,
    )[0]


def matching_numeric_values(events: list[Any]) -> list[float]:
    values: list[float] = []
    for event in events:
        if event.action.target_speed_kmh is not None:
            values.append(float(event.action.target_speed_kmh))
        elif event.action.restriction_value is not None:
            values.append(float(event.action.restriction_value))
        elif event.ocr.numeric_value is not None:
            values.append(float(event.ocr.numeric_value))
    return values


def numeric_matches(expected: float | None, values: list[float]) -> bool | None:
    if expected is None:
        return None
    return any(abs(value - expected) <= 1.0 for value in values)


def summarize_events(events: list[Any]) -> str:
    return ";".join(
        f"{event.semantic_sign_id}:{float(event.confidence):.3f}" for event in events[:5]
    )


def evaluate_assignment_pipeline(
    *,
    engine: InferenceEngine,
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    per_class_total: Counter[str] = Counter()
    per_class_any_match: Counter[str] = Counter()
    per_class_primary_match: Counter[str] = Counter()
    runtimes: list[float] = []

    for index, row in enumerate(rows, start=1):
        image_path = project_path(row["path"])
        raw_image = cv2.imread(str(image_path))
        if raw_image is None:
            raise ValueError(f"Unable to read assignment image: {row['path']}")
        image = cast(UInt8Image, raw_image)
        expected = row["semantic_sign_id"].strip()
        expected_parameter = parse_optional_float(row.get("parameter_value"))

        started = time.perf_counter()
        result = engine.new_session().process_frame(image, assume_stable=True)
        runtime_ms = (time.perf_counter() - started) * 1000
        runtimes.append(runtime_ms)

        events = sorted(result.events, key=lambda event: float(event.confidence), reverse=True)
        primary = choose_primary_event(events)
        expected_events = [event for event in events if event.semantic_sign_id == expected]
        any_match = bool(expected_events)
        primary_match = bool(primary and primary.semantic_sign_id == expected)
        values = matching_numeric_values(expected_events)
        numeric_match = numeric_matches(expected_parameter, values)

        per_class_total[expected] += 1
        if any_match:
            per_class_any_match[expected] += 1
        if primary_match:
            per_class_primary_match[expected] += 1

        predictions.append(
            {
                "external_test_id": row.get("external_test_id", ""),
                "path": row["path"],
                "expected": expected,
                "expected_parameter": expected_parameter if expected_parameter is not None else "",
                "event_count": len(events),
                "primary_semantic": primary.semantic_sign_id if primary else "",
                "primary_confidence": round(float(primary.confidence), 6) if primary else "",
                "expected_in_events": any_match,
                "primary_correct": primary_match,
                "numeric_match": "" if numeric_match is None else numeric_match,
                "numeric_values": "|".join(f"{value:g}" for value in values),
                "advisory_headline": primary.advisory.headline.en if primary else "",
                "action_code": enum_value(primary.action.code) if primary else "",
                "safe_to_announce": primary.advisory.safe_to_announce if primary else "",
                "runtime_ms": round(runtime_ms, 3),
                "under_two_seconds": runtime_ms < 2000,
                "events": summarize_events(events),
            }
        )
        print(
            f"[{index:02d}/{len(rows)}] {row.get('external_test_id', '')}: "
            f"{len(events)} events, expected_match={any_match}, {runtime_ms:.1f} ms"
        )

    numeric_rows = [row for row in predictions if row["expected_parameter"] != ""]
    numeric_scored = [row for row in numeric_rows if row["numeric_match"] != ""]
    per_class = {
        label: {
            "samples": per_class_total[label],
            "expected_in_events": per_class_any_match[label],
            "expected_event_recall": per_class_any_match[label] / per_class_total[label],
            "primary_correct": per_class_primary_match[label],
            "primary_accuracy": per_class_primary_match[label] / per_class_total[label],
        }
        for label in sorted(per_class_total)
    }
    metrics = {
        "samples": len(predictions),
        "images_with_events": sum(int(row["event_count"]) > 0 for row in predictions),
        "images_with_events_rate": sum(int(row["event_count"]) > 0 for row in predictions)
        / max(1, len(predictions)),
        "expected_in_events": sum(bool(row["expected_in_events"]) for row in predictions),
        "expected_in_events_rate": sum(bool(row["expected_in_events"]) for row in predictions)
        / max(1, len(predictions)),
        "primary_correct": sum(bool(row["primary_correct"]) for row in predictions),
        "primary_accuracy": sum(bool(row["primary_correct"]) for row in predictions)
        / max(1, len(predictions)),
        "macro_expected_event_recall": mean(
            value["expected_event_recall"] for value in per_class.values()
        )
        if per_class
        else None,
        "runtime_ms": {
            "mean": mean(runtimes) if runtimes else None,
            "p95": percentile(runtimes, 0.95),
            "maximum": max(runtimes) if runtimes else None,
        },
        "under_two_seconds": sum(bool(row["under_two_seconds"]) for row in predictions),
        "under_two_seconds_rate": sum(bool(row["under_two_seconds"]) for row in predictions)
        / max(1, len(predictions)),
        "numeric_parameter_rows": len(numeric_rows),
        "numeric_scored_rows": len(numeric_scored),
        "numeric_accuracy": (
            sum(row["numeric_match"] is True for row in numeric_scored) / len(numeric_scored)
            if numeric_scored
            else None
        ),
        "per_class": per_class,
    }
    return predictions, metrics


def read_image_rgb_or_bgr(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        background = np.full_like(bgr, 255.0)
        return (bgr * alpha + background * (1.0 - alpha)).astype(np.uint8)
    return image[:, :, :3]


def letterbox_frame(
    image: np.ndarray[Any, np.dtype[np.uint8]],
    *,
    width: int = 960,
    height: int = 540,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    frame = np.full((height, width, 3), (18, 31, 28), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x0 = (width - resized.shape[1]) // 2
    y0 = (height - resized.shape[0]) // 2
    frame[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return frame


def add_demo_caption(
    frame: np.ndarray[Any, np.dtype[np.uint8]],
    label: str,
    index: int,
    total: int,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    captioned = frame.copy()
    height = captioned.shape[0]
    cv2.putText(
        captioned,
        f"RoadSign Assist fallback {index}/{total}",
        (34, height - 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (214, 242, 232),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        captioned,
        label.replace("_", " "),
        (34, height - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (142, 230, 190),
        2,
        cv2.LINE_AA,
    )
    return captioned


def detector_demo_candidates(source_root: Path) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(source_root.glob(pattern)))
    return paths


def create_and_score_fallback_video(
    *,
    engine: InferenceEngine,
    source_root: Path,
    output_path: Path,
    limit: int = 14,
    fps: int = 2,
) -> dict[str, Any]:
    selected: list[tuple[Path, np.ndarray[Any, np.dtype[np.uint8]], Any]] = []
    scanned = 0
    for image_path in detector_demo_candidates(source_root):
        scanned += 1
        image = read_image_rgb_or_bgr(image_path)
        frame = letterbox_frame(image)
        result = engine.new_session().process_frame(cast(UInt8Image, frame), assume_stable=True)
        if result.events:
            selected.append((image_path, frame, result))
        if len(selected) >= limit:
            break
    if not selected:
        raise RuntimeError(f"No detector-positive fallback frames found in {source_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (960, 540),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create fallback demo video: {output_path}")

    frame_results: list[dict[str, Any]] = []
    for index, (image_path, frame, result) in enumerate(selected, start=1):
        caption = add_demo_caption(frame, image_path.stem, index, len(selected))
        for _ in range(fps):
            writer.write(caption)
        predictions = sorted({event.semantic_sign_id for event in result.events})
        frame_results.append(
            {
                "source_image": project_rel(image_path),
                "events": len(result.events),
                "predictions": predictions,
                "best_confidence": max(float(event.confidence) for event in result.events),
                "latency_ms": result.latency_ms,
            }
        )
    writer.release()

    runtimes = [float(row["latency_ms"]) for row in frame_results]
    return {
        "video": project_rel(output_path),
        "source": project_rel(source_root),
        "source_note": "controlled fallback video from detector validation full frames; not newly collected road footage",
        "source_frames_scanned": scanned,
        "fps": fps,
        "frames_written": len(selected) * fps,
        "unique_frames": len(selected),
        "smoke_frames_with_events": sum(int(row["events"]) > 0 for row in frame_results),
        "smoke_event_rate": sum(int(row["events"]) > 0 for row in frame_results)
        / max(1, len(frame_results)),
        "runtime_ms": {
            "mean": mean(runtimes) if runtimes else None,
            "p95": percentile(runtimes, 0.95),
            "maximum": max(runtimes) if runtimes else None,
        },
        "frames": frame_results,
    }


def summarize_stage_artifacts() -> dict[str, Any]:
    detector_selection = read_json("outputs/evaluation/stage_f_detector/stage_f_selection_report.json")
    detector_slices = read_json("outputs/evaluation/emtd_segmenter_s30/recall_slices.json")
    classifier_external = read_json(
        "outputs/evaluation/stage_g_classifier/"
        "stage_e_current_efficientnet_v2_s_embedding_q97_assignment_report.json"
    )
    classifier_embedding = read_json(
        "outputs/evaluation/stage_g_classifier/"
        "stage_e_current_efficientnet_v2_s_embedding_q97_report.json"
    )
    training_metrics = read_json("outputs/training/stage_e_current_efficientnet_v2_s/metrics.json")
    ocr_smoke = read_json("outputs/evaluation/ocr_smoke/metrics.json")
    split_audit = read_json("outputs/audit/final_split_audit.json")

    selected_detector = (detector_selection or {}).get("selected_metrics", {})
    selected_classifier_test = (classifier_embedding or {}).get("test", {})
    return {
        "detector": {
            "source": "outputs/evaluation/stage_f_detector/stage_f_selection_report.json",
            "selected_profile": (detector_selection or {}).get("selected_profile"),
            "positive_recall_at_iou_0_50": selected_detector.get(
                "positive_recall_at_iou_0_50"
            ),
            "positive_precision_at_iou_0_50": selected_detector.get(
                "positive_precision_at_iou_0_50"
            ),
            "negative_false_boxes_per_100_images": selected_detector.get(
                "negative_false_boxes_per_100_images"
            ),
            "negative_image_false_positive_rate": selected_detector.get(
                "negative_image_false_positive_rate"
            ),
            "small_sign_recall_at_iou_0_50": (
                (detector_slices or {}).get("slices", {}).get("small", {}).get("recall_at_iou")
            ),
        },
        "classifier": {
            "source": (
                "outputs/evaluation/stage_g_classifier/"
                "stage_e_current_efficientnet_v2_s_embedding_q97_report.json"
            ),
            "frozen_test_accuracy": selected_classifier_test.get("accuracy"),
            "frozen_test_accepted_accuracy": selected_classifier_test.get("accepted_accuracy"),
            "frozen_test_coverage": selected_classifier_test.get("coverage"),
            "frozen_test_macro_f1": (training_metrics or {}).get("macro_f1_observed"),
            "assignment_external_raw_accuracy": (classifier_external or {}).get("raw_accuracy"),
            "assignment_external_selective_accuracy": (classifier_external or {}).get(
                "selective_accuracy"
            ),
            "assignment_external_selective_coverage": (classifier_external or {}).get(
                "selective_coverage"
            ),
            "assignment_external_unknown_rejection_rate": (
                None
                if classifier_external is None
                else 1.0 - float(classifier_external.get("selective_coverage", 0.0))
            ),
            "unknown_auroc": (classifier_embedding or {}).get("unknown_auroc"),
            "unknown_auroc_reason": (classifier_embedding or {}).get("unknown_auroc_reason"),
        },
        "ocr": {
            "source": "outputs/evaluation/ocr_smoke/metrics.json",
            "synthetic_exact_match_rate": (ocr_smoke or {}).get("exact_match_rate"),
            "warm_mean_latency_ms": (ocr_smoke or {}).get("warm_mean_latency_ms"),
            "claims_real_road_accuracy": (ocr_smoke or {}).get("claims_real_road_accuracy"),
        },
        "dataset": {
            "source": "outputs/audit/final_split_audit.json",
            "samples": (split_audit or {}).get("samples"),
            "labels": (split_audit or {}).get("labels"),
            "assignment_external_test_samples": (split_audit or {}).get(
                "assignment_external_test_samples"
            ),
            "coursework_images_in_training": (split_audit or {}).get(
                "coursework_images_in_training"
            ),
            "completion_checks": (split_audit or {}).get("completion_checks"),
        },
    }


def summarize_common_signs(assignment_metrics: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    per_class = assignment_metrics.get("per_class", {})
    observed = {
        label: per_class[label]
        for label in COMMON_DEMO_LABELS
        if isinstance(per_class, dict) and label in per_class
    }
    return {
        "selected_labels": list(COMMON_DEMO_LABELS),
        "assignment_observed_labels": observed,
        "assignment_observed_count": len(observed),
        "classifier_frozen_test_accuracy": artifacts["classifier"].get("frozen_test_accuracy"),
        "classifier_frozen_test_macro_f1": artifacts["classifier"].get("frozen_test_macro_f1"),
        "note": (
            "Common-sign coverage uses the frozen Stage E/G classifier test plus any "
            "assignment external rows for these labels. Real local route footage is still "
            "an owner-side demo check."
        ),
    }


def gate_status(report: dict[str, Any]) -> dict[str, Any]:
    assignment = report["assignment_pipeline"]
    common = report["common_malaysian_signs"]
    fallback = report["fallback_video"]
    artifacts = report["artifact_metrics"]
    assignment_rate = float(assignment["expected_in_events_rate"])
    classifier_accuracy = artifacts["classifier"].get("assignment_external_selective_accuracy")
    classifier_coverage = artifacts["classifier"].get("assignment_external_selective_coverage")
    common_accuracy = common.get("classifier_frozen_test_accuracy")
    latency_p95 = assignment["runtime_ms"]["p95"]

    def status(condition: bool, partial: bool = False) -> str:
        if condition:
            return "pass"
        return "partial" if partial else "fail"

    return {
        "app_handles_assignment_signs_in_scope": {
            "status": status(
                assignment_rate >= 0.70,
                classifier_accuracy is not None
                and float(classifier_accuracy) >= 0.85,
            ),
            "evidence": {
                "full_pipeline_expected_in_events_rate": assignment_rate,
                "classifier_assignment_selective_accuracy": classifier_accuracy,
                "classifier_assignment_selective_coverage": classifier_coverage,
            },
        },
        "app_handles_chosen_common_malaysian_signs": {
            "status": status(
                common_accuracy is not None and float(common_accuracy) >= 0.90,
                bool(common.get("assignment_observed_labels")),
            ),
            "evidence": {
                "classifier_frozen_test_accuracy": common_accuracy,
                "assignment_observed_common_labels": common.get("assignment_observed_count"),
            },
        },
        "app_runs_within_required_time": {
            "status": status(latency_p95 is not None and float(latency_p95) < 2000.0),
            "evidence": {"assignment_pipeline_p95_runtime_ms": latency_p95},
        },
        "demo_has_reliable_fallback_video": {
            "status": status(
                Path(project_path(fallback["video"])).is_file()
                and float(fallback["smoke_event_rate"]) >= 0.80
            ),
            "evidence": {
                "video": fallback["video"],
                "smoke_event_rate": fallback["smoke_event_rate"],
                "source": fallback["source"],
            },
        },
    }


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No prediction rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_root / "stage_i_assignment_pipeline_predictions.csv"
    report_json = output_root / "stage_i_final_acceptance_report.json"
    fallback_video = output_root / "stage_i_fallback_demo_video.mp4"

    rows = load_manifest(args.assignment_manifest)
    engine = InferenceEngine(args.config)
    warmup = engine.warmup()
    predictions, assignment_metrics = evaluate_assignment_pipeline(engine=engine, rows=rows)
    write_predictions_csv(predictions_csv, predictions)
    fallback = create_and_score_fallback_video(
        engine=engine,
        source_root=project_path(args.fallback_video_source),
        output_path=fallback_video,
        limit=args.fallback_video_signs,
        fps=args.fallback_video_fps,
    )

    artifacts = summarize_stage_artifacts()
    config = load_yaml(args.config)
    common = summarize_common_signs(assignment_metrics, artifacts)
    report: dict[str, Any] = {
        "schema_version": "stage_i.final_acceptance.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stage": "Stage I - Final Evaluation",
        "config": project_rel(args.config),
        "selected_models": {
            "detector": config["detector"].get("model_path"),
            "detector_profile": {
                "backend": config["detector"].get("backend"),
                "confidence_threshold": config["detector"].get("confidence_threshold"),
                "fallback_to_baseline": config["detector"].get("fallback_to_baseline"),
                "image_size": config["detector"].get("image_size"),
            },
            "classifier": config["classifier"].get("model_path"),
            "classifier_labels": config["classifier"].get("labels_path"),
            "classifier_calibration": config["classifier"].get("calibration_path"),
        },
        "test_sets": {
            "assignment_external_test": project_rel(args.assignment_manifest),
            "malaysian_common_signs": "Stage E/G frozen test and common assignment subset",
            "demo_fallback_video": fallback["video"],
            "no_sign_negative_set": "data/processed/stage_f_negative_eval",
            "small_sign_slice": "outputs/evaluation/emtd_segmenter_s30/recall_slices.json",
        },
        "assignment_pipeline": assignment_metrics,
        "common_malaysian_signs": common,
        "fallback_video": fallback,
        "artifact_metrics": artifacts,
        "warmup": warmup,
        "artifacts": {
            "report": project_rel(report_json),
            "assignment_predictions_csv": project_rel(predictions_csv),
            "fallback_video": fallback["video"],
        },
        "limitations": [
            "Assignment labels are draft single-review labels and are treated as external acceptance evidence.",
            "Fallback video is a controlled presenter backup generated from detector validation full frames, not newly collected road footage.",
            "OCR real-road numeric accuracy still needs owner-side camera/video testing; current OCR metric is synthetic smoke coverage.",
            "No reviewed out-of-distribution classifier set exists yet, so unknown rejection is measured through selective coverage and Stage F no-sign detector negatives.",
        ],
    }
    report["completion_gates"] = gate_status(report)
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage I final acceptance evaluation.")
    parser.add_argument("--config", default="configs/inference/experimental.yaml")
    parser.add_argument(
        "--assignment-manifest",
        default="data/manifests/assignment_external_test.csv",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/evaluation/stage_i_final",
    )
    parser.add_argument("--fallback-video-signs", type=int, default=14)
    parser.add_argument("--fallback-video-fps", type=int, default=2)
    parser.add_argument(
        "--fallback-video-source",
        default="data/processed/emtd_segmentation/images/test",
    )
    args = parser.parse_args()

    report = build_report(args)
    print(json.dumps(report["completion_gates"], indent=2))
    print(f"Stage I report: {report['artifacts']['report']}")


if __name__ == "__main__":
    main()
