"""Guarded Phase E end-to-end evaluation and local runtime promotion."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.catalogue.models import ActionCode, Severity
from roadsign_assist.catalogue.repository import catalogue_by_id
from roadsign_assist.config import load_yaml
from roadsign_assist.datasets.phase_e_benchmark import (
    BENCHMARK_ID,
    CANONICAL_PATH,
    phase_e_benchmark_status,
)
from roadsign_assist.inference.engine import InferenceEngine, crop_detection
from roadsign_assist.inference.models import BoundingBoxModel, OCRModel
from roadsign_assist.inference.preprocessing import load_image_file
from roadsign_assist.paths import PROJECT_ROOT
from roadsign_assist.semantics.rules import SemanticRuleEngine

PHASE_E_OUTPUT = Path("outputs/evaluation/phase_e")
PHASE_E_CONFIG = Path("configs/inference/phase_e_candidate.yaml")
PHASE_E_RUNTIME_CONFIG = Path("configs/inference/phase_e_runtime.yaml")
GATE_PATH = PHASE_E_OUTPUT / "gate_decision.json"

DETECTOR_ID = "pd_v1_yolo26s_640_s2513"
DETECTOR_PT_SHA256 = "1b18c8a10a8dfe66ec1b6cbff84dd6434063de2b0d75d04126fa43110248ae3d"
DETECTOR_ONNX_SHA256 = "8b8caa9e5ed6a9437dd86ba02a1f2a3b2eaa0b2af1ddd6cc352bd21f83109703"
CLASSIFIER_ID = "pb_v3_effnetv2m_320_s2513"
CLASSIFIER_THRESHOLD = 0.696052074432373
CLASSIFIER_ONNX_SHA256 = "330148d353766f9218ff85a3fb12385a6015765c22566f61bfcbd13bfe633fe8"
CLASSIFIER_LABELS_SHA256 = "64332c13f020150a66da05d7df420b67d4f27b60517d34e1cabd106f8c9dc155"
CLASSIFIER_CALIBRATION_SHA256 = "404b5981d75092a0bae5adcbd7de4b5daea9d78f65dc8888dedecdbd764ccdbb"
DETECTOR_CONFIDENCE = 0.37
DETECTOR_IMAGE_SIZE = 640
MATCH_IOU = 0.50
CROP_PADDING = 0.06

PREDICTION_FIELDS = [
    "profile",
    "image_id",
    "image_path",
    "instance_id",
    "domain",
    "condition_bucket",
    "expected_kind",
    "expected_semantic",
    "expected_parameter",
    "expected_parameter_unit",
    "small_sign",
    "very_small_sign",
    "matched",
    "match_iou",
    "predicted_bbox",
    "detector_confidence",
    "conditional_semantic",
    "conditional_confidence",
    "conditional_unknown_score",
    "oracle_semantic",
    "oracle_confidence",
    "oracle_unknown_score",
    "final_semantic",
    "final_confidence",
    "predicted_parameter",
    "predicted_parameter_unit",
    "action_code",
    "advisory_safe_to_announce",
    "unsafe_strong_action",
    "event_count",
    "runtime_ms",
    "failure_reason",
]
IMAGE_PREDICTION_FIELDS = [
    "profile",
    "image_id",
    "image_path",
    "domain",
    "condition_bucket",
    "ground_truth_boxes",
    "matched_ground_truth_boxes",
    "predicted_boxes",
    "failure_count",
    "unsafe_strong_action_count",
    "runtime_ms",
]


def _resolve(root: Path, path: Path | str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _project_rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(quantile * (len(ordered) - 1)))]


def _environment(profile: str) -> dict[str, object]:
    try:
        import onnxruntime as ort

        providers = [str(item) for item in cast(Any, ort).get_available_providers()]
    except ImportError:
        providers = []
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except (ImportError, AssertionError, RuntimeError):
        cuda_available = False
        gpu_name = None
    return {
        "profile": profile,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "onnxruntime_providers": providers,
    }


def validate_phase_e_artifacts(
    *,
    project_root: Path = PROJECT_ROOT,
    require_benchmark: bool = True,
) -> dict[str, object]:
    root = project_root.resolve()
    candidate_root = root / "models/candidates/phase_d" / DETECTOR_ID
    pt_path = candidate_root / "model.pt"
    onnx_path = candidate_root / "model.onnx"
    locked = _read_json(candidate_root / "locked_test_report.json")
    candidate = _read_json(candidate_root / "candidate_manifest.json")
    classifier = _read_json(root / "models/exported/runtime/sign_classifier.runtime.json")
    calibration = _read_json(root / "models/exported/runtime/sign_classifier.calibration.json")
    failures: list[str] = []
    if _sha256(pt_path) != DETECTOR_PT_SHA256:
        failures.append("detector_pytorch_hash")
    if _sha256(onnx_path) != DETECTOR_ONNX_SHA256:
        failures.append("detector_onnx_hash")
    if candidate.get("candidate_id") != DETECTOR_ID or candidate.get("gate_d_passed") is not True:
        failures.append("detector_candidate_manifest")
    if locked.get("gate_d_passed") is not True or locked.get("test_evaluated") is not True:
        failures.append("detector_locked_test")
    if (
        float(locked.get("confidence", -1)) != DETECTOR_CONFIDENCE
        or int(locked.get("image_size", -1)) != DETECTOR_IMAGE_SIZE
    ):
        failures.append("detector_locked_configuration")
    if (
        classifier.get("source_run") != CLASSIFIER_ID
        or classifier.get("onnx_parity_passed") is not True
        or abs(float(classifier.get("confidence_threshold", -1)) - CLASSIFIER_THRESHOLD) > 1e-12
    ):
        failures.append("classifier_runtime_manifest")
    classifier_hashes = {
        "sign_classifier.onnx": CLASSIFIER_ONNX_SHA256,
        "sign_classifier.labels.json": CLASSIFIER_LABELS_SHA256,
        "sign_classifier.calibration.json": CLASSIFIER_CALIBRATION_SHA256,
    }
    for name, expected_hash in classifier_hashes.items():
        path = root / "models/exported/runtime" / name
        if not path.is_file() or _sha256(path) != expected_hash:
            failures.append(f"classifier_artifact_hash:{name}")
    if (
        classifier.get("internal_academic_only") is not True
        or classifier.get("publication_prohibited") is not True
    ):
        failures.append("classifier_policy")
    if abs(float(calibration.get("confidence_threshold", -1)) - CLASSIFIER_THRESHOLD) > 1e-12:
        failures.append("classifier_threshold")
    if int(calibration.get("image_size", -1)) != 320:
        failures.append("classifier_image_size")
    config = load_yaml(PHASE_E_CONFIG)
    detector = cast(dict[str, Any], config.get("detector", {}))
    classifier_config = cast(dict[str, Any], config.get("classifier", {}))
    profiles = cast(dict[str, Any], detector.get("profiles", {}))
    gpu_profile = cast(dict[str, Any], profiles.get("gpu", {}))
    cpu_profile = cast(dict[str, Any], profiles.get("cpu", {}))
    if (
        detector.get("task") != "detect"
        or int(detector.get("image_size", -1)) != DETECTOR_IMAGE_SIZE
        or abs(float(detector.get("confidence_threshold", -1)) - DETECTOR_CONFIDENCE) > 1e-12
        or abs(float(detector.get("nms_iou_threshold", -1)) - MATCH_IOU) > 1e-12
        or bool(detector.get("fallback_to_baseline"))
        or abs(float(config.get("crop_padding", -1)) - CROP_PADDING) > 1e-12
    ):
        failures.append("phase_e_detector_config")
    if (
        gpu_profile.get("model_path") != f"models/candidates/phase_d/{DETECTOR_ID}/model.pt"
        or gpu_profile.get("sha256") != DETECTOR_PT_SHA256
        or str(gpu_profile.get("device")) != "0"
        or cpu_profile.get("model_path") != f"models/candidates/phase_d/{DETECTOR_ID}/model.onnx"
        or cpu_profile.get("sha256") != DETECTOR_ONNX_SHA256
        or str(cpu_profile.get("device")) != "cpu"
    ):
        failures.append("phase_e_detector_profiles")
    if (
        int(classifier_config.get("image_size", -1)) != 320
        or abs(float(classifier_config.get("confidence_threshold", -1)) - CLASSIFIER_THRESHOLD)
        > 1e-12
        or classifier_config.get("model_path") != "models/exported/runtime/sign_classifier.onnx"
        or classifier_config.get("labels_path")
        != "models/exported/runtime/sign_classifier.labels.json"
        or classifier_config.get("calibration_path")
        != "models/exported/runtime/sign_classifier.calibration.json"
        or classifier_config.get("internal_academic_only") is not True
    ):
        failures.append("phase_e_classifier_config")
    benchmark: dict[str, object] | None = None
    if require_benchmark:
        status = phase_e_benchmark_status(project_root=root)
        if status.get("status") != "frozen":
            failures.append("phase_e_benchmark_not_frozen")
        else:
            benchmark_path = _resolve(root, CANONICAL_PATH)
            benchmark = {
                "path": _project_rel(root, benchmark_path),
                "sha256": _sha256(benchmark_path),
                "rows": status.get("rows"),
                "images": status.get("images"),
            }
    if failures:
        raise ValueError(f"Phase E preflight failed: {', '.join(failures)}")
    return {
        "detector_id": DETECTOR_ID,
        "detector_pytorch_sha256": DETECTOR_PT_SHA256,
        "detector_onnx_sha256": DETECTOR_ONNX_SHA256,
        "classifier_id": CLASSIFIER_ID,
        "classifier_threshold": CLASSIFIER_THRESHOLD,
        "benchmark": benchmark,
        "test_retuning_allowed": False,
        "assignment_only": True,
        "publication_prohibited": True,
    }


def _bbox(row: Mapping[str, str]) -> BoundingBoxModel:
    return BoundingBoxModel(
        x1=float(row["x1"]),
        y1=float(row["y1"]),
        x2=float(row["x2"]),
        y2=float(row["y2"]),
    )


def match_boxes(
    ground_truth: list[BoundingBoxModel],
    predictions: list[BoundingBoxModel],
    *,
    minimum_iou: float = MATCH_IOU,
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        (
            (truth.iou(prediction), truth_index, prediction_index)
            for truth_index, truth in enumerate(ground_truth)
            for prediction_index, prediction in enumerate(predictions)
            if truth.iou(prediction) >= minimum_iou
        ),
        reverse=True,
    )
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, truth_index, prediction_index in candidates:
        if truth_index in used_truth or prediction_index in used_predictions:
            continue
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index, iou))
    return sorted(matches)


def _macro_f1(
    truth: list[str],
    predictions: list[str],
    *,
    extra_predictions: Iterable[str] = (),
) -> float | None:
    labels = sorted(set(truth))
    if not labels:
        return None
    extras = list(extra_predictions)
    scores: list[float] = []
    for label in labels:
        tp = sum(
            actual == label and predicted == label
            for actual, predicted in zip(truth, predictions, strict=True)
        )
        fn = sum(
            actual == label and predicted != label
            for actual, predicted in zip(truth, predictions, strict=True)
        )
        fp = sum(
            actual != label and predicted == label
            for actual, predicted in zip(truth, predictions, strict=True)
        )
        fp += sum(predicted == label for predicted in extras)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return statistics.mean(scores)


def _parse_detector_confidence(evidence: list[str]) -> float | None:
    for item in evidence:
        if item.startswith("detector:"):
            try:
                return float(item.rsplit(":", 1)[-1])
            except ValueError:
                return None
    return None


def _unsafe_strong_action(
    expected: str,
    predicted: str,
    action: object,
    safe: bool,
    *,
    parameter_correct: bool = True,
) -> bool:
    code = getattr(action, "value", action)
    return bool(
        safe
        and str(code) != ActionCode.UNKNOWN_CAUTION.value
        and (expected not in catalogue_by_id() or predicted != expected or not parameter_correct)
    )


def _safe_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _event_parameter(event: Any) -> float | None:
    for value in (
        event.action.target_speed_kmh,
        event.action.restriction_value,
        event.ocr.numeric_value,
    ):
        if value is not None:
            return float(value)
    return None


def _parameter_matches(
    event: Any,
    expected_value: float | None,
    expected_unit: str | None = None,
) -> bool:
    if expected_value is None:
        return True
    actual = _event_parameter(event)
    if actual is None or abs(actual - expected_value) > 1e-3:
        return False
    if expected_unit:
        return bool(
            event.ocr.unit
            and str(event.ocr.unit).strip().casefold() == expected_unit.strip().casefold()
        )
    return True


def _coursework_evaluation(engine: InferenceEngine, root: Path, output: Path) -> dict[str, object]:
    source = _read_csv(root / "data/manifests/assignment_external_test.csv")
    predictions: list[dict[str, object]] = []
    runtimes: list[float] = []
    numeric_total = 0
    numeric_correct = 0
    unsafe = 0
    for row in source:
        image = load_image_file(_resolve(root, row["path"]))
        started = time.perf_counter()
        result = engine.new_session().process_frame(image, assume_stable=True)
        elapsed = (time.perf_counter() - started) * 1000
        runtimes.append(elapsed)
        events = sorted(result.events, key=lambda event: event.confidence, reverse=True)
        primary = events[0] if events else None
        expected = row["semantic_sign_id"]
        expected_parameter = _safe_float(row.get("parameter_value"))
        matched_events = [event for event in events if event.semantic_sign_id == expected]
        numeric_match: bool | None = None
        if expected_parameter is not None:
            numeric_total += 1
            values = [
                value
                for event in matched_events
                for value in (
                    event.action.target_speed_kmh,
                    event.action.restriction_value,
                    event.ocr.numeric_value,
                )
                if value is not None
            ]
            numeric_match = any(abs(float(value) - expected_parameter) <= 1.0 for value in values)
            numeric_correct += int(numeric_match)
        row_unsafe = sum(
            _unsafe_strong_action(
                expected,
                event.semantic_sign_id,
                event.action.code,
                event.advisory.safe_to_announce,
                parameter_correct=_parameter_matches(
                    event, expected_parameter, row.get("parameter_unit")
                ),
            )
            for event in events
        )
        unsafe += row_unsafe
        predictions.append(
            {
                "external_test_id": row.get("external_test_id", ""),
                "path": row["path"],
                "expected": expected,
                "expected_parameter": expected_parameter if expected_parameter is not None else "",
                "event_count": len(events),
                "primary_semantic": primary.semantic_sign_id if primary else "",
                "primary_correct": bool(primary and primary.semantic_sign_id == expected),
                "expected_in_events": bool(matched_events),
                "numeric_match": "" if numeric_match is None else numeric_match,
                "unsafe_strong_actions": row_unsafe,
                "runtime_ms": round(elapsed, 3),
                "under_two_seconds": elapsed < 2000,
            }
        )
    _write_csv(
        output / "coursework_predictions.csv",
        predictions,
        [
            "external_test_id",
            "path",
            "expected",
            "expected_parameter",
            "event_count",
            "primary_semantic",
            "primary_correct",
            "expected_in_events",
            "numeric_match",
            "unsafe_strong_actions",
            "runtime_ms",
            "under_two_seconds",
        ],
    )
    return {
        "images": len(predictions),
        "completed": len(predictions),
        "images_with_expected_event": sum(bool(row["expected_in_events"]) for row in predictions),
        "primary_correct": sum(bool(row["primary_correct"]) for row in predictions),
        "semantic_accuracy": (
            sum(bool(row["primary_correct"]) for row in predictions) / len(predictions)
            if predictions
            else None
        ),
        "numeric_rows": numeric_total,
        "numeric_correct": numeric_correct,
        "numeric_accuracy": numeric_correct / numeric_total if numeric_total else None,
        "unsafe_strong_actions": unsafe,
        "runtime_ms": {
            "mean": statistics.mean(runtimes),
            "p95": _percentile(runtimes, 0.95),
            "maximum": max(runtimes),
        },
        "all_under_two_seconds": all(value < 2000 for value in runtimes),
    }


def _ocr_advisory_contract() -> dict[str, object]:
    rules = SemanticRuleEngine()
    cases = [
        (
            "maximum_speed",
            0.99,
            OCRModel(confidence=0.99, numeric_value=80, unit="KM/H"),
            ActionCode.SET_TARGET_SPEED,
            80.0,
            True,
        ),
        (
            "width_restriction",
            0.99,
            OCRModel(confidence=0.99, numeric_value=2.5, unit="M"),
            ActionCode.WIDTH_RESTRICTION,
            2.5,
            True,
        ),
        (
            "maximum_speed",
            0.99,
            OCRModel(confidence=0.99, numeric_value=999, unit="KM/H"),
            ActionCode.UNKNOWN_CAUTION,
            None,
            False,
        ),
        (
            "width_restriction",
            0.99,
            OCRModel(confidence=0.99, numeric_value=2.5, unit="FT"),
            ActionCode.UNKNOWN_CAUTION,
            None,
            False,
        ),
        (
            "maximum_speed",
            0.99,
            OCRModel(confidence=0.50, numeric_value=80, unit="KM/H"),
            ActionCode.UNKNOWN_CAUTION,
            None,
            False,
        ),
        ("stop", 0.40, OCRModel(), ActionCode.UNKNOWN_CAUTION, None, False),
    ]
    results: list[dict[str, object]] = []
    for label, confidence, ocr, expected_code, expected_value, expected_safe in cases:
        meaning, _, action = rules.action_for(label, confidence, ocr)
        advisory = rules.advisory_for(label, meaning, confidence, action)
        actual_value = (
            action.target_speed_kmh
            if action.target_speed_kmh is not None
            else action.restriction_value
        )
        passed = (
            action.code is expected_code
            and actual_value == expected_value
            and advisory.safe_to_announce is expected_safe
        )
        results.append(
            {
                "label": label,
                "expected_code": expected_code.value,
                "actual_code": action.code.value,
                "expected_value": expected_value,
                "actual_value": actual_value,
                "expected_safe": expected_safe,
                "actual_safe": advisory.safe_to_announce,
                "passed": passed,
            }
        )
    return {"cases": results, "passed": all(bool(row["passed"]) for row in results)}


def _streaming_benchmark(
    engine: InferenceEngine, image: UInt8Image, frames: int = 30
) -> dict[str, object]:
    session = engine.new_session()
    started = time.perf_counter()
    first_stable_seconds: float | None = None
    for _ in range(frames):
        result = session.process_frame(image)
        if first_stable_seconds is None and any(event.stable for event in result.events):
            first_stable_seconds = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    return {
        "frames": frames,
        "elapsed_seconds": elapsed,
        "frames_per_second": frames / elapsed,
        "first_stable_warning_seconds": first_stable_seconds,
    }


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    if not labels or len(set(labels)) < 2:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    positives = sum(labels)
    negatives = len(labels) - positives
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate_phase_e_profile(
    profile: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    if profile not in {"gpu", "cpu"}:
        raise ValueError("Phase E profile must be gpu or cpu")
    root = project_root.resolve()
    preflight = validate_phase_e_artifacts(project_root=root, require_benchmark=True)
    benchmark_path = _resolve(root, CANONICAL_PATH)
    rows = _read_csv(benchmark_path)
    by_image: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    output = _resolve(root, PHASE_E_OUTPUT / profile)
    output.mkdir(parents=True, exist_ok=True)
    engine = InferenceEngine(PHASE_E_CONFIG, runtime_profile=profile)
    warmup = engine.warmup()
    if not warmup.get("detector") or not warmup.get("classifier"):
        raise RuntimeError(f"Phase E {profile} model warm-up failed: {warmup}")
    predictions: list[dict[str, object]] = []
    latencies: list[float] = []
    benchmark_started = time.perf_counter()
    streaming_image: UInt8Image | None = None
    for image_id, truth_rows in sorted(by_image.items()):
        image_path = _resolve(root, truth_rows[0]["image_path"])
        image = load_image_file(image_path)
        if streaming_image is None and any(row["expected_kind"] == "sign" for row in truth_rows):
            streaming_image = image
        started = time.perf_counter()
        result = engine.new_session().process_frame(image, assume_stable=True)
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)
        events = list(result.events)
        conditional = [
            engine.classifier.classify(crop_detection(image, event.bbox, padding=CROP_PADDING))
            for event in events
        ]
        sign_rows = [row for row in truth_rows if row["expected_kind"] == "sign"]
        matches = match_boxes([_bbox(row) for row in sign_rows], [event.bbox for event in events])
        by_truth = {
            truth_index: (prediction_index, iou) for truth_index, prediction_index, iou in matches
        }
        matched_predictions = {prediction_index for _, prediction_index, _ in matches}
        for truth_index, truth in enumerate(sign_rows):
            oracle = engine.classifier.classify(
                crop_detection(image, _bbox(truth), padding=CROP_PADDING)
            )
            match = by_truth.get(truth_index)
            event = events[match[0]] if match else None
            classifier = conditional[match[0]] if match else None
            expected = truth["semantic_sign_id"]
            expected_parameter = _safe_float(truth.get("parameter_value"))
            expected_parameter_unit = truth.get("parameter_unit", "")
            parameter_correct = (
                bool(
                    event and _parameter_matches(event, expected_parameter, expected_parameter_unit)
                )
                if expected_parameter is not None
                else True
            )
            unsafe = bool(
                event
                and _unsafe_strong_action(
                    expected,
                    event.semantic_sign_id,
                    event.action.code,
                    event.advisory.safe_to_announce,
                    parameter_correct=parameter_correct,
                )
            )
            predictions.append(
                {
                    "profile": profile,
                    "image_id": image_id,
                    "image_path": truth["image_path"],
                    "instance_id": truth["instance_id"],
                    "domain": truth["domain"],
                    "condition_bucket": truth["condition_bucket"],
                    "expected_kind": "sign",
                    "expected_semantic": expected,
                    "expected_parameter": expected_parameter
                    if expected_parameter is not None
                    else "",
                    "expected_parameter_unit": expected_parameter_unit,
                    "small_sign": truth["small_sign"],
                    "very_small_sign": truth["very_small_sign"],
                    "matched": match is not None,
                    "match_iou": round(match[1], 6) if match else "",
                    "predicted_bbox": (
                        f"{event.bbox.x1:.3f}|{event.bbox.y1:.3f}|{event.bbox.x2:.3f}|{event.bbox.y2:.3f}"
                        if event
                        else ""
                    ),
                    "detector_confidence": _parse_detector_confidence(event.evidence)
                    if event
                    else "",
                    "conditional_semantic": classifier.semantic_sign_id
                    if classifier
                    else "missed_detection",
                    "conditional_confidence": classifier.confidence if classifier else "",
                    "conditional_unknown_score": classifier.unknown_score if classifier else "",
                    "oracle_semantic": oracle.semantic_sign_id,
                    "oracle_confidence": oracle.confidence,
                    "oracle_unknown_score": oracle.unknown_score,
                    "final_semantic": event.semantic_sign_id if event else "missed_detection",
                    "final_confidence": event.confidence if event else "",
                    "predicted_parameter": _event_parameter(event) if event else "",
                    "predicted_parameter_unit": event.ocr.unit if event else "",
                    "action_code": event.action.code.value if event else "",
                    "advisory_safe_to_announce": event.advisory.safe_to_announce if event else "",
                    "unsafe_strong_action": unsafe,
                    "event_count": len(events),
                    "runtime_ms": round(elapsed, 3),
                    "failure_reason": (
                        "missed_detection"
                        if event is None
                        else (
                            "semantic_error"
                            if event.semantic_sign_id != expected
                            else ("parameter_error" if not parameter_correct else "")
                        )
                    ),
                }
            )
        for prediction_index, event in enumerate(events):
            if prediction_index in matched_predictions:
                continue
            classifier = conditional[prediction_index]
            no_sign = not sign_rows
            unsafe = bool(
                no_sign
                and event.advisory.safe_to_announce
                and event.action.code is not ActionCode.UNKNOWN_CAUTION
            )
            predictions.append(
                {
                    "profile": profile,
                    "image_id": image_id,
                    "image_path": truth_rows[0]["image_path"],
                    "instance_id": f"{image_id}:false-positive-{prediction_index:03d}",
                    "domain": truth_rows[0]["domain"],
                    "condition_bucket": truth_rows[0]["condition_bucket"],
                    "expected_kind": "no_sign" if no_sign else "unmatched_prediction",
                    "expected_semantic": "no_sign" if no_sign else "",
                    "matched": False,
                    "predicted_bbox": f"{event.bbox.x1:.3f}|{event.bbox.y1:.3f}|{event.bbox.x2:.3f}|{event.bbox.y2:.3f}",
                    "detector_confidence": _parse_detector_confidence(event.evidence),
                    "conditional_semantic": classifier.semantic_sign_id,
                    "conditional_confidence": classifier.confidence,
                    "conditional_unknown_score": classifier.unknown_score,
                    "final_semantic": event.semantic_sign_id,
                    "final_confidence": event.confidence,
                    "action_code": event.action.code.value,
                    "advisory_safe_to_announce": event.advisory.safe_to_announce,
                    "unsafe_strong_action": unsafe,
                    "event_count": len(events),
                    "runtime_ms": round(elapsed, 3),
                    "failure_reason": "no_sign_false_positive"
                    if no_sign
                    else "unmatched_prediction",
                }
            )
        if not sign_rows and not events:
            predictions.append(
                {
                    "profile": profile,
                    "image_id": image_id,
                    "image_path": truth_rows[0]["image_path"],
                    "instance_id": truth_rows[0]["instance_id"],
                    "domain": truth_rows[0]["domain"],
                    "condition_bucket": truth_rows[0]["condition_bucket"],
                    "expected_kind": "no_sign",
                    "expected_semantic": "no_sign",
                    "matched": True,
                    "event_count": 0,
                    "runtime_ms": round(elapsed, 3),
                    "unsafe_strong_action": False,
                    "failure_reason": "",
                }
            )
    total_elapsed = time.perf_counter() - benchmark_started
    catalogue = catalogue_by_id()
    sign_predictions = [row for row in predictions if row["expected_kind"] == "sign"]
    supported = [row for row in sign_predictions if str(row["expected_semantic"]) in catalogue]
    matched_supported = [row for row in supported if bool(row["matched"])]
    conditional_truth = [str(row["expected_semantic"]) for row in matched_supported]
    conditional_pred = [str(row["conditional_semantic"]) for row in matched_supported]
    end_truth = [str(row["expected_semantic"]) for row in supported]
    end_pred = [str(row["final_semantic"]) for row in supported]
    extra_predictions = [
        str(row["final_semantic"])
        for row in predictions
        if row["expected_kind"] in {"no_sign", "unmatched_prediction"}
        and str(row.get("final_semantic", "")) in catalogue
    ]
    critical = {
        label for label, definition in catalogue.items() if definition.severity is Severity.CRITICAL
    }
    critical_rows = [row for row in supported if str(row["expected_semantic"]) in critical]
    ood_rows = [
        row
        for row in sign_predictions
        if str(row["expected_semantic"]) in {"out_of_ontology", "unreadable"}
    ]
    oracle_auc_rows = [*supported, *ood_rows]
    oracle_auc_labels = [int(row in ood_rows) for row in oracle_auc_rows]
    oracle_auc_scores = [
        float(cast(Any, row["oracle_unknown_score"]))
        for row in oracle_auc_rows
        if row.get("oracle_unknown_score") not in {"", None}
    ]
    if len(oracle_auc_scores) != len(oracle_auc_labels):
        oracle_auc_labels = []
        oracle_auc_scores = []
    conditional_auc_rows = [
        row
        for row in oracle_auc_rows
        if bool(row["matched"]) and row.get("conditional_unknown_score") not in {"", None}
    ]
    conditional_auc_labels = [int(row in ood_rows) for row in conditional_auc_rows]
    conditional_auc_scores = [
        float(cast(Any, row["conditional_unknown_score"])) for row in conditional_auc_rows
    ]
    no_sign_image_ids = {row["image_id"] for row in rows if row["expected_kind"] == "no_sign"}
    no_sign_false_rows = [
        row
        for row in predictions
        if row["image_id"] in no_sign_image_ids and row.get("final_semantic")
    ]
    no_sign_false_images = {row["image_id"] for row in no_sign_false_rows}

    def recall_slice(items: list[dict[str, object]]) -> dict[str, object]:
        return {
            "ground_truth": len(items),
            "matched": sum(bool(row["matched"]) for row in items),
            "recall": sum(bool(row["matched"]) for row in items) / len(items) if items else None,
        }

    slices: dict[str, object] = {}
    for name in sorted({str(row["condition_bucket"]) for row in sign_predictions}):
        slices[name] = recall_slice(
            [row for row in sign_predictions if row["condition_bucket"] == name]
        )
    streaming = (
        _streaming_benchmark(engine, streaming_image)
        if profile == "gpu" and streaming_image is not None
        else None
    )
    coursework = _coursework_evaluation(engine, root, output) if profile == "cpu" else None
    unsafe_actions = sum(bool(row.get("unsafe_strong_action")) for row in predictions)
    safety_error_signatures = sorted(
        str(row["instance_id"]) for row in predictions if bool(row.get("unsafe_strong_action"))
    )
    metrics: dict[str, object] = {
        "schema_version": "1.0",
        "phase": "E",
        "profile": profile,
        "status": "evaluated",
        "preflight": preflight,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_sha256": _sha256(benchmark_path),
        "configuration": {
            "detector_id": DETECTOR_ID,
            "classifier_id": CLASSIFIER_ID,
            "detector_image_size": DETECTOR_IMAGE_SIZE,
            "detector_confidence": DETECTOR_CONFIDENCE,
            "match_iou": MATCH_IOU,
            "classifier_threshold": CLASSIFIER_THRESHOLD,
            "crop_padding": CROP_PADDING,
            "test_retuning_allowed": False,
        },
        "images": len(by_image),
        "sign_boxes": len(sign_predictions),
        "detector": {
            "overall": recall_slice(sign_predictions),
            "full_road": recall_slice(sign_predictions),
            "small": recall_slice([row for row in sign_predictions if row["small_sign"] == "true"]),
            "very_small": recall_slice(
                [row for row in sign_predictions if row["very_small_sign"] == "true"]
            ),
            "condition_slices": slices,
        },
        "conditional_classifier": {
            "samples": len(matched_supported),
            "accuracy": (
                sum(
                    actual == predicted
                    for actual, predicted in zip(conditional_truth, conditional_pred, strict=True)
                )
                / len(conditional_truth)
                if conditional_truth
                else None
            ),
            "macro_f1": _macro_f1(conditional_truth, conditional_pred),
        },
        "oracle_crop_classifier": {
            "samples": len(supported),
            "accuracy": (
                sum(
                    str(row["oracle_semantic"]) == str(row["expected_semantic"])
                    for row in supported
                )
                / len(supported)
                if supported
                else None
            ),
            "macro_f1": _macro_f1(end_truth, [str(row["oracle_semantic"]) for row in supported]),
        },
        "end_to_end": {
            "samples": len(supported),
            "accuracy": (
                sum(
                    actual == predicted
                    for actual, predicted in zip(end_truth, end_pred, strict=True)
                )
                / len(end_truth)
                if end_truth
                else None
            ),
            "macro_f1": _macro_f1(end_truth, end_pred, extra_predictions=extra_predictions),
            "safety_critical_samples": len(critical_rows),
            "safety_critical_recall": (
                sum(row["final_semantic"] == row["expected_semantic"] for row in critical_rows)
                / len(critical_rows)
                if critical_rows
                else None
            ),
        },
        "unknown": {
            "supported_boxes": len(supported),
            "out_of_ontology_boxes": len(ood_rows),
            "auroc": _roc_auc(conditional_auc_labels, conditional_auc_scores),
            "conditional_auroc": _roc_auc(conditional_auc_labels, conditional_auc_scores),
            "oracle_crop_auroc": _roc_auc(oracle_auc_labels, oracle_auc_scores),
        },
        "no_sign": {
            "images": len(no_sign_image_ids),
            "images_with_false_positive": len(no_sign_false_images),
            "image_false_positive_rate": len(no_sign_false_images) / len(no_sign_image_ids)
            if no_sign_image_ids
            else None,
            "false_boxes": len(no_sign_false_rows),
            "false_boxes_per_100_images": len(no_sign_false_rows) * 100 / len(no_sign_image_ids)
            if no_sign_image_ids
            else None,
        },
        "safety": {
            "unsafe_strong_actions": unsafe_actions,
            "error_signatures": safety_error_signatures,
            "ocr_advisory_contract": _ocr_advisory_contract(),
        },
        "runtime": {
            "warmup": warmup,
            "wall_seconds": total_elapsed,
            "throughput_images_per_second": len(by_image) / total_elapsed,
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "p95": _percentile(latencies, 0.95),
                "maximum": max(latencies),
            },
            "streaming": streaming,
            "coursework": coursework,
        },
        "environment": {**_environment(profile), "model_status": engine.model_status},
        "assignment_only": True,
        "publication_prohibited": True,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
    _write_csv(output / "predictions.csv", predictions, PREDICTION_FIELDS)
    prediction_groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        prediction_groups[str(row["image_id"])].append(row)
    image_predictions: list[dict[str, object]] = []
    for image_id, image_rows in sorted(prediction_groups.items()):
        truth_rows = by_image[image_id]
        image_predictions.append(
            {
                "profile": profile,
                "image_id": image_id,
                "image_path": truth_rows[0]["image_path"],
                "domain": truth_rows[0]["domain"],
                "condition_bucket": truth_rows[0]["condition_bucket"],
                "ground_truth_boxes": sum(row["expected_kind"] == "sign" for row in image_rows),
                "matched_ground_truth_boxes": sum(
                    row["expected_kind"] == "sign" and bool(row["matched"]) for row in image_rows
                ),
                "predicted_boxes": max(int(cast(Any, row["event_count"])) for row in image_rows),
                "failure_count": sum(bool(row.get("failure_reason")) for row in image_rows),
                "unsafe_strong_action_count": sum(
                    bool(row.get("unsafe_strong_action")) for row in image_rows
                ),
                "runtime_ms": image_rows[0]["runtime_ms"],
            }
        )
    _write_csv(output / "per_image_predictions.csv", image_predictions, IMAGE_PREDICTION_FIELDS)
    _write_csv(
        output / "failures.csv",
        [row for row in predictions if row.get("failure_reason")],
        PREDICTION_FIELDS,
    )
    _write_json(output / "metrics.json", metrics)
    _write_json(
        output / "condition_report.json",
        {
            "profile": profile,
            "condition_slices": slices,
            "small_sign": cast(dict[str, Any], metrics["detector"])["small"],
            "very_small_sign": cast(dict[str, Any], metrics["detector"])["very_small"],
        },
    )
    failure_counts = Counter(
        str(row["failure_reason"]) for row in predictions if row.get("failure_reason")
    )
    _write_json(
        output / "failure_case_report.json",
        {
            "profile": profile,
            "total_failures": sum(failure_counts.values()),
            "failure_counts": dict(sorted(failure_counts.items())),
            "detector_misses": failure_counts.get("missed_detection", 0),
            "unsafe_strong_actions": unsafe_actions,
        },
    )
    _write_json(output / "environment.json", cast(dict[str, object], metrics["environment"]))
    _write_json(output / "latency.json", cast(dict[str, object], metrics["runtime"]))
    _write_json(
        output / "coursework_ocr_advisory_audit.json",
        {
            "profile": profile,
            "coursework": coursework,
            "ocr_advisory_contract": cast(dict[str, Any], metrics["safety"])[
                "ocr_advisory_contract"
            ],
            "unsafe_strong_actions": unsafe_actions,
        },
    )
    (output / "report.md").write_text(_render_profile_report(metrics), encoding="utf-8")
    return metrics


def _render_profile_report(metrics: Mapping[str, object]) -> str:
    detector = cast(dict[str, Any], metrics["detector"])
    conditional = cast(dict[str, Any], metrics["conditional_classifier"])
    end_to_end = cast(dict[str, Any], metrics["end_to_end"])
    no_sign = cast(dict[str, Any], metrics["no_sign"])
    runtime = cast(dict[str, Any], metrics["runtime"])
    return (
        f"# Phase E {metrics['profile']} profile\n\n"
        f"- Detector recall: {detector['overall']['recall']:.4f}\n"
        f"- Small-sign recall: {detector['small']['recall']:.4f}\n"
        f"- Conditional classifier macro-F1: {conditional['macro_f1']:.4f}\n"
        f"- End-to-end macro-F1: {end_to_end['macro_f1']:.4f}\n"
        f"- Safety-critical recall: {end_to_end['safety_critical_recall']}\n"
        f"- No-sign image false-positive rate: {no_sign['image_false_positive_rate']:.4f}\n"
        f"- Mean / p95 latency: {runtime['latency_ms']['mean']:.1f} / {runtime['latency_ms']['p95']:.1f} ms\n"
        "- Test retuning allowed: false\n"
    )


def finalize_phase_e_gate(*, project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    root = project_root.resolve()
    gpu = _read_json(_resolve(root, PHASE_E_OUTPUT / "gpu/metrics.json"))
    cpu = _read_json(_resolve(root, PHASE_E_OUTPUT / "cpu/metrics.json"))

    def nested(payload: dict[str, Any], *keys: str) -> Any:
        value: Any = payload
        for key in keys:
            value = value[key]
        return value

    def pair_values(*keys: str) -> tuple[float, float] | None:
        values = (nested(gpu, *keys), nested(cpu, *keys))
        if any(value is None for value in values):
            return None
        try:
            return float(values[0]), float(values[1])
        except (TypeError, ValueError):
            return None

    def pair_at_least(minimum: float, *keys: str) -> bool:
        values = pair_values(*keys)
        return values is not None and min(values) >= minimum

    def pair_at_most(maximum: float, *keys: str) -> bool:
        values = pair_values(*keys)
        return values is not None and max(values) <= maximum

    checks = {
        "full_road_detector_recall": pair_at_least(0.90, "detector", "full_road", "recall"),
        "small_sign_recall": pair_at_least(0.80, "detector", "small", "recall"),
        "conditional_classifier_macro_f1": pair_at_least(
            0.85, "conditional_classifier", "macro_f1"
        ),
        "end_to_end_macro_f1": pair_at_least(0.85, "end_to_end", "macro_f1"),
        "safety_critical_recall": pair_at_least(0.90, "end_to_end", "safety_critical_recall"),
        "unknown_auroc": pair_at_least(0.85, "unknown", "auroc"),
        "no_sign_image_false_positive_rate": pair_at_most(
            0.025, "no_sign", "image_false_positive_rate"
        ),
        "no_sign_false_boxes_per_100": pair_at_most(2.5, "no_sign", "false_boxes_per_100_images"),
        "zero_unsafe_strong_actions": int(nested(gpu, "safety", "unsafe_strong_actions"))
        + int(nested(cpu, "safety", "unsafe_strong_actions"))
        == 0,
        "ocr_advisory_contract": bool(nested(gpu, "safety", "ocr_advisory_contract", "passed"))
        and bool(nested(cpu, "safety", "ocr_advisory_contract", "passed")),
        "coursework_complete": int(nested(cpu, "runtime", "coursework", "completed")) == 84,
        "coursework_cpu_under_two_seconds": bool(
            nested(cpu, "runtime", "coursework", "all_under_two_seconds")
        ),
        "coursework_zero_unsafe_actions": int(
            nested(cpu, "runtime", "coursework", "unsafe_strong_actions")
        )
        == 0,
        "gpu_streaming_fps": nested(gpu, "runtime", "streaming", "frames_per_second") is not None
        and float(nested(gpu, "runtime", "streaming", "frames_per_second")) >= 15.0,
        "stable_warning_under_one_second": nested(
            gpu, "runtime", "streaming", "first_stable_warning_seconds"
        )
        is not None
        and float(nested(gpu, "runtime", "streaming", "first_stable_warning_seconds")) <= 1.0,
    }
    parity_metrics = [
        ("detector_recall", ("detector", "full_road", "recall")),
        ("small_recall", ("detector", "small", "recall")),
        ("conditional_macro_f1", ("conditional_classifier", "macro_f1")),
        ("end_to_end_macro_f1", ("end_to_end", "macro_f1")),
    ]
    parity_differences: dict[str, float | None] = {}
    for name, path in parity_metrics:
        values = pair_values(*path)
        parity_differences[name] = abs(values[0] - values[1]) if values is not None else None
    checks["gpu_cpu_metric_parity"] = all(
        value is not None and value <= 0.005 for value in parity_differences.values()
    )
    checks["no_profile_specific_safety_error"] = nested(
        gpu, "safety", "error_signatures"
    ) == nested(cpu, "safety", "error_signatures")
    failures = [name for name, passed in checks.items() if not passed]
    coursework = cast(dict[str, Any], nested(cpu, "runtime", "coursework"))
    coursework_images = int(coursework["images"])
    recorded_semantic_accuracy = coursework.get("semantic_accuracy")
    coursework_semantic_accuracy = (
        float(recorded_semantic_accuracy)
        if recorded_semantic_accuracy is not None
        else int(coursework["primary_correct"]) / coursework_images
    )
    measured: dict[str, object] = {
        "gpu": {
            "detector_recall": nested(gpu, "detector", "full_road", "recall"),
            "small_sign_recall": nested(gpu, "detector", "small", "recall"),
            "conditional_macro_f1": nested(gpu, "conditional_classifier", "macro_f1"),
            "end_to_end_macro_f1": nested(gpu, "end_to_end", "macro_f1"),
            "safety_critical_recall": nested(gpu, "end_to_end", "safety_critical_recall"),
            "unknown_auroc": nested(gpu, "unknown", "auroc"),
            "unsafe_strong_actions": nested(gpu, "safety", "unsafe_strong_actions"),
            "streaming_fps": nested(gpu, "runtime", "streaming", "frames_per_second"),
            "stable_warning_seconds": nested(
                gpu, "runtime", "streaming", "first_stable_warning_seconds"
            ),
        },
        "cpu": {
            "detector_recall": nested(cpu, "detector", "full_road", "recall"),
            "small_sign_recall": nested(cpu, "detector", "small", "recall"),
            "conditional_macro_f1": nested(cpu, "conditional_classifier", "macro_f1"),
            "end_to_end_macro_f1": nested(cpu, "end_to_end", "macro_f1"),
            "safety_critical_recall": nested(cpu, "end_to_end", "safety_critical_recall"),
            "unknown_auroc": nested(cpu, "unknown", "auroc"),
            "unsafe_strong_actions": nested(cpu, "safety", "unsafe_strong_actions"),
        },
        "no_sign": cpu["no_sign"],
        "coursework": {
            "images": coursework_images,
            "semantic_accuracy": coursework_semantic_accuracy,
            "numeric_accuracy": coursework["numeric_accuracy"],
            "unsafe_strong_actions": coursework["unsafe_strong_actions"],
            "maximum_runtime_ms": nested(cpu, "runtime", "coursework", "runtime_ms", "maximum"),
        },
    }
    decision: dict[str, object] = {
        "schema_version": "1.0",
        "phase": "E",
        "benchmark_id": BENCHMARK_ID,
        "gate_e_passed": not failures,
        "decision": "promote" if not failures else "do_not_promote",
        "checks": checks,
        "failures": failures,
        "passed_gate_count": sum(checks.values()),
        "total_gate_count": len(checks),
        "profile_metric_absolute_differences": parity_differences,
        "measured": measured,
        "test_retuning_allowed": False,
        "runtime_mutated": False,
        "assignment_only": True,
        "publication_prohibited": True,
        "decided_at": datetime.now(UTC).isoformat(),
    }
    gate_path = _resolve(root, GATE_PATH)
    _write_json(gate_path, decision)
    gpu_measured = cast(dict[str, Any], measured["gpu"])
    cpu_measured = cast(dict[str, Any], measured["cpu"])
    coursework_measured = cast(dict[str, Any], measured["coursework"])
    (gate_path.parent / "gate_report.md").write_text(
        "# Phase E gate decision\n\n"
        f"**Decision:** {decision['decision']}\n\n"
        "## Measured results\n\n"
        f"- Detector recall (GPU/CPU): {gpu_measured['detector_recall']:.4f} / "
        f"{cpu_measured['detector_recall']:.4f}\n"
        f"- Small-sign recall (GPU/CPU): {gpu_measured['small_sign_recall']:.4f} / "
        f"{cpu_measured['small_sign_recall']:.4f}\n"
        f"- Conditional macro-F1 (GPU/CPU): {gpu_measured['conditional_macro_f1']:.4f} / "
        f"{cpu_measured['conditional_macro_f1']:.4f}\n"
        f"- End-to-end macro-F1 (GPU/CPU): {gpu_measured['end_to_end_macro_f1']:.4f} / "
        f"{cpu_measured['end_to_end_macro_f1']:.4f}\n"
        f"- Safety-critical recall (GPU/CPU): {gpu_measured['safety_critical_recall']:.4f} / "
        f"{cpu_measured['safety_critical_recall']:.4f}\n"
        f"- Unknown AUROC (GPU/CPU): {gpu_measured['unknown_auroc']:.4f} / "
        f"{cpu_measured['unknown_auroc']:.4f}\n"
        f"- GPU streaming FPS / stable warning: {gpu_measured['streaming_fps']:.2f} / "
        f"{gpu_measured['stable_warning_seconds']:.3f} s\n"
        f"- Coursework semantic / numeric accuracy: "
        f"{coursework_measured['semantic_accuracy']:.4f} / "
        f"{coursework_measured['numeric_accuracy']:.4f}\n\n"
        "## Gate checks\n\n"
        + "\n".join(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
        + "\n",
        encoding="utf-8",
    )
    return decision


def phase_e_status(*, project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    root = project_root.resolve()
    benchmark = phase_e_benchmark_status(project_root=root)
    profiles: dict[str, object] = {}
    for profile in ("gpu", "cpu"):
        path = _resolve(root, PHASE_E_OUTPUT / profile / "metrics.json")
        profiles[profile] = {
            "state": "evaluated" if path.is_file() else "pending",
            "metrics_path": _project_rel(root, path),
        }
    gate = _resolve(root, GATE_PATH)
    gate_detail: dict[str, object]
    if gate.is_file():
        gate_payload = _read_json(gate)
        gate_detail = {
            "state": "finalized",
            "decision": gate_payload.get("decision", "unknown"),
            "gate_e_passed": gate_payload.get("gate_e_passed", False),
        }
    else:
        gate_detail = {"state": "pending", "decision": "pending"}
    runtime_manifest = root / "models/exported/runtime/pipeline.runtime.json"
    if runtime_manifest.is_file():
        state = "promoted"
    elif gate_detail.get("decision") == "do_not_promote":
        state = "complete_not_promoted"
    elif benchmark.get("status") == "owner_review_required":
        state = "awaiting_owner_review"
    else:
        state = "in_progress"
    status: dict[str, object] = {
        "state": state,
        "benchmark": {"state": benchmark.get("status", "unknown"), **benchmark},
        "profiles": profiles,
        "gate": gate_detail,
        "runtime_promoted": runtime_manifest.is_file(),
    }
    return status


def promote_phase_e_runtime(
    *,
    project_root: Path = PROJECT_ROOT,
    internal_only: bool = False,
) -> dict[str, object]:
    if not internal_only:
        raise ValueError("Phase E artifacts are assignment-only; --internal-only is required")
    root = project_root.resolve()
    gate_path = _resolve(root, GATE_PATH)
    gate = _read_json(gate_path)
    if gate.get("gate_e_passed") is not True or gate.get("decision") != "promote":
        raise ValueError("Phase E runtime promotion is blocked because Gate E did not pass")
    preflight = validate_phase_e_artifacts(project_root=root, require_benchmark=True)
    runtime = root / "models/exported/runtime"
    backup_parent = root / "models/exported/runtime_backups"
    backup_parent.mkdir(parents=True, exist_ok=True)
    backup_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_parent / backup_id
    if backup.exists():
        raise FileExistsError(backup)
    staging = root / "models/exported" / f".phase_e_staging_{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    classifier_files = (
        "sign_classifier.onnx",
        "sign_classifier.labels.json",
        "sign_classifier.calibration.json",
        "sign_classifier.runtime.json",
    )
    runtime_hashes_before = _tree_hashes(runtime)
    try:
        for name in classifier_files:
            source = runtime / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, staging / name)
        candidate = root / "models/candidates/phase_d" / DETECTOR_ID
        shutil.copy2(candidate / "model.pt", staging / "sign_detector.pt")
        shutil.copy2(candidate / "model.onnx", staging / "sign_detector.onnx")
        detector_manifest = {
            "schema_version": "1.0",
            "source_candidate": DETECTOR_ID,
            "task": "detect",
            "image_size": DETECTOR_IMAGE_SIZE,
            "confidence_threshold": DETECTOR_CONFIDENCE,
            "nms_iou_threshold": 0.50,
            "gpu_artifact": {"path": "sign_detector.pt", "sha256": DETECTOR_PT_SHA256},
            "cpu_artifact": {"path": "sign_detector.onnx", "sha256": DETECTOR_ONNX_SHA256},
            "assignment_only": True,
            "publication_prohibited": True,
        }
        _write_json(staging / "sign_detector.runtime.json", detector_manifest)
        pipeline_manifest = {
            "schema_version": "1.0",
            "phase": "E",
            "gate_e_passed": True,
            "gate_report": _project_rel(root, gate_path),
            "gate_report_sha256": _sha256(gate_path),
            "benchmark": preflight["benchmark"],
            "detector": detector_manifest,
            "classifier_source": CLASSIFIER_ID,
            "classifier_threshold": CLASSIFIER_THRESHOLD,
            "classifier_artifacts": {
                "onnx": {
                    "path": "sign_classifier.onnx",
                    "sha256": CLASSIFIER_ONNX_SHA256,
                },
                "labels": {
                    "path": "sign_classifier.labels.json",
                    "sha256": CLASSIFIER_LABELS_SHA256,
                },
                "calibration": {
                    "path": "sign_classifier.calibration.json",
                    "sha256": CLASSIFIER_CALIBRATION_SHA256,
                },
            },
            "crop_padding": CROP_PADDING,
            "preferred_profile": "auto",
            "internal_academic_only": True,
            "publication_prohibited": True,
            "dvc_remote_push_allowed": False,
            "backup_id": backup_id,
            "promoted_at": datetime.now(UTC).isoformat(),
        }
        _write_json(staging / "pipeline.runtime.json", pipeline_manifest)
        if (
            _sha256(staging / "sign_detector.pt") != DETECTOR_PT_SHA256
            or _sha256(staging / "sign_detector.onnx") != DETECTOR_ONNX_SHA256
            or _sha256(staging / "sign_classifier.onnx") != CLASSIFIER_ONNX_SHA256
            or _sha256(staging / "sign_classifier.labels.json") != CLASSIFIER_LABELS_SHA256
            or _sha256(staging / "sign_classifier.calibration.json")
            != CLASSIFIER_CALIBRATION_SHA256
        ):
            raise RuntimeError("Staged Phase E artifact hashes do not match the reviewed bundle")
        dvc_pointer = root / "models/exported/runtime.dvc"
        dvc_before = _sha256(dvc_pointer) if dvc_pointer.is_file() else None
        runtime.rename(backup)
        try:
            staging.rename(runtime)
            smoke = InferenceEngine(PHASE_E_RUNTIME_CONFIG)
            warmup = smoke.warmup()
            if not warmup.get("detector") or not warmup.get("classifier"):
                raise RuntimeError(f"Promoted Phase E runtime warm-up failed: {warmup}")
            smoke.process_frame(
                cast(UInt8Image, np.zeros((640, 640, 3), dtype=np.uint8)), assume_stable=True
            )
        except Exception:
            failed = root / "models/exported" / f"runtime_failed_{backup_id}"
            if runtime.exists():
                runtime.rename(failed)
            backup.rename(runtime)
            raise
        dvc_after = _sha256(dvc_pointer) if dvc_pointer.is_file() else None
        if dvc_before != dvc_after:
            failed = root / "models/exported" / f"runtime_failed_dvc_{backup_id}"
            runtime.rename(failed)
            backup.rename(runtime)
            raise RuntimeError("Phase E promotion unexpectedly modified runtime.dvc")
        _write_json(
            backup_parent / f"{backup_id}.json",
            {
                "schema_version": "1.0",
                "backup_id": backup_id,
                "reason": "phase_e_atomic_internal_runtime_promotion",
                "restorable": True,
                "runtime_hashes": runtime_hashes_before,
            },
        )
        result = dict(pipeline_manifest)
        result["pipeline_manifest_sha256"] = _sha256(runtime / "pipeline.runtime.json")
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def rollback_phase_e_runtime(
    backup_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    if len(backup_id) != 16 or not backup_id.endswith("Z"):
        raise ValueError("Backup ID must be the UTC timestamp generated by Phase E promotion")
    root = project_root.resolve()
    backup = root / "models/exported/runtime_backups" / backup_id
    backup_manifest = backup.parent / f"{backup_id}.json"
    if not backup_manifest.is_file() or _read_json(backup_manifest).get("restorable") is not True:
        raise ValueError(f"Phase E backup is missing or not restorable: {backup}")
    runtime = root / "models/exported/runtime"
    staging = root / "models/exported" / f".phase_e_rollback_{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(backup, staging, dirs_exist_ok=True)
        expected_hashes = cast(dict[str, str], _read_json(backup_manifest)["runtime_hashes"])
        if _tree_hashes(staging) != expected_hashes:
            raise RuntimeError("Phase E backup hashes do not match the promotion-time snapshot")
        required = {
            "sign_classifier.onnx",
            "sign_classifier.labels.json",
            "sign_classifier.calibration.json",
            "sign_classifier.runtime.json",
        }
        if not required <= {path.name for path in staging.iterdir()}:
            raise ValueError("Phase E backup does not contain a complete classifier runtime")
        replaced = (
            root
            / "models/exported"
            / f"runtime_replaced_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        runtime.rename(replaced)
        try:
            staging.rename(runtime)
        except Exception:
            replaced.rename(runtime)
            raise
        if _tree_hashes(runtime) != expected_hashes:
            failed = root / "models/exported" / f"runtime_failed_rollback_{backup_id}"
            runtime.rename(failed)
            replaced.rename(runtime)
            raise RuntimeError("Restored Phase E backup failed its post-rollback hash check")
        return {
            "schema_version": "1.0",
            "restored_backup_id": backup_id,
            "replaced_runtime": _project_rel(root, replaced),
            "runtime_root": _project_rel(root, runtime),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)
