from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
from PIL import ExifTags, Image

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.classification.onnx_backend import ONNXSignClassifier
from roadsign_assist.detection.hybrid_backend import HybridSignDetector
from roadsign_assist.detection.ultralytics_backend import UltralyticsDetector
from roadsign_assist.inference.engine import InferenceEngine, crop_detection, decode_image
from roadsign_assist.inference.models import BoundingBoxModel, DetectionModel, FrameResultModel
from roadsign_assist.paths import PROJECT_ROOT, project_path

INCIDENT_SCHEMA_VERSION = "1.0"
DEFAULT_THRESHOLDS = (0.01, 0.05, 0.10, 0.20)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/incidents/legacy_replacement_r0_20260904"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _git_metadata() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--porcelain=v1")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "clean_worktree": not status,
            "status_porcelain": status.splitlines(),
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc), "clean_worktree": False}


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
    }
    for name in ("numpy", "Pillow", "ultralytics", "onnxruntime", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _source_metadata(data: bytes) -> dict[str, object]:
    from io import BytesIO

    with Image.open(BytesIO(data)) as source:
        orientation_tag = next(
            (key for key, value in ExifTags.TAGS.items() if value == "Orientation"),
            274,
        )
        exif = source.getexif()
        return {
            "format": source.format,
            "mode": source.mode,
            "encoded_width": source.width,
            "encoded_height": source.height,
            "exif_orientation": exif.get(orientation_tag),
        }


def normalized_thresholds(
    configured_threshold: float,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[float]:
    values = {float(configured_threshold), *(float(value) for value in thresholds)}
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Detector thresholds must be between zero and one")
    return sorted(values)


def _validate_human_box(box: BoundingBoxModel, image: UInt8Image) -> None:
    height, width = image.shape[:2]
    if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
        raise ValueError(
            f"Human box {box.model_dump()} falls outside decoded image {width}x{height}"
        )


def _unwrap_detector(engine: InferenceEngine) -> UltralyticsDetector:
    detector = engine.detector
    if isinstance(detector, HybridSignDetector):
        detector = detector.primary
    if not isinstance(detector, UltralyticsDetector):
        raise TypeError(
            "R0 threshold sweeps require an Ultralytics detector; "
            f"received {type(detector).__name__}"
        )
    return detector


def _classification_diagnostics(
    classifier: object,
    crop: UInt8Image,
) -> dict[str, object]:
    prediction = classifier.classify(crop)  # type: ignore[attr-defined]
    detail: dict[str, object] = {
        "prediction": prediction.model_dump(mode="json"),
        "raw_logits": None,
        "calibrated_logits": None,
        "temperature": None,
    }
    if not isinstance(classifier, ONNXSignClassifier):
        detail["diagnostic_note"] = "Raw logits are available only for the ONNX classifier."
        return detail

    session = classifier._load()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: classifier._preprocess(crop)})
    logits = np.asarray(outputs[0], dtype=np.float64)[0]
    temperature = float(classifier._temperature)
    detail.update(
        {
            "raw_logits": logits.tolist(),
            "calibrated_logits": (logits / temperature).tolist(),
            "temperature": temperature,
            "confidence_threshold": classifier.confidence_threshold,
            "input_size": classifier.image_size,
            "providers": list(classifier.active_providers),
        }
    )
    return detail


def _detection_payload(detection: DetectionModel) -> dict[str, object]:
    return detection.model_dump(mode="json")


def _draw_overlay(
    image: UInt8Image,
    detections: Sequence[DetectionModel],
    *,
    title: str,
    human_box: BoundingBoxModel,
) -> UInt8Image:
    canvas = image.copy()
    hx1, hy1, hx2, hy2 = (int(value) for value in human_box.model_dump().values())
    cv2.rectangle(canvas, (hx1, hy1), (hx2, hy2), (255, 0, 255), 3)
    cv2.putText(
        canvas,
        "human box",
        (hx1, max(18, hy1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    for detection in detections:
        box = detection.bbox
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        color = (0, 200, 0) if box.iou(human_box) >= 0.50 else (0, 180, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{detection.confidence:.3f} iou={box.iou(human_box):.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _write_image(path: Path, image: UInt8Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV could not write {path}")


def _artifact_identity(engine: InferenceEngine, config_path: Path) -> dict[str, object]:
    paths: dict[str, Path | None] = {
        "config": config_path,
        "detector": getattr(_unwrap_detector(engine), "model_path", None),
        "classifier": getattr(engine.classifier, "model_path", None),
        "classifier_labels": getattr(engine.classifier, "labels_path", None),
        "classifier_calibration": getattr(engine.classifier, "calibration_path", None),
    }
    artifacts: dict[str, object] = {}
    for name, path in paths.items():
        artifacts[name] = (
            {
                "path": _project_relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            if path is not None and path.is_file()
            else {"path": _project_relative(path) if path is not None else None, "missing": True}
        )
    lock = PROJECT_ROOT / "uv.lock"
    if lock.is_file():
        artifacts["dependency_lock"] = {
            "path": _project_relative(lock),
            "size_bytes": lock.stat().st_size,
            "sha256": _sha256_file(lock),
        }
    return artifacts


def _event_signature(result: FrameResultModel | dict[str, Any]) -> list[dict[str, object]]:
    payload = result.model_dump(mode="json") if isinstance(result, FrameResultModel) else result
    events = cast(list[dict[str, Any]], payload.get("events", []))
    return [
        {
            "semantic_sign_id": event.get("semantic_sign_id"),
            "confidence": round(float(event.get("confidence", 0.0)), 6),
            "bbox": {
                key: round(float(cast(dict[str, Any], event["bbox"])[key]), 3)
                for key in ("x1", "y1", "x2", "y2")
            },
            "stable": bool(event.get("stable")),
            "should_announce": bool(event.get("should_announce")),
        }
        for event in events
    ]


def compare_direct_and_website(
    direct: FrameResultModel,
    website_result: dict[str, Any],
) -> dict[str, object]:
    direct_signature = _event_signature(direct)
    website_signature = _event_signature(website_result)
    return {
        "matches": direct_signature == website_signature,
        "direct_signature": direct_signature,
        "website_signature": website_signature,
    }


def compare_model_status(
    direct_status: dict[str, object],
    website_status: dict[str, Any],
) -> dict[str, object]:
    direct_detector = cast(dict[str, object], direct_status.get("detector_profile", {}))
    website_detector = cast(dict[str, object], website_status.get("detector_profile", {}))
    direct_classifier = cast(dict[str, object], direct_status.get("classifier_profile", {}))
    website_classifier = cast(dict[str, object], website_status.get("classifier_profile", {}))
    fields = {
        "mode": (direct_status.get("mode"), website_status.get("mode")),
        "detector_model_path": (
            direct_detector.get("model_path"),
            website_detector.get("model_path"),
        ),
        "detector_artifact_sha256": (
            direct_detector.get("artifact_sha256"),
            website_detector.get("artifact_sha256"),
        ),
        "detector_image_size": (
            direct_detector.get("image_size"),
            website_detector.get("image_size"),
        ),
        "detector_confidence_threshold": (
            direct_detector.get("confidence_threshold"),
            website_detector.get("confidence_threshold"),
        ),
        "classifier_model_path": (
            direct_classifier.get("model_path"),
            website_classifier.get("model_path"),
        ),
        "classifier_artifact_sha256": (
            direct_classifier.get("artifact_sha256"),
            website_classifier.get("artifact_sha256"),
        ),
        "classifier_image_size": (
            direct_classifier.get("image_size"),
            website_classifier.get("image_size"),
        ),
        "classifier_confidence_threshold": (
            direct_classifier.get("confidence_threshold"),
            website_classifier.get("confidence_threshold"),
        ),
    }
    comparisons = {
        name: {"direct": values[0], "website": values[1], "matches": values[0] == values[1]}
        for name, values in fields.items()
    }
    return {
        "matches": all(value["matches"] for value in comparisons.values()),
        "fields": comparisons,
    }


def _website_inference(
    base_url: str,
    source_name: str,
    data: bytes,
    *,
    verify_tls: bool,
) -> dict[str, object]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Website URL must be an absolute http(s) URL")
    root = base_url.rstrip("/")
    with httpx.Client(verify=verify_tls, timeout=180.0) as client:
        model_response = client.get(f"{root}/api/v1/models")
        model_response.raise_for_status()
        response = client.post(
            f"{root}/api/v1/infer/image",
            files={"file": (source_name, data, "application/octet-stream")},
        )
        response.raise_for_status()
    payload = cast(dict[str, Any], response.json())
    return {
        "base_url": base_url,
        "request_bytes": len(data),
        "request_sha256": _sha256_bytes(data),
        "model_status": model_response.json(),
        "result": cast(dict[str, Any], payload["result"]),
    }


def classify_failure_locations(
    *,
    expected_label: str,
    human_box: BoundingBoxModel,
    deployed_threshold: float,
    sweep: dict[str, list[dict[str, object]]],
    proposal_classifications: dict[str, dict[str, object]],
    tracking_results: Sequence[dict[str, Any]],
    website_comparison: dict[str, object] | None,
    human_classification: dict[str, object] | None = None,
    minimum_iou: float = 0.50,
) -> list[str]:
    threshold_values = sorted(float(value) for value in sweep)
    lowest = sweep[f"{threshold_values[0]:.6f}"]
    deployed = sweep[f"{deployed_threshold:.6f}"]

    def overlapping(values: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        return [
            value
            for value in values
            if BoundingBoxModel.model_validate(value["bbox"]).iou(human_box) >= minimum_iou
        ]

    low_overlap = overlapping(lowest)
    deployed_overlap = overlapping(deployed)
    locations: list[str] = []
    deployed_semantics_succeeded = False

    def prediction_succeeded(detail: dict[str, object]) -> bool:
        prediction = cast(dict[str, Any], detail.get("prediction", {}))
        return bool(
            prediction.get("semantic_sign_id") == expected_label
            and prediction.get("accepted") is True
        )

    if not low_overlap:
        locations.append("detector_representation")
    elif not deployed_overlap:
        locations.append("detector_calibration")
    else:
        best = max(
            deployed_overlap,
            key=lambda value: float(cast(float, value["confidence"])),
        )
        classification = proposal_classifications.get(str(best["detection_id"]), {})
        deployed_semantics_succeeded = prediction_succeeded(classification)

    classifier_failed = bool(
        human_classification is not None
        and not prediction_succeeded(human_classification)
    )
    if low_overlap:
        best_low = max(
            low_overlap,
            key=lambda value: (
                BoundingBoxModel.model_validate(value["bbox"]).iou(human_box),
                float(cast(float, value["confidence"])),
            ),
        )
        best_low_classification = proposal_classifications.get(
            str(best_low["detection_id"]),
            {},
        )
        if best_low_classification:
            classifier_failed = classifier_failed or not prediction_succeeded(
                best_low_classification
            )
    if classifier_failed:
        locations.append("classifier")

    if website_comparison is not None and website_comparison.get("matches") is not True:
        locations.append("runtime_integration")

    if deployed_semantics_succeeded:
        final_events = tracking_results[-1].get("events", []) if tracking_results else []
        if not any(event.get("semantic_sign_id") == expected_label for event in final_events):
            locations.append("tracking_advisory")
    return locations


def _run_profile(
    *,
    name: str,
    config_path: Path,
    runtime_profile: str | None,
    image: UInt8Image,
    human_box: BoundingBoxModel,
    expected_label: str,
    output: Path,
    thresholds: Sequence[float],
    tracking_frames: int,
    website_url: str | None,
    website_verify_tls: bool,
    source_name: str,
    source_data: bytes,
) -> dict[str, object]:
    engine = InferenceEngine(config_path, runtime_profile=runtime_profile)
    detector = _unwrap_detector(engine)
    deployed_threshold = detector.confidence_threshold
    sweep_thresholds = normalized_thresholds(deployed_threshold, thresholds)
    profile_root = output / name
    profile_root.mkdir(parents=True, exist_ok=False)

    sweep: dict[str, list[dict[str, object]]] = {}
    detections_by_threshold: dict[float, list[DetectionModel]] = {}
    for threshold in sweep_thresholds:
        detector.confidence_threshold = threshold
        detections = detector.detect(image)
        detections_by_threshold[threshold] = detections
        key = f"{threshold:.6f}"
        sweep[key] = [_detection_payload(detection) for detection in detections]
        _write_image(
            profile_root / "overlays" / f"threshold_{threshold:.6f}.jpg",
            _draw_overlay(
                image,
                detections,
                title=f"{name} detector threshold={threshold:.3f}",
                human_box=human_box,
            ),
        )

    detector.confidence_threshold = deployed_threshold
    lowest_detections = detections_by_threshold[sweep_thresholds[0]]
    proposal_classifications: dict[str, dict[str, object]] = {}
    for index, detection in enumerate(lowest_detections):
        crop = crop_detection(image, detection.bbox, padding=engine.crop_padding)
        crop_name = f"proposal_{index:03d}_{detection.confidence:.6f}.jpg"
        _write_image(profile_root / "crops" / crop_name, crop)
        proposal_classifications[detection.detection_id] = {
            "crop": f"crops/{crop_name}",
            "source_detection": _detection_payload(detection),
            "human_box_iou": detection.bbox.iou(human_box),
            **_classification_diagnostics(engine.classifier, crop),
        }

    human_crop = crop_detection(image, human_box, padding=0.0)
    _write_image(profile_root / "crops" / "human_crop.jpg", human_crop)
    human_classification = {
        "crop": "crops/human_crop.jpg",
        "expected_label": expected_label,
        **_classification_diagnostics(engine.classifier, human_crop),
    }

    direct_session = engine.new_session()
    direct_result = direct_session.process_frame(image, assume_stable=True)
    tracking_session = engine.new_session()
    tracking_results = [
        tracking_session.process_frame(image, assume_stable=False).model_dump(mode="json")
        for _ in range(tracking_frames)
    ]

    website: dict[str, object] | None = None
    website_comparison: dict[str, object] | None = None
    if website_url:
        website = _website_inference(
            website_url,
            source_name,
            source_data,
            verify_tls=website_verify_tls,
        )
        output_comparison = compare_direct_and_website(
            direct_result,
            cast(dict[str, Any], website["result"]),
        )
        identity_comparison = compare_model_status(
            engine.model_status,
            cast(dict[str, Any], website["model_status"]),
        )
        website_comparison = {
            "matches": bool(
                output_comparison["matches"] and identity_comparison["matches"]
            ),
            "output": output_comparison,
            "model_identity": identity_comparison,
        }

    locations = classify_failure_locations(
        expected_label=expected_label,
        human_box=human_box,
        deployed_threshold=deployed_threshold,
        sweep=sweep,
        proposal_classifications=proposal_classifications,
        tracking_results=tracking_results,
        website_comparison=website_comparison,
        human_classification=human_classification,
    )
    status = engine.model_status
    return {
        "name": name,
        "config_path": _project_relative(config_path),
        "runtime_profile_requested": runtime_profile,
        "deployed_threshold": deployed_threshold,
        "thresholds": sweep_thresholds,
        "threshold_sweep": sweep,
        "proposal_classifications": proposal_classifications,
        "human_classification": human_classification,
        "tracking_disabled_result": {
            "description": "Raw detector proposals and independent crop classifications; no "
            "track state or temporal fusion applied.",
            "deployed_proposals": sweep[f"{deployed_threshold:.6f}"],
        },
        "direct_assume_stable_result": direct_result.model_dump(mode="json"),
        "tracking_enabled_results": tracking_results,
        "website": website,
        "website_comparison": website_comparison,
        "failure_locations": locations,
        "model_status": status,
        "artifacts": _artifact_identity(engine, config_path),
        "preprocessing": {
            "decode": "Pillow decode -> EXIF transpose -> RGB -> contiguous OpenCV BGR uint8",
            "detector_input": {
                "array_shape": list(image.shape),
                "dtype": str(image.dtype),
                "color_order": "BGR",
                "configured_image_size": detector.image_size,
                "ultralytics_imgsz_argument_explicit": True,
                "resize_and_letterbox_owner": "Ultralytics",
                "nms_iou_threshold": detector.nms_iou_threshold,
                "device_argument": detector.device,
            },
            "classifier_input": {
                "color_conversion": "BGR to RGB",
                "resize": "OpenCV INTER_AREA square resize",
                "configured_image_size": getattr(engine.classifier, "configured_image_size", None),
                "effective_image_size": getattr(engine.classifier, "image_size", None),
                "normalization": "ImageNet mean/std",
            },
        },
    }


def _report_markdown(bundle: dict[str, Any]) -> str:
    profiles = cast(dict[str, dict[str, Any]], bundle["profiles"])
    lines = [
        "# Legacy replacement R0 incident report",
        "",
        f"**Incident ID:** `{bundle['incident_id']}`  ",
        f"**Generated:** {bundle['generated_at']}  ",
        f"**Gate R0:** {'PASS' if bundle['gate_r0_passed'] else 'INCOMPLETE'}  ",
        "",
        "## Input",
        "",
        f"- Preserved bytes: `{bundle['source']['preserved_path']}`",
        f"- SHA-256: `{bundle['source']['sha256']}`",
        f"- Expected semantic label: `{bundle['ground_truth']['expected_label']}`",
        f"- Human box: `{bundle['ground_truth']['human_box']}`",
        "",
        "## Failure localization",
        "",
    ]
    for name, profile in profiles.items():
        locations = profile.get("failure_locations", [])
        lines.append(f"- **{name}:** {', '.join(locations) if locations else 'not localized'}")
        comparison = profile.get("website_comparison")
        if comparison is None:
            lines.append("  - Website parity: not run")
        else:
            lines.append(
                "  - Website parity: "
                + ("matched" if comparison.get("matches") else "mismatched")
            )
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            "R0 passes only when both profiles ran, at least one failure location was identified "
            "for each profile, and the candidate was compared with a live website/API path using "
            "the exact preserved bytes.",
            "",
            "This report is diagnostic evidence only. It does not authorize retraining or model "
            "promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def run_recovery_incident(
    *,
    image_path: str | Path,
    human_box: BoundingBoxModel,
    expected_label: str,
    legacy_config: str | Path,
    candidate_config: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    legacy_profile: str | None = None,
    candidate_profile: str | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    tracking_frames: int = 3,
    candidate_website_url: str | None = None,
    website_verify_tls: bool = True,
    capture_domain: str = "electronic_screen",
    camera_orientation: str = "unknown",
    browser_resize: str = "unknown",
) -> dict[str, object]:
    if tracking_frames < 1:
        raise ValueError("tracking_frames must be at least one")
    source = project_path(image_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    legacy = project_path(legacy_config)
    candidate = project_path(candidate_config)
    for config in (legacy, candidate):
        if not config.is_file():
            raise FileNotFoundError(config)
    output = project_path(output_path)
    if output.exists():
        raise FileExistsError(
            f"Incident output already exists; select a new immutable output path: {output}"
        )

    source_data = source.read_bytes()
    image = decode_image(source_data)
    _validate_human_box(human_box, image)
    output.mkdir(parents=True, exist_ok=False)
    preserved = output / "source" / f"original{source.suffix.casefold() or '.bin'}"
    preserved.parent.mkdir(parents=True, exist_ok=False)
    preserved.write_bytes(source_data)

    source_metadata = _source_metadata(source_data)
    source_metadata.update(
        {
            "original_path": str(source.resolve()),
            "preserved_path": _project_relative(preserved),
            "size_bytes": len(source_data),
            "sha256": _sha256_bytes(source_data),
            "decoded_width": int(image.shape[1]),
            "decoded_height": int(image.shape[0]),
            "decoded_color_order": "BGR",
            "capture_domain": capture_domain,
            "camera_orientation": camera_orientation,
            "browser_resize": browser_resize,
        }
    )

    profiles: dict[str, dict[str, object]] = {}
    profile_specs = (
        ("legacy", legacy, legacy_profile, None),
        ("candidate", candidate, candidate_profile, candidate_website_url),
    )
    for name, config, runtime_profile, website_url in profile_specs:
        try:
            profiles[name] = _run_profile(
                name=name,
                config_path=config,
                runtime_profile=runtime_profile,
                image=image,
                human_box=human_box,
                expected_label=expected_label,
                output=output,
                thresholds=thresholds,
                tracking_frames=tracking_frames,
                website_url=website_url,
                website_verify_tls=website_verify_tls,
                source_name=source.name,
                source_data=source_data,
            )
        except Exception as exc:
            profiles[name] = {
                "name": name,
                "config_path": _project_relative(config),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failure_locations": [],
            }

    both_ran = all("error" not in profile for profile in profiles.values())
    localized = all(bool(profile.get("failure_locations")) for profile in profiles.values())
    candidate_parity = cast(
        dict[str, object] | None,
        profiles.get("candidate", {}).get("website_comparison"),
    )
    bundle: dict[str, object] = {
        "schema_version": INCIDENT_SCHEMA_VERSION,
        "incident_id": output.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "gate_r0_passed": bool(both_ran and localized and candidate_parity is not None),
        "source": source_metadata,
        "ground_truth": {
            "expected_label": expected_label,
            "human_box": human_box.model_dump(mode="json"),
        },
        "profiles": profiles,
        "environment": {
            "git": _git_metadata(),
            "packages": _package_versions(),
            "argv": sys.argv,
        },
        "limitations": [
            *(
                []
                if candidate_website_url
                else ["Candidate live website/API parity was not run (no URL supplied)."]
            ),
            "Browser camera resizing must be supplied as metadata; upload inference sends exact bytes.",
            "Phase A preprocessing changes occurred after the preserved R0 failure was localized; "
            "the R0 bundle remains evidence for the pre-change runtime.",
        ],
    }
    _write_json(output / "incident.json", bundle)
    (output / "report.md").write_text(
        _report_markdown(cast(dict[str, Any], bundle)),
        encoding="utf-8",
    )
    return bundle
