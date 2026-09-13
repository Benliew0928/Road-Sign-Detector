"""Phase B3 automated proposals and semantic triage for quarantined candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from roadsign_assist.classification.onnx_backend import ONNXSignClassifier
from roadsign_assist.datasets.recovery_acquisition import (
    DEFAULT_ACQUISITION_ROOT,
    DEFAULT_AUDIT_ROOT,
)
from roadsign_assist.datasets.recovery_normalize import NormalizedCandidate
from roadsign_assist.datasets.recovery_phase_b import PHASE_B_FIELDS
from roadsign_assist.detection.ultralytics_backend import UltralyticsDetector
from roadsign_assist.inference.models import ClassificationModel, DetectionModel
from roadsign_assist.paths import project_path

PROPOSAL_VERSION = "recovery_proposals_v1"
DEFAULT_DETECTOR = Path("models/candidates/phase_d/pd_v1_yolo26s_640_s2513/model.pt")
DEFAULT_CLASSIFIER = Path(
    "models/exported/experimental/stage_e_current_efficientnet_v2_s_embedding_q97.onnx"
)
DEFAULT_LABELS = Path("models/exported/experimental/stage_e_current_efficientnet_v2_s.labels.json")
DEFAULT_CALIBRATION = Path(
    "models/exported/experimental/stage_e_current_efficientnet_v2_s_embedding_q97.calibration.json"
)


class DetectorProtocol(Protocol):
    @property
    def name(self) -> str: ...

    def detect(self, image: np.ndarray[Any, np.dtype[np.uint8]]) -> list[DetectionModel]: ...


class ClassifierProtocol(Protocol):
    @property
    def name(self) -> str: ...

    def classify(self, crop: np.ndarray[Any, np.dtype[np.uint8]]) -> ClassificationModel: ...


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    object_id: str
    source_id: str
    group_id: str
    image_path: str
    image_sha256: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    bbox_xyxy: list[float]
    area_ratio: float = Field(ge=0.0, le=1.0)
    size_bucket: str
    detector_confidence: float = Field(ge=0.0, le=1.0)
    detector_members: list[str]
    semantic_sign_id: str
    classifier_confidence: float = Field(ge=0.0, le=1.0)
    classifier_accepted: bool
    classifier_top_k: list[dict[str, object]]
    classifier_unknown_score: float | None = None
    embedding_distance: float | None = None
    nearest_prototype: str = ""
    rejection_reasons: list[str]
    semantic_review_flags: list[str]
    candidate_state: str
    priority_score: int = Field(ge=0, le=100)
    licence_status: str = "unreviewed"
    permitted_use: str = "none_pending_review"
    review_status: str = "pending"


class ImageProposalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_batch_id: str
    processed_at: datetime
    object_id: str
    source_id: str
    group_id: str
    image_path: str
    image_sha256: str
    image_width: int = Field(default=0, ge=0)
    image_height: int = Field(default=0, ge=0)
    proposals: list[Proposal] = Field(default_factory=lambda: list[Proposal]())
    image_state: str
    inference_error: str = ""
    licence_status: str = "unreviewed"
    permitted_use: str = "none_pending_review"
    review_status: str = "pending"


def _now() -> datetime:
    return datetime.now(UTC)


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file():
        return []
    rows: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                rows.append(cast(Mapping[str, object], value))
    return rows


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, default=str) + "\n")
        handle.flush()


def _iou(left: DetectionModel, right: DetectionModel) -> float:
    return left.bbox.iou(right.bbox)


def _fuse_detections(
    detector_results: Sequence[tuple[str, Sequence[DetectionModel]]], *, iou_threshold: float
) -> list[tuple[DetectionModel, list[str]]]:
    flattened = [
        (name, detection)
        for name, detections in detector_results
        for detection in detections
    ]
    flattened.sort(key=lambda item: item[1].confidence, reverse=True)
    fused: list[tuple[DetectionModel, list[str]]] = []
    for name, detection in flattened:
        match = next(
            (index for index, (lead, _members) in enumerate(fused) if _iou(lead, detection) >= iou_threshold),
            None,
        )
        if match is None:
            fused.append((detection, [name]))
        elif name not in fused[match][1]:
            fused[match][1].append(name)
    return fused


def _crop(
    image: np.ndarray[Any, np.dtype[np.uint8]], detection: DetectionModel
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    height, width = image.shape[:2]
    box = detection.bbox
    padding_x = 0.10 * box.width
    padding_y = 0.10 * box.height
    x1 = max(0, int(box.x1 - padding_x))
    y1 = max(0, int(box.y1 - padding_y))
    x2 = min(width, int(box.x2 + padding_x + 0.5))
    y2 = min(height, int(box.y2 + padding_y + 0.5))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("detector produced an empty crop")
    return image[y1:y2, x1:x2]


def _size_bucket(area_ratio: float) -> str:
    if area_ratio <= 0.001:
        return "very_small"
    if area_ratio <= 0.01:
        return "small"
    if area_ratio <= 0.05:
        return "medium"
    return "large"


def _direction(label: str) -> str:
    if "left" in label:
        return "left"
    if "right" in label:
        return "right"
    if "straight" in label:
        return "straight"
    return ""


def _family(label: str) -> str:
    if label.startswith("no_"):
        return "prohibition"
    if any(token in label for token in ("turn", "straight", "keep_", "one_way", "pass_")):
        return "directional"
    if any(token in label for token in ("maximum_speed", "height_", "weight_", "width_")):
        return "numeric_restriction"
    if label in {"stop", "give_way", "no_entry"}:
        return "priority_regulatory"
    if label == "unknown_sign":
        return "unknown"
    return "warning_or_information"


def _semantic_flags(classification: ClassificationModel, size_bucket: str) -> list[str]:
    flags: list[str] = []
    if not classification.accepted:
        flags.append("classifier_rejected_unknown")
    flags.extend(f"classifier_rejection:{reason}" for reason in classification.rejection_reasons)
    if classification.embedding_distance is None:
        flags.append("embedding_distance_unavailable")
    if len(classification.top_k) >= 2:
        first_label, first_score = classification.top_k[0]
        second_label, second_score = classification.top_k[1]
        if first_score - second_score < 0.15:
            flags.append("top1_top2_margin_below_0.15")
        if _family(first_label) != _family(second_label) and first_score - second_score < 0.25:
            flags.append("close_cross_family_conflict")
        left_direction = _direction(first_label)
        right_direction = _direction(second_label)
        if left_direction and right_direction and left_direction != right_direction:
            flags.append("opposite_direction_confusion")
    if size_bucket in {"small", "very_small"}:
        flags.append(f"{size_bucket}_proposal")
    return sorted(set(flags))


def _candidate_state(classification: ClassificationModel, flags: Sequence[str]) -> str:
    if classification.semantic_sign_id == "unknown_sign":
        if classification.confidence < 0.35:
            return "candidate_unreadable"
        return "candidate_unknown"
    if any("conflict" in flag or "confusion" in flag for flag in flags):
        return "candidate_positive_conflict"
    return "candidate_positive_agreement"


def _priority(
    classification: ClassificationModel,
    detector_confidence: float,
    size_bucket: str,
    state: str,
) -> int:
    family = _family(classification.top_k[0][0] if classification.top_k else "unknown_sign")
    score = 30
    if family in {"directional", "numeric_restriction", "priority_regulatory", "prohibition"}:
        score += 30
    if state in {"candidate_positive_conflict", "candidate_unknown", "candidate_unreadable"}:
        score += 20
    if size_bucket in {"small", "very_small"}:
        score += 15
    score += round(5 * detector_confidence)
    return min(100, score)


def _default_models(
    *,
    detector_path: Path,
    classifier_path: Path,
    labels_path: Path,
    calibration_path: Path,
    device: str,
) -> tuple[list[DetectorProtocol], ClassifierProtocol]:
    detectors: list[DetectorProtocol] = [
        UltralyticsDetector(
            detector_path,
            confidence_threshold=0.03,
            nms_iou_threshold=0.55,
            device=device,
            image_size=640,
            profile_name="recovery-sensitive-640",
        ),
        UltralyticsDetector(
            detector_path,
            confidence_threshold=0.05,
            nms_iou_threshold=0.55,
            device=device,
            image_size=960,
            profile_name="recovery-small-sign-960",
        ),
    ]
    classifier: ClassifierProtocol = ONNXSignClassifier(
        classifier_path,
        labels_path,
        calibration_path=calibration_path,
        confidence_threshold=0.72,
        image_size=224,
    )
    return detectors, classifier


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    arrow: Any = pa
    parquet: Any = pq
    table: Any = arrow.Table.from_pylist(rows)
    parquet.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _source_urls(object_ledger: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in _load_jsonl(object_ledger):
        object_id = str(row.get("object_id", ""))
        url = str(row.get("source_url", ""))
        if object_id and url:
            values.setdefault(object_id, url)
    return values


def _write_exports(
    package: Path,
    results: Sequence[ImageProposalResult],
    *,
    proposal_batch_id: str,
    source_urls: Mapping[str, str],
) -> tuple[Path, Path]:
    package.mkdir(parents=True, exist_ok=True)
    coco_path = package / "cvat_coco.json"
    intake_path = package / "canonical_intake_proposals.csv"
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    intake_rows: list[dict[str, object]] = []
    annotation_id = 1
    for image_id, result in enumerate(results, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": result.image_path,
                "width": result.image_width,
                "height": result.image_height,
                "source_object_id": result.object_id,
            }
        )
        proposals: Sequence[Proposal | None] = result.proposals or [None]
        for proposal in proposals:
            expected_kind = "no_sign" if proposal is None else (
                "unknown_sign"
                if proposal.candidate_state in {"candidate_unknown", "candidate_unreadable"}
                else "sign"
            )
            semantic_id = "unknown_sign" if proposal is None else proposal.semantic_sign_id
            instance_id = (
                f"{result.object_id}:no-sign" if proposal is None else proposal.proposal_id
            )
            bbox = [0.0, 0.0, 0.0, 0.0] if proposal is None else proposal.bbox_xyxy
            area_ratio = 0.0 if proposal is None else proposal.area_ratio
            size_bucket = "" if proposal is None else proposal.size_bucket
            if proposal is not None:
                x1, y1, x2, y2 = bbox
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                        "attributes": {
                            "proposal_id": proposal.proposal_id,
                            "candidate_state": proposal.candidate_state,
                            "semantic_sign_id": proposal.semantic_sign_id,
                            "detector_confidence": proposal.detector_confidence,
                            "classifier_confidence": proposal.classifier_confidence,
                            "review_status": "pending",
                        },
                    }
                )
                annotation_id += 1
            values: dict[str, object] = {field: "" for field in PHASE_B_FIELDS}
            values.update(
                {
                    "schema_version": "1.0",
                    "release_id": proposal_batch_id,
                    "sample_id": result.object_id,
                    "instance_id": instance_id,
                    "split": "",
                    "expected_kind": expected_kind,
                    "semantic_sign_id": semantic_id,
                    "capture_domain": "road_scene",
                    "image_path": result.image_path,
                    "label_path": str(coco_path),
                    "image_sha256": result.image_sha256,
                    "image_width": result.image_width,
                    "image_height": result.image_height,
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "area_ratio": area_ratio,
                    "size_bucket": size_bucket,
                    "source_id": result.source_id,
                    "related_capture_group_id": result.group_id,
                    "annotation_status": "automated_proposal_unreviewed",
                    "review_decision": "pending",
                    "review_notes": "Automated Phase B3 proposal; not ground truth or release-approved.",
                    "licence_status": result.licence_status,
                    "use_policy": "none_pending_review",
                    "source_url": source_urls.get(result.object_id, ""),
                }
            )
            intake_rows.append(values)
    coco = {
        "info": {
            "description": "Phase B3 automated proposals for CVAT review; not ground truth",
            "version": PROPOSAL_VERSION,
            "proposal_batch_id": proposal_batch_id,
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "traffic_sign_candidate", "supercategory": "sign"}],
        "images": images,
        "annotations": annotations,
    }
    coco_path.write_text(json.dumps(coco, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with intake_path.open("w", newline="", encoding="utf-8") as handle:
        writer: Any = csv.DictWriter(handle, fieldnames=list(PHASE_B_FIELDS))
        writer.writeheader()
        writer.writerows(intake_rows)
    return coco_path, intake_path


def propose_normalized_batch(
    normalization_manifest: str | Path,
    *,
    proposal_batch_id: str,
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    detector_path: str | Path = DEFAULT_DETECTOR,
    classifier_path: str | Path = DEFAULT_CLASSIFIER,
    labels_path: str | Path = DEFAULT_LABELS,
    calibration_path: str | Path = DEFAULT_CALIBRATION,
    device: str = "auto",
    max_objects: int = 10_000,
    fusion_iou: float = 0.45,
    dry_run: bool = False,
    resume: bool = False,
    detectors: Sequence[DetectorProtocol] | None = None,
    classifier: ClassifierProtocol | None = None,
) -> dict[str, object]:
    """Run fused detector/classifier proposals over every retained normalized image."""
    if not 0 < max_objects <= 10_000:
        raise ValueError("max_objects must be within 1..10000")
    if not 0.0 < fusion_iou <= 1.0:
        raise ValueError("fusion_iou must be within (0, 1]")
    root = project_path(acquisition_root)
    manifest_path = project_path(normalization_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    normalization_report: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _mapping(normalization_report)
    outputs = _mapping(manifest.get("outputs"))
    progress_source = project_path(str(outputs.get("progress_jsonl", "")))
    normalized = [
        NormalizedCandidate.model_validate(row)
        for row in _load_jsonl(progress_source)
        if row.get("triage_state") == "normalized_candidate"
    ][:max_objects]
    detector_file = project_path(detector_path)
    classifier_file = project_path(classifier_path)
    label_file = project_path(labels_path)
    calibration_file = project_path(calibration_path)
    report_path = root / "manifests" / "proposals" / f"{proposal_batch_id}.json"
    parquet_path = root / "manifests" / "proposals" / f"{proposal_batch_id}.parquet"
    progress_path = root / "manifests" / "proposals" / f"{proposal_batch_id}.jsonl"
    audit_path = project_path(audit_root) / "proposals" / f"{proposal_batch_id}.json"
    package = root / "review_packages" / proposal_batch_id
    if report_path.exists():
        if resume:
            value: object = json.loads(report_path.read_text(encoding="utf-8"))
            return dict(_mapping(value)) | {"resume_reused": True}
        raise FileExistsError(f"Refusing to overwrite proposal report: {report_path}")
    preflight: dict[str, object] = {
        "proposal_batch_id": proposal_batch_id,
        "normalization_manifest": str(manifest_path),
        "candidate_images": len(normalized),
        "detector_path": str(detector_file),
        "classifier_path": str(classifier_file),
        "labels_path": str(label_file),
        "calibration_path": str(calibration_file),
        "model_files_available": all(
            path.is_file() for path in (detector_file, classifier_file, label_file, calibration_file)
        ),
        "dry_run": dry_run,
    }
    if dry_run:
        return preflight
    if detectors is None or classifier is None:
        if not preflight["model_files_available"]:
            raise FileNotFoundError("One or more Phase B3 model artifacts are missing")
        default_detectors, default_classifier = _default_models(
            detector_path=detector_file,
            classifier_path=classifier_file,
            labels_path=label_file,
            calibration_path=calibration_file,
            device=device,
        )
        active_detectors = list(detectors or default_detectors)
        active_classifier = classifier or default_classifier
    else:
        active_detectors = list(detectors)
        active_classifier = classifier
    existing = {
        str(row.get("object_id", "")): ImageProposalResult.model_validate(row)
        for row in _load_jsonl(progress_path)
        if row.get("proposal_batch_id") == proposal_batch_id
    }
    results = dict(existing)
    for candidate in sorted(normalized, key=lambda row: row.object_id):
        if candidate.object_id in results:
            continue
        image_path = project_path(candidate.anonymized_path or candidate.normalized_path)
        image_sha256 = _hash(image_path) if image_path.is_file() else candidate.normalized_sha256
        image_raw: Any = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_raw is None:
            result = ImageProposalResult(
                proposal_batch_id=proposal_batch_id,
                processed_at=_now(),
                object_id=candidate.object_id,
                source_id=candidate.source_id,
                group_id=candidate.group_id,
                image_path=str(image_path),
                image_sha256=image_sha256,
                image_state="inference_failure",
                inference_error="normalized_image_decode_failed",
                licence_status=candidate.licence_status,
            )
        else:
            image = cast(np.ndarray[Any, np.dtype[np.uint8]], image_raw)
            height, width = image.shape[:2]
            try:
                detector_results = [
                    (f"config-{index}:{detector.name}", detector.detect(image))
                    for index, detector in enumerate(active_detectors, start=1)
                ]
                fused = _fuse_detections(detector_results, iou_threshold=fusion_iou)
                proposals: list[Proposal] = []
                for index, (detection, members) in enumerate(fused):
                    classification = active_classifier.classify(_crop(image, detection))
                    area_ratio = min(1.0, detection.bbox.area / (width * height))
                    size_bucket = _size_bucket(area_ratio)
                    flags = _semantic_flags(classification, size_bucket)
                    state = _candidate_state(classification, flags)
                    proposal_id = _stable_id(
                        proposal_batch_id,
                        candidate.object_id,
                        index,
                        *(round(value, 2) for value in (
                            detection.bbox.x1,
                            detection.bbox.y1,
                            detection.bbox.x2,
                            detection.bbox.y2,
                        )),
                    )
                    proposals.append(
                        Proposal(
                            proposal_id=proposal_id,
                            object_id=candidate.object_id,
                            source_id=candidate.source_id,
                            group_id=candidate.group_id,
                            image_path=str(image_path),
                            image_sha256=image_sha256,
                            image_width=width,
                            image_height=height,
                            bbox_xyxy=[
                                round(detection.bbox.x1, 3),
                                round(detection.bbox.y1, 3),
                                round(detection.bbox.x2, 3),
                                round(detection.bbox.y2, 3),
                            ],
                            area_ratio=round(area_ratio, 8),
                            size_bucket=size_bucket,
                            detector_confidence=detection.confidence,
                            detector_members=members,
                            semantic_sign_id=classification.semantic_sign_id,
                            classifier_confidence=classification.confidence,
                            classifier_accepted=classification.accepted,
                            classifier_top_k=[
                                {"label": label, "score": round(score, 8)}
                                for label, score in classification.top_k
                            ],
                            classifier_unknown_score=classification.unknown_score,
                            embedding_distance=classification.embedding_distance,
                            nearest_prototype=classification.nearest_prototype or "",
                            rejection_reasons=classification.rejection_reasons,
                            semantic_review_flags=flags,
                            candidate_state=state,
                            priority_score=_priority(
                                classification, detection.confidence, size_bucket, state
                            ),
                            licence_status=candidate.licence_status,
                        )
                    )
                result = ImageProposalResult(
                    proposal_batch_id=proposal_batch_id,
                    processed_at=_now(),
                    object_id=candidate.object_id,
                    source_id=candidate.source_id,
                    group_id=candidate.group_id,
                    image_path=str(image_path),
                    image_sha256=image_sha256,
                    image_width=width,
                    image_height=height,
                    proposals=proposals,
                    image_state="proposals_generated" if proposals else "candidate_no_sign",
                    licence_status=candidate.licence_status,
                )
            except (RuntimeError, ValueError, cv2.error) as exc:
                result = ImageProposalResult(
                    proposal_batch_id=proposal_batch_id,
                    processed_at=_now(),
                    object_id=candidate.object_id,
                    source_id=candidate.source_id,
                    group_id=candidate.group_id,
                    image_path=str(image_path),
                    image_sha256=image_sha256,
                    image_width=width,
                    image_height=height,
                    image_state="inference_failure",
                    inference_error=f"{type(exc).__name__}:{exc}",
                    licence_status=candidate.licence_status,
                )
        payload = result.model_dump(mode="json")
        _append_jsonl(progress_path, payload)
        results[result.object_id] = result
    ordered = [results[key] for key in sorted(results)]
    proposals = [proposal for result in ordered for proposal in result.proposals]
    _write_parquet(parquet_path, [proposal.model_dump(mode="json") for proposal in proposals])
    source_urls = _source_urls(root / "manifests" / "object_events.jsonl")
    coco_path, intake_path = _write_exports(
        package, ordered, proposal_batch_id=proposal_batch_id, source_urls=source_urls
    )
    image_states = Counter(result.image_state for result in ordered)
    proposal_states = Counter(proposal.candidate_state for proposal in proposals)
    size_buckets = Counter(proposal.size_bucket for proposal in proposals)
    class_counts = Counter(proposal.semantic_sign_id for proposal in proposals)
    report: dict[str, object] = preflight | {
        "dry_run": False,
        "proposal_version": PROPOSAL_VERSION,
        "detector_sha256": _hash(detector_file) if detector_file.is_file() else "injected-test-double",
        "classifier_sha256": _hash(classifier_file)
        if classifier_file.is_file()
        else "injected-test-double",
        "detector_configurations": [
            f"config-{index}:{detector.name}"
            for index, detector in enumerate(active_detectors, start=1)
        ],
        "classifier_name": active_classifier.name,
        "images_processed": len(ordered),
        "proposal_count": len(proposals),
        "image_states": dict(sorted(image_states.items())),
        "proposal_states": dict(sorted(proposal_states.items())),
        "size_buckets": dict(sorted(size_buckets.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "embedding_distance_available": sum(
            proposal.embedding_distance is not None for proposal in proposals
        ),
        "review_flags": dict(
            sorted(Counter(flag for proposal in proposals for flag in proposal.semantic_review_flags).items())
        ),
        "outputs": {
            "report": str(report_path),
            "audit_report": str(audit_path),
            "proposal_parquet": str(parquet_path),
            "progress_jsonl": str(progress_path),
            "cvat_coco": str(coco_path),
            "canonical_intake_proposals": str(intake_path),
        },
        "exit_condition": {
            "passed": len(ordered) == len(normalized)
            and all(result.image_state for result in ordered),
            "basis": "Every retained normalized image has proposal results, a no-sign proposal state, or an explicit inference failure.",
        },
        "resume_reused": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "PROPOSAL_VERSION",
    "ImageProposalResult",
    "Proposal",
    "propose_normalized_batch",
]
