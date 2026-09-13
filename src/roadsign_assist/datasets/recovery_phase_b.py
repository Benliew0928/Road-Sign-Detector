"""Schema, leakage checks, and collection-floor audits for recovery Phase B."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from roadsign_assist.datasets.recovery_approval import has_training_approval, load_training_approval
from roadsign_assist.paths import project_path

SCHEMA_VERSION = "1.0"
SPLITS = frozenset({"train", "validation", "internal_test", "phase_e2"})
EXPECTED_KINDS = frozenset({"sign", "no_sign", "unknown_sign", "unreadable"})
CAPTURE_DOMAINS = frozenset({"road_scene", "electronic_screen", "printed_card"})
SIZE_BUCKETS = frozenset({"very_small", "small", "medium", "large"})
GROUP_FIELDS = (
    "image_sha256",
    "source_artwork_id",
    "base_rendering_id",
    "session_id",
    "route_id",
    "video_id",
    "physical_sign_id",
    "related_capture_group_id",
)
PHASE_B_FIELDS = (
    "schema_version",
    "release_id",
    "sample_id",
    "instance_id",
    "split",
    "expected_kind",
    "semantic_sign_id",
    "capture_domain",
    "image_path",
    "label_path",
    "image_sha256",
    "image_width",
    "image_height",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "area_ratio",
    "size_bucket",
    "source_id",
    "source_artwork_id",
    "base_rendering_id",
    "session_id",
    "route_id",
    "video_id",
    "physical_sign_id",
    "related_capture_group_id",
    "camera_id",
    "camera_pipeline_id",
    "device_id",
    "weather",
    "timeofday",
    "lighting",
    "road_type",
    "orientation_degrees",
    "screen_orientation",
    "presentation_material",
    "presentation_distance",
    "presentation_brightness",
    "presentation_glare",
    "presentation_background",
    "scene_composition",
    "negative_kind",
    "annotation_status",
    "review_decision",
    "primary_reviewer",
    "secondary_reviewer",
    "review_notes",
    "licence_status",
    "use_policy",
    "source_url",
)
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
NUMERIC_PARAMETER_TYPES = frozenset(
    {"speed", "height", "width", "weight", "axle_weight", "time", "distance"}
)


@dataclass(frozen=True)
class PhaseBFloors:
    full_context_images: int = 15_000
    annotated_sign_instances: int = 20_000
    negative_fraction_min: float = 0.20
    negative_fraction_max: float = 0.30
    independent_sessions: int = 30
    camera_pipelines: int = 3
    small_instances: int = 3_000
    very_small_instances: int = 1_000
    electronic_devices_per_class: int = 3
    independent_eval_groups_per_safety_class: int = 30


DEFAULT_FLOORS = PhaseBFloors()


def _is_real_capture(row: Mapping[str, str]) -> bool:
    """Synthetic augmentations and unverified artwork do not meet camera floors."""
    return (
        row.get("capture_evidence") not in {"synthetic", "unverified_context"}
        and "synthetic" not in row.get("source_id", "").lower()
        and "synthetic" not in row.get("camera_pipeline_id", "").lower()
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normal(value: object) -> str:
    return str(value or "").strip()


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _size_bucket(area_ratio: float) -> str:
    if area_ratio <= 0.001:
        return "very_small"
    if area_ratio <= 0.01:
        return "small"
    if area_ratio <= 0.05:
        return "medium"
    return "large"


def _catalogue_policy(
    catalogue_path: Path,
    supported_labels_path: Path | None,
) -> tuple[set[str], set[str]]:
    payload = json.loads(catalogue_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    by_id = {str(entry["semantic_sign_id"]): entry for entry in entries}
    if supported_labels_path is None:
        known = set(by_id)
    else:
        raw_labels: object = json.loads(supported_labels_path.read_text(encoding="utf-8"))
        if not isinstance(raw_labels, list):
            raise ValueError("Supported classifier labels must be a non-empty JSON string list")
        label_values = cast(list[object], raw_labels)
        if not label_values or not all(isinstance(label, str) and label for label in label_values):
            raise ValueError("Supported classifier labels must be a non-empty JSON string list")
        labels = cast(list[str], label_values)
        known = set(labels)
        if len(known) != len(labels):
            raise ValueError("Supported classifier labels must be unique")
        missing = sorted(known - set(by_id))
        if missing:
            raise ValueError(f"Supported labels missing from catalogue: {missing}")
    double_review = {
        semantic
        for semantic, entry in by_id.items()
        if semantic in known
        if entry.get("severity") == "critical"
        or entry.get("parameter_type") == "direction"
        or entry.get("parameter_type") in NUMERIC_PARAMETER_TYPES
    }
    return known, double_review


def write_phase_b_template(path: str | Path, *, release_id: str) -> Path:
    """Write an empty, versioned intake manifest without implying collected data."""
    destination = Path(path)
    if not destination.is_absolute():
        destination = project_path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite manifest template: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PHASE_B_FIELDS))
        writer.writeheader()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "manifest": destination.name,
        "rows": 0,
        "status": "collection_not_started",
    }
    _write_json(destination.with_suffix(".template.json"), metadata)
    return destination


def _split_leakage(rows: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, object]]]:
    findings: dict[str, list[dict[str, object]]] = {}
    for field in GROUP_FIELDS:
        assignments: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = _normal(row.get(field))
            split = _normal(row.get("split"))
            if value and split:
                assignments[value].add(split)
        collisions: list[dict[str, object]] = [
            {"value": value, "splits": sorted(splits)}
            for value, splits in sorted(assignments.items())
            if len(splits) > 1
        ]
        if collisions:
            findings[field] = collisions
    return findings


def _sample_consistency(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    by_sample: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_sample[_normal(row.get("sample_id"))].append(row)
    fields = (
        "split",
        "capture_domain",
        "image_path",
        "image_sha256",
        "image_width",
        "image_height",
        "session_id",
        "route_id",
        "video_id",
        "related_capture_group_id",
        "camera_pipeline_id",
    )
    findings: list[dict[str, object]] = []
    for sample_id, sample_rows in sorted(by_sample.items()):
        if not sample_id:
            continue
        conflicts = {
            field: sorted({_normal(row.get(field)) for row in sample_rows})
            for field in fields
            if len({_normal(row.get(field)) for row in sample_rows}) > 1
        }
        if conflicts:
            findings.append({"sample_id": sample_id, "conflicts": conflicts})
    return findings


def _split_balance(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    development = [row for row in rows if _normal(row.get("split")) != "phase_e2"]
    groups: dict[str, str] = {}
    for row in development:
        group = _normal(row.get("related_capture_group_id"))
        split = _normal(row.get("split"))
        if group and split:
            groups[group] = split
    counts = Counter(groups.values())
    total = sum(counts.values())
    proportions = {
        split: (counts[split] / total if total else 0.0)
        for split in ("train", "validation", "internal_test")
    }
    passed = bool(total) and (
        0.65 <= proportions["train"] <= 0.75
        and 0.10 <= proportions["validation"] <= 0.20
        and 0.10 <= proportions["internal_test"] <= 0.20
    )
    return {"groups": dict(counts), "proportions": proportions, "passed": passed}


def audit_phase_b_manifest(
    manifest_path: str | Path,
    *,
    catalogue_path: str | Path = "configs/catalogue/malaysia_signs.v1.json",
    supported_labels_path: str | Path | None = (
        "models/exported/runtime/sign_classifier.labels.json"
    ),
    floors: PhaseBFloors = DEFAULT_FLOORS,
    training_approval_policy: str | Path | None = None,
) -> dict[str, Any]:
    """Audit a Phase B manifest without mutating data or split assignments."""
    manifest = project_path(manifest_path)
    approval_policy = load_training_approval(training_approval_policy)
    catalogue = project_path(catalogue_path)
    labels = project_path(supported_labels_path) if supported_labels_path is not None else None
    fields, rows = _read_csv(manifest)
    known_classes, double_review_classes = _catalogue_policy(catalogue, labels)
    missing_columns = sorted(set(PHASE_B_FIELDS) - set(fields))
    errors: list[dict[str, object]] = []
    duplicate_instances: set[str] = set()
    seen_instances: set[str] = set()
    sign_rows: list[dict[str, str]] = []

    common_required = (
        "schema_version",
        "release_id",
        "sample_id",
        "split",
        "expected_kind",
        "capture_domain",
        "image_path",
        "image_sha256",
        "image_width",
        "image_height",
        "source_id",
        "session_id",
        "related_capture_group_id",
        "camera_pipeline_id",
        "annotation_status",
        "review_decision",
        "primary_reviewer",
        "licence_status",
        "use_policy",
    )
    for row_number, row in enumerate(rows, start=2):
        for field in common_required:
            if not _normal(row.get(field)):
                errors.append({"row": row_number, "code": "missing_value", "field": field})
        schema_version = _normal(row.get("schema_version"))
        split = _normal(row.get("split"))
        kind = _normal(row.get("expected_kind"))
        domain = _normal(row.get("capture_domain"))
        if schema_version != SCHEMA_VERSION:
            errors.append({"row": row_number, "code": "schema_version", "value": schema_version})
        if split not in SPLITS:
            errors.append({"row": row_number, "code": "split", "value": split})
        if kind not in EXPECTED_KINDS:
            errors.append({"row": row_number, "code": "expected_kind", "value": kind})
        if domain not in CAPTURE_DOMAINS:
            errors.append({"row": row_number, "code": "capture_domain", "value": domain})
        digest = _normal(row.get("image_sha256"))
        if not SHA256_PATTERN.fullmatch(digest):
            errors.append({"row": row_number, "code": "image_sha256", "value": digest})
        width = _int(_normal(row.get("image_width")))
        height = _int(_normal(row.get("image_height")))
        if width is None or width <= 0 or height is None or height <= 0:
            errors.append({"row": row_number, "code": "image_dimensions"})
        if _normal(row.get("annotation_status")) != "approved":
            errors.append({"row": row_number, "code": "annotation_not_approved"})
        if _normal(row.get("review_decision")) != "accept":
            errors.append({"row": row_number, "code": "review_not_accepted"})

        if domain == "road_scene":
            for field in ("route_id", "camera_id", "weather", "timeofday", "lighting", "road_type"):
                if not _normal(row.get(field)):
                    errors.append(
                        {"row": row_number, "code": "missing_road_metadata", "field": field}
                    )
        elif domain == "electronic_screen":
            for field in (
                "source_artwork_id",
                "base_rendering_id",
                "device_id",
                "orientation_degrees",
                "screen_orientation",
                "presentation_distance",
                "presentation_brightness",
                "presentation_glare",
                "presentation_background",
                "scene_composition",
            ):
                if not _normal(row.get(field)):
                    errors.append(
                        {"row": row_number, "code": "missing_screen_metadata", "field": field}
                    )
        elif domain == "printed_card":
            for field in (
                "source_artwork_id",
                "base_rendering_id",
                "camera_id",
                "orientation_degrees",
                "presentation_material",
                "presentation_distance",
                "presentation_glare",
                "presentation_background",
                "scene_composition",
            ):
                if not _normal(row.get(field)):
                    errors.append(
                        {"row": row_number, "code": "missing_print_metadata", "field": field}
                    )

        if kind == "sign":
            sign_rows.append(row)
            instance_id = _normal(row.get("instance_id"))
            semantic = _normal(row.get("semantic_sign_id"))
            if not instance_id:
                errors.append({"row": row_number, "code": "missing_instance_id"})
            elif instance_id in seen_instances:
                duplicate_instances.add(instance_id)
            seen_instances.add(instance_id)
            if semantic not in known_classes:
                errors.append({"row": row_number, "code": "unknown_semantic_id", "value": semantic})
            coordinates = [
                _float(_normal(row.get(field)))
                for field in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
            ]
            if any(value is None for value in coordinates) or width is None or height is None:
                errors.append({"row": row_number, "code": "invalid_bbox"})
            else:
                x1, y1, x2, y2 = (float(value) for value in coordinates if value is not None)
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    errors.append({"row": row_number, "code": "invalid_bbox"})
                else:
                    expected_area = ((x2 - x1) * (y2 - y1)) / (width * height)
                    area = _float(_normal(row.get("area_ratio")))
                    bucket = _normal(row.get("size_bucket"))
                    if area is None or abs(area - expected_area) > 1e-6:
                        errors.append({"row": row_number, "code": "area_ratio"})
                    if bucket not in SIZE_BUCKETS or bucket != _size_bucket(expected_area):
                        errors.append({"row": row_number, "code": "size_bucket"})
            if domain == "road_scene" and not _normal(row.get("physical_sign_id")):
                errors.append({"row": row_number, "code": "missing_physical_sign_id"})
            if semantic in double_review_classes:
                primary = _normal(row.get("primary_reviewer"))
                secondary = _normal(row.get("secondary_reviewer"))
                if (not secondary or secondary == primary) and not has_training_approval(
                    row, approval_policy
                ):
                    errors.append({"row": row_number, "code": "double_review_required"})
        elif kind == "no_sign":
            if not _normal(row.get("negative_kind")):
                errors.append({"row": row_number, "code": "missing_negative_kind"})
            if any(
                _normal(row.get(field)) for field in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
            ):
                errors.append({"row": row_number, "code": "negative_has_bbox"})

    leakage = _split_leakage(rows)
    sample_conflicts = _sample_consistency(rows)
    split_balance = _split_balance(rows)
    first_by_sample: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = _normal(row.get("sample_id"))
        if sample_id and sample_id not in first_by_sample:
            first_by_sample[sample_id] = row
    images = list(first_by_sample.values())
    accepted_negatives = [
        row
        for row in images
        if _normal(row.get("expected_kind")) == "no_sign"
        and _normal(row.get("review_decision")) == "accept"
    ]
    road_development_images = [
        row
        for row in images
        if _normal(row.get("split")) != "phase_e2"
        and _normal(row.get("capture_domain")) == "road_scene"
        and _is_real_capture(row)
    ]
    road_development_negatives = [
        row
        for row in accepted_negatives
        if _normal(row.get("split")) != "phase_e2"
        and _normal(row.get("capture_domain")) == "road_scene"
        and _is_real_capture(row)
    ]
    negative_fraction = (
        len(road_development_negatives) / len(road_development_images)
        if road_development_images
        else 0.0
    )
    road_development_sign_rows = [
        row
        for row in sign_rows
        if _normal(row.get("split")) != "phase_e2"
        and _normal(row.get("capture_domain")) == "road_scene"
        and _is_real_capture(row)
    ]
    sign_areas = [
        area
        for row in road_development_sign_rows
        if (area := _float(_normal(row.get("area_ratio")))) is not None
    ]
    small_count = sum(area <= 0.01 for area in sign_areas)
    very_small_count = sum(area <= 0.001 for area in sign_areas)
    sessions = {
        _normal(row.get("session_id"))
        for row in road_development_images
        if _normal(row.get("session_id")) and "unknown" not in row["session_id"]
    }
    camera_pipelines = {
        _normal(row.get("camera_pipeline_id"))
        for row in road_development_images
        if _normal(row.get("camera_pipeline_id")) and "unknown" not in row["camera_pipeline_id"]
    }
    domains = Counter(_normal(row.get("capture_domain")) for row in images)
    splits = Counter(_normal(row.get("split")) for row in images)

    screen_rows = [
        row
        for row in sign_rows
        if _normal(row.get("capture_domain")) == "electronic_screen"
        and _normal(row.get("split")) != "phase_e2"
        and _is_real_capture(row)
    ]
    print_rows = [
        row
        for row in sign_rows
        if _normal(row.get("capture_domain")) == "printed_card"
        and _normal(row.get("split")) != "phase_e2"
        and _is_real_capture(row)
    ]
    screen_devices: dict[str, set[str]] = defaultdict(set)
    screen_orientations: dict[str, set[str]] = defaultdict(set)
    screen_rotations: dict[str, list[float]] = defaultdict(list)
    screen_distances: dict[str, set[str]] = defaultdict(set)
    screen_brightness: dict[str, set[str]] = defaultdict(set)
    screen_glare: dict[str, set[str]] = defaultdict(set)
    screen_backgrounds: dict[str, set[str]] = defaultdict(set)
    screen_compositions: dict[str, set[str]] = defaultdict(set)
    print_materials: dict[str, set[str]] = defaultdict(set)
    print_distances: dict[str, set[str]] = defaultdict(set)
    for row in screen_rows:
        semantic = _normal(row.get("semantic_sign_id"))
        screen_devices[semantic].add(_normal(row.get("device_id")))
        screen_orientations[semantic].add(_normal(row.get("screen_orientation")))
        screen_distances[semantic].add(_normal(row.get("presentation_distance")))
        screen_brightness[semantic].add(_normal(row.get("presentation_brightness")))
        screen_glare[semantic].add(_normal(row.get("presentation_glare")))
        screen_backgrounds[semantic].add(_normal(row.get("presentation_background")))
        screen_compositions[semantic].add(_normal(row.get("scene_composition")))
        rotation = _float(_normal(row.get("orientation_degrees")))
        if rotation is not None:
            screen_rotations[semantic].append(rotation)
    for row in print_rows:
        semantic = _normal(row.get("semantic_sign_id"))
        print_materials[semantic].add(_normal(row.get("presentation_material")))
        print_distances[semantic].add(_normal(row.get("presentation_distance")))
    screen_class_gaps = sorted(known_classes - set(screen_devices))
    print_class_gaps = sorted(known_classes - set(print_materials))
    screen_device_gaps = sorted(
        semantic
        for semantic in known_classes
        if len(screen_devices.get(semantic, set()) - {""}) < floors.electronic_devices_per_class
    )
    screen_orientation_gaps = sorted(
        semantic
        for semantic in known_classes
        if not {"portrait", "landscape"}.issubset(screen_orientations.get(semantic, set()))
    )
    screen_rotation_gaps = sorted(
        semantic
        for semantic in known_classes
        if not screen_rotations.get(semantic)
        or min(screen_rotations[semantic]) > -40
        or max(screen_rotations[semantic]) < 40
    )
    screen_distance_gaps = sorted(
        semantic
        for semantic in known_classes
        if not {"near", "medium", "far"}.issubset(screen_distances.get(semantic, set()))
    )
    screen_brightness_gaps = sorted(
        semantic
        for semantic in known_classes
        if not {"low", "medium", "high"}.issubset(screen_brightness.get(semantic, set()))
    )
    screen_glare_gaps = sorted(
        semantic
        for semantic in known_classes
        if not {"none", "glare"}.issubset(screen_glare.get(semantic, set()))
    )
    screen_background_gaps = sorted(
        semantic
        for semantic in known_classes
        if len(screen_backgrounds.get(semantic, set()) - {""}) < 2
    )
    screen_composition_gaps = sorted(
        semantic
        for semantic in known_classes
        if not {"single_sign", "search_results", "multi_sign"}.issubset(
            screen_compositions.get(semantic, set())
        )
    )
    print_material_gaps = sorted(
        semantic
        for semantic in known_classes
        if not {"matte", "glossy"}.issubset(print_materials.get(semantic, set()))
    )
    print_distance_gaps = sorted(
        semantic
        for semantic in known_classes
        if not {"near", "medium", "far"}.issubset(print_distances.get(semantic, set()))
    )
    screen_negative_kinds = {
        _normal(row.get("negative_kind"))
        for row in images
        if _normal(row.get("capture_domain")) == "electronic_screen"
        and _normal(row.get("expected_kind")) == "no_sign"
        and _normal(row.get("split")) != "phase_e2"
        and _is_real_capture(row)
    }
    screen_negative_kind_gaps = sorted(
        {"icons_or_logos", "yellow_diamond", "unrelated_traffic_imagery"} - screen_negative_kinds
    )
    road_rows = road_development_images
    observed_weather = {_normal(row.get("weather")) for row in road_rows}
    observed_time = {_normal(row.get("timeofday")) for row in road_rows}
    observed_lighting = {_normal(row.get("lighting")) for row in road_rows}
    condition_gaps = {
        "weather": sorted({"clear", "rain", "haze"} - observed_weather),
        "timeofday": sorted({"day", "dusk", "night"} - observed_time),
        "lighting": sorted({"normal", "glare", "shadow", "backlight"} - observed_lighting),
    }
    evaluation_rows = [
        row for row in sign_rows if _normal(row.get("split")) in {"validation", "internal_test"}
    ]
    evaluation_groups: dict[str, set[str]] = defaultdict(set)
    for row in evaluation_rows:
        semantic = _normal(row.get("semantic_sign_id"))
        if semantic in double_review_classes:
            group = _normal(row.get("related_capture_group_id"))
            if group:
                evaluation_groups[semantic].add(group)
    safety_group_gaps = {
        semantic: len(evaluation_groups.get(semantic, set()))
        for semantic in sorted(double_review_classes)
        if len(evaluation_groups.get(semantic, set()))
        < floors.independent_eval_groups_per_safety_class
    }

    checks = {
        "required_columns": not missing_columns,
        "rows_present": bool(rows),
        "row_validation": not errors,
        "unique_instance_ids": not duplicate_instances,
        "consistent_image_metadata": not sample_conflicts,
        "zero_cross_split_leakage": not leakage,
        "group_split_balance": bool(split_balance["passed"]),
        "phase_e2_present": splits["phase_e2"] > 0,
        "full_context_image_floor": (len(road_development_images) >= floors.full_context_images),
        "annotated_instance_floor": (
            len(road_development_sign_rows) >= floors.annotated_sign_instances
        ),
        "negative_fraction_floor": (
            floors.negative_fraction_min <= negative_fraction <= floors.negative_fraction_max
        ),
        "independent_session_floor": len(sessions) >= floors.independent_sessions,
        "camera_pipeline_floor": len(camera_pipelines) >= floors.camera_pipelines,
        "small_instance_floor": small_count >= floors.small_instances,
        "very_small_instance_floor": very_small_count >= floors.very_small_instances,
        "required_road_conditions": not any(condition_gaps.values()),
        "screen_all_classes": not screen_class_gaps,
        "screen_device_floor": not screen_device_gaps,
        "screen_orientation_coverage": not screen_orientation_gaps,
        "screen_rotation_coverage": not screen_rotation_gaps,
        "screen_distance_coverage": not screen_distance_gaps,
        "screen_brightness_coverage": not screen_brightness_gaps,
        "screen_glare_coverage": not screen_glare_gaps,
        "screen_background_coverage": not screen_background_gaps,
        "screen_composition_coverage": not screen_composition_gaps,
        "screen_negative_coverage": not screen_negative_kind_gaps,
        "print_all_classes": not print_class_gaps,
        "print_material_coverage": not print_material_gaps,
        "print_distance_coverage": not print_distance_gaps,
        "safety_eval_group_floor": not safety_group_gaps,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "B",
        "training_approval_policy": approval_policy,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "audited_at": datetime.now(UTC).isoformat(),
        "ontology": {
            "catalogue_path": str(catalogue),
            "catalogue_sha256": _sha256(catalogue),
            "supported_labels_path": str(labels) if labels is not None else None,
            "supported_labels_sha256": _sha256(labels) if labels is not None else None,
            "supported_class_count": len(known_classes),
        },
        "floors": asdict(floors),
        "counts": {
            "rows": len(rows),
            "images": len(images),
            "sign_instances": len(sign_rows),
            "accepted_negative_images": len(accepted_negatives),
            "road_development_images": len(road_development_images),
            "road_development_sign_instances": len(road_development_sign_rows),
            "road_development_accepted_negative_images": len(road_development_negatives),
            "road_development_negative_fraction": negative_fraction,
            "road_development_independent_sessions": len(sessions),
            "road_development_camera_pipelines": len(camera_pipelines),
            "road_development_small_instances_at_most_1pct": small_count,
            "road_development_very_small_instances_at_most_0_1pct": very_small_count,
            "domains": dict(domains),
            "splits": dict(splits),
        },
        "missing_columns": missing_columns,
        "validation_errors": errors,
        "duplicate_instance_ids": sorted(duplicate_instances),
        "image_metadata_conflicts": sample_conflicts,
        "cross_split_leakage": leakage,
        "split_balance": split_balance,
        "coverage_gaps": {
            "road_conditions": condition_gaps,
            "screen_classes": screen_class_gaps,
            "screen_device_classes": screen_device_gaps,
            "screen_orientation_classes": screen_orientation_gaps,
            "screen_rotation_classes": screen_rotation_gaps,
            "screen_distance_classes": screen_distance_gaps,
            "screen_brightness_classes": screen_brightness_gaps,
            "screen_glare_classes": screen_glare_gaps,
            "screen_background_classes": screen_background_gaps,
            "screen_composition_classes": screen_composition_gaps,
            "screen_negative_kinds": screen_negative_kind_gaps,
            "print_classes": print_class_gaps,
            "print_material_classes": print_material_gaps,
            "print_distance_classes": print_distance_gaps,
            "safety_evaluation_groups": safety_group_gaps,
        },
        "checks": checks,
        "gate_b_passed": all(checks.values()),
    }


def audit_existing_recovery_inputs(
    detector_manifest: str | Path,
    classifier_manifest: str | Path,
    *,
    catalogue_path: str | Path = "configs/catalogue/malaysia_signs.v1.json",
    supported_labels_path: str | Path | None = (
        "models/exported/runtime/sign_classifier.labels.json"
    ),
    floors: PhaseBFloors = DEFAULT_FLOORS,
) -> dict[str, Any]:
    """Measure current releases against Phase B without treating them as new data."""
    detector_path = project_path(detector_manifest)
    classifier_path = project_path(classifier_manifest)
    _, detector_rows = _read_csv(detector_path)
    _, classifier_rows = _read_csv(classifier_path)
    catalogue = project_path(catalogue_path)
    labels = project_path(supported_labels_path) if supported_labels_path is not None else None
    known_classes, double_review_classes = _catalogue_policy(catalogue, labels)
    detector_splits = Counter(_normal(row.get("split")) for row in detector_rows)
    negative_splits = Counter(
        _normal(row.get("split"))
        for row in detector_rows
        if _normal(row.get("is_negative")).lower() == "true"
    )
    sign_instances = sum(_int(_normal(row.get("bbox_count"))) or 0 for row in detector_rows)
    areas: list[float] = []
    missing_labels = 0
    for row in detector_rows:
        label_value = _normal(row.get("dataset_label_path"))
        if not label_value:
            continue
        label_path = project_path(label_value)
        if not label_path.is_file():
            missing_labels += 1
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                width = _float(parts[3])
                height = _float(parts[4])
                if width is not None and height is not None:
                    areas.append(width * height)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in detector_rows:
        group = _normal(row.get("group_id"))
        split = _normal(row.get("split"))
        if group and split:
            group_splits[group].add(split)
    detector_leaks = {
        group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1
    }
    classifier_splits = Counter(_normal(row.get("split")) for row in classifier_rows)
    train_counts = Counter(
        _normal(row.get("semantic_sign_id"))
        for row in classifier_rows
        if _normal(row.get("split")) == "train"
    )
    eval_groups: dict[str, set[str]] = defaultdict(set)
    for row in classifier_rows:
        if _normal(row.get("split")) not in {"validation", "test"}:
            continue
        semantic = _normal(row.get("semantic_sign_id"))
        group = _normal(row.get("source_group")) or _normal(row.get("source_video_group"))
        if semantic and group:
            eval_groups[semantic].add(group)
    train_values = list(train_counts.values())
    metadata_absent = [
        "capture_domain",
        "session_id",
        "route_id",
        "physical_sign_id",
        "camera_id",
        "camera_pipeline_id",
        "weather",
        "timeofday",
        "lighting",
        "road_type",
        "source_artwork_id",
        "base_rendering_id",
        "device_id",
    ]
    negative_count = sum(negative_splits.values())
    negative_fraction = negative_count / len(detector_rows) if detector_rows else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "B",
        "status": "baseline_gap_measurement",
        "audited_at": datetime.now(UTC).isoformat(),
        "gate_b_passed": False,
        "floors": asdict(floors),
        "ontology": {
            "catalogue_path": str(catalogue),
            "catalogue_sha256": _sha256(catalogue),
            "supported_labels_path": str(labels) if labels is not None else None,
            "supported_labels_sha256": _sha256(labels) if labels is not None else None,
            "supported_class_count": len(known_classes),
        },
        "detector": {
            "manifest": str(detector_path),
            "manifest_sha256": _sha256(detector_path),
            "images": len(detector_rows),
            "annotated_sign_instances": sign_instances,
            "accepted_negative_images": negative_count,
            "negative_fraction": negative_fraction,
            "negative_images_by_split": dict(negative_splits),
            "images_by_split": dict(detector_splits),
            "small_instances_at_most_1pct": sum(area <= 0.01 for area in areas),
            "very_small_instances_at_most_0_1pct": sum(area <= 0.001 for area in areas),
            "box_area_counts_receive_gate_credit": False,
            "box_area_count_limitation": (
                "Nominal YOLO box areas are measurable, but the release lacks the target-domain, "
                "full-context, route/session, and camera metadata required for Phase B credit."
            ),
            "label_files_missing": missing_labels,
            "group_leakage_count": len(detector_leaks),
            "group_leakage": detector_leaks,
        },
        "classifier": {
            "manifest": str(classifier_path),
            "manifest_sha256": _sha256(classifier_path),
            "rows": len(classifier_rows),
            "rows_by_split": dict(classifier_splits),
            "classes": len({key for key in train_counts if key}),
            "supported_classes": len(known_classes),
            "minimum_train_examples_per_class": min(train_values) if train_values else 0,
            "median_train_examples_per_class": statistics.median(train_values)
            if train_values
            else 0,
            "classes_below_100_train_examples": sorted(
                semantic for semantic in known_classes if train_counts[semantic] < 100
            ),
            "safety_classes_below_30_eval_groups": {
                semantic: len(eval_groups.get(semantic, set()))
                for semantic in sorted(double_review_classes)
                if len(eval_groups.get(semantic, set())) < 30
            },
        },
        "unavailable_required_metadata": metadata_absent,
        "blocking_gaps": [
            "new Phase B release manifest has not been populated",
            "full-context detector image and instance floors are not met",
            "accepted negatives are absent from detector training and validation",
            "required route/session/camera/environment metadata is absent",
            "electronic-screen and printed-card coverage is not measurable",
            "independent Phase E2 collection is absent",
            "classifier field/presentation class floors are not met",
        ],
    }


def write_phase_b_audit(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = project_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "audit.json"
    markdown_path = destination / "report.md"
    _write_json(json_path, report)
    raw_counts = report.get("counts", {})
    raw_checks = report.get("checks", {})
    lines = [
        "# Recovery Phase B Data Audit",
        "",
        f"- Status: `{'PASS' if report.get('gate_b_passed') else 'BLOCKED'}`",
        f"- Audited at: `{report.get('audited_at', '')}`",
        f"- Manifest: `{report.get('manifest', '')}`",
        "",
    ]
    if isinstance(raw_counts, Mapping) and raw_counts:
        counts = cast(Mapping[str, object], raw_counts)
        lines.extend(["## Counts", ""])
        for key, value in counts.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    if isinstance(raw_checks, Mapping) and raw_checks:
        checks = cast(Mapping[str, object], raw_checks)
        lines.extend(["## Gate checks", "", "| Check | Result |", "| --- | --- |"])
        for key, value in checks.items():
            lines.append(f"| `{key}` | {'PASS' if value else 'FAIL'} |")
        lines.append("")
    raw_detector = report.get("detector")
    raw_classifier = report.get("classifier")
    if isinstance(raw_detector, Mapping) and isinstance(raw_classifier, Mapping):
        detector = cast(Mapping[str, object], raw_detector)
        classifier = cast(Mapping[str, object], raw_classifier)
        lines.extend(
            [
                "## Existing release baseline",
                "",
                "| Measure | Value |",
                "| --- | ---: |",
                f"| Detector images | {detector.get('images', 0)} |",
                f"| Detector sign instances | {detector.get('annotated_sign_instances', 0)} |",
                f"| Accepted negative images | {detector.get('accepted_negative_images', 0)} |",
                f"| Negative fraction | {detector.get('negative_fraction', 0)} |",
                f"| Classifier training rows | {cast(Mapping[str, object], classifier.get('rows_by_split', {})).get('train', 0)} |",
                f"| Minimum classifier train examples/class | {classifier.get('minimum_train_examples_per_class', 0)} |",
                f"| Median classifier train examples/class | {classifier.get('median_train_examples_per_class', 0)} |",
                "",
            ]
        )
    raw_blocking = report.get("blocking_gaps", [])
    if (
        isinstance(raw_blocking, Iterable)
        and not isinstance(raw_blocking, (str, bytes))
        and raw_blocking
    ):
        blocking = cast(Iterable[object], raw_blocking)
        lines.extend(["## Blocking gaps", ""])
        lines.extend(f"- {value}" for value in blocking)
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


__all__ = [
    "CAPTURE_DOMAINS",
    "EXPECTED_KINDS",
    "GROUP_FIELDS",
    "PHASE_B_FIELDS",
    "PhaseBFloors",
    "audit_existing_recovery_inputs",
    "audit_phase_b_manifest",
    "write_phase_b_audit",
    "write_phase_b_template",
]
