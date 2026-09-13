"""Generate predeclared Phase B road and presentation collection plans."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from roadsign_assist.paths import project_path

PLAN_VERSION = "1.0"
ROAD_PLAN_FIELDS = (
    "plan_version",
    "plan_status",
    "planned_session_slot",
    "actual_session_id",
    "split",
    "camera_pipeline_slot",
    "actual_camera_pipeline_id",
    "route_slot",
    "actual_route_id",
    "weather_target",
    "timeofday_target",
    "lighting_target",
    "road_type_target",
    "planned_full_context_frames",
    "planned_negative_frames",
    "planned_sign_instances",
    "privacy_consent_status",
    "review_status",
)
PRESENTATION_PLAN_FIELDS = (
    "plan_version",
    "plan_status",
    "task_id",
    "semantic_sign_id",
    "expected_kind",
    "capture_domain",
    "split",
    "device_slot",
    "actual_device_id",
    "source_artwork_slot",
    "actual_source_artwork_id",
    "base_rendering_slot",
    "actual_base_rendering_id",
    "screen_orientation",
    "orientation_degrees",
    "presentation_distance",
    "presentation_brightness",
    "presentation_glare",
    "presentation_background",
    "scene_composition",
    "presentation_material",
    "negative_kind",
    "actual_session_id",
    "privacy_consent_status",
    "review_status",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_labels(path: Path) -> list[str]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Supported labels must be a JSON list")
    values = cast(list[object], raw)
    if not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError("Supported labels must be a non-empty JSON string list")
    labels = cast(list[str], values)
    if len(labels) != len(set(labels)):
        raise ValueError("Supported labels must be unique")
    return labels


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _planned_count(row: dict[str, object], field: str) -> int:
    value = row[field]
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Planned count {field} must be an integer")


def _road_rows() -> list[dict[str, object]]:
    weather = ("clear", "rain", "haze")
    timeofday = ("day", "dusk", "night")
    lighting = ("normal", "glare", "shadow", "backlight")
    road_types = ("urban", "rural", "expressway", "residential", "construction")
    rows: list[dict[str, object]] = []
    for index in range(30):
        split = "train" if index < 21 else "validation" if index < 26 else "internal_test"
        rows.append(
            {
                "plan_version": PLAN_VERSION,
                "plan_status": "planned_no_data",
                "planned_session_slot": f"development-session-{index + 1:02d}",
                "actual_session_id": "",
                "split": split,
                "camera_pipeline_slot": f"camera-pipeline-{index % 3 + 1}",
                "actual_camera_pipeline_id": "",
                "route_slot": f"development-route-{index + 1:02d}",
                "actual_route_id": "",
                "weather_target": weather[index % len(weather)],
                "timeofday_target": timeofday[index % len(timeofday)],
                "lighting_target": lighting[index % len(lighting)],
                "road_type_target": road_types[index % len(road_types)],
                "planned_full_context_frames": 600,
                "planned_negative_frames": 150,
                "planned_sign_instances": 700,
                "privacy_consent_status": "pending",
                "review_status": "pending",
            }
        )
    for index in range(6):
        rows.append(
            {
                "plan_version": PLAN_VERSION,
                "plan_status": "planned_no_data",
                "planned_session_slot": f"phase-e2-session-{index + 1:02d}",
                "actual_session_id": "",
                "split": "phase_e2",
                "camera_pipeline_slot": f"independent-camera-pipeline-{index % 3 + 1}",
                "actual_camera_pipeline_id": "",
                "route_slot": f"phase-e2-route-{index + 1:02d}",
                "actual_route_id": "",
                "weather_target": weather[(index + 1) % len(weather)],
                "timeofday_target": timeofday[(index + 1) % len(timeofday)],
                "lighting_target": lighting[(index + 1) % len(lighting)],
                "road_type_target": road_types[(index + 2) % len(road_types)],
                "planned_full_context_frames": 300,
                "planned_negative_frames": 75,
                "planned_sign_instances": 350,
                "privacy_consent_status": "pending",
                "review_status": "locked_pending_collection",
            }
        )
    return rows


def _presentation_sign_rows(labels: list[str]) -> list[dict[str, object]]:
    development = (
        ("train", "phone", "portrait", -45, "near", "low", "none", "room", "single_sign"),
        (
            "validation",
            "tablet",
            "landscape",
            0,
            "medium",
            "medium",
            "glare",
            "vehicle",
            "search_results",
        ),
        (
            "internal_test",
            "monitor",
            "portrait",
            45,
            "far",
            "high",
            "glare",
            "outdoor",
            "multi_sign",
        ),
    )
    rows: list[dict[str, object]] = []
    for semantic in labels:
        for variant, (
            split,
            device,
            orientation,
            rotation,
            distance,
            brightness,
            glare,
            background,
            composition,
        ) in enumerate(development, start=1):
            rows.append(
                {
                    "plan_version": PLAN_VERSION,
                    "plan_status": "planned_no_data",
                    "task_id": f"screen-{semantic}-{variant}",
                    "semantic_sign_id": semantic,
                    "expected_kind": "sign",
                    "capture_domain": "electronic_screen",
                    "split": split,
                    "device_slot": device,
                    "actual_device_id": "",
                    "source_artwork_slot": f"{semantic}-screen-art-{variant}",
                    "actual_source_artwork_id": "",
                    "base_rendering_slot": f"{semantic}-screen-base-{variant}",
                    "actual_base_rendering_id": "",
                    "screen_orientation": orientation,
                    "orientation_degrees": rotation,
                    "presentation_distance": distance,
                    "presentation_brightness": brightness,
                    "presentation_glare": glare,
                    "presentation_background": background,
                    "scene_composition": composition,
                    "presentation_material": "screen",
                    "negative_kind": "",
                    "actual_session_id": "",
                    "privacy_consent_status": "pending",
                    "review_status": "pending",
                }
            )
        rows.append(
            {
                **rows[-1],
                "task_id": f"screen-{semantic}-phase-e2",
                "split": "phase_e2",
                "device_slot": "independent_device",
                "source_artwork_slot": f"{semantic}-screen-art-phase-e2",
                "base_rendering_slot": f"{semantic}-screen-base-phase-e2",
                "screen_orientation": "landscape",
                "orientation_degrees": 20,
                "presentation_distance": "medium",
                "presentation_brightness": "medium",
                "presentation_glare": "none",
                "presentation_background": "independent",
                "scene_composition": "single_sign",
                "review_status": "locked_pending_collection",
            }
        )
        print_variants = (
            ("train", "matte", "near", -20, "none", "room", "single_sign"),
            ("validation", "glossy", "medium", 0, "glare", "vehicle", "multi_sign"),
            ("internal_test", "matte", "far", 20, "none", "outdoor", "search_results"),
        )
        for variant, (split, material, distance, rotation, glare, background, composition) in enumerate(
            print_variants,
            start=1,
        ):
            rows.append(
                {
                    "plan_version": PLAN_VERSION,
                    "plan_status": "planned_no_data",
                    "task_id": f"print-{semantic}-{variant}",
                    "semantic_sign_id": semantic,
                    "expected_kind": "sign",
                    "capture_domain": "printed_card",
                    "split": split,
                    "device_slot": "capture_camera",
                    "actual_device_id": "",
                    "source_artwork_slot": f"{semantic}-print-art-{variant}",
                    "actual_source_artwork_id": "",
                    "base_rendering_slot": f"{semantic}-print-base-{variant}",
                    "actual_base_rendering_id": "",
                    "screen_orientation": "",
                    "orientation_degrees": rotation,
                    "presentation_distance": distance,
                    "presentation_brightness": "ambient",
                    "presentation_glare": glare,
                    "presentation_background": background,
                    "scene_composition": composition,
                    "presentation_material": material,
                    "negative_kind": "",
                    "actual_session_id": "",
                    "privacy_consent_status": "pending",
                    "review_status": "pending",
                }
            )
        rows.append(
            {
                **rows[-1],
                "task_id": f"print-{semantic}-phase-e2",
                "split": "phase_e2",
                "source_artwork_slot": f"{semantic}-print-art-phase-e2",
                "base_rendering_slot": f"{semantic}-print-base-phase-e2",
                "presentation_material": "glossy",
                "presentation_distance": "medium",
                "orientation_degrees": 10,
                "presentation_glare": "none",
                "presentation_background": "independent",
                "scene_composition": "single_sign",
                "review_status": "locked_pending_collection",
            }
        )
    return rows


def _presentation_negative_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind_index, negative_kind in enumerate(
        ("icons_or_logos", "yellow_diamond", "unrelated_traffic_imagery")
    ):
        for split_index, split in enumerate(
            ("train", "validation", "internal_test", "phase_e2")
        ):
            rows.append(
                {
                    "plan_version": PLAN_VERSION,
                    "plan_status": "planned_no_data",
                    "task_id": f"screen-negative-{negative_kind}-{split}",
                    "semantic_sign_id": "",
                    "expected_kind": "no_sign",
                    "capture_domain": "electronic_screen",
                    "split": split,
                    "device_slot": ("phone", "tablet", "monitor", "independent_device")[
                        split_index
                    ],
                    "actual_device_id": "",
                    "source_artwork_slot": f"negative-{negative_kind}-{split}",
                    "actual_source_artwork_id": "",
                    "base_rendering_slot": f"negative-base-{negative_kind}-{split}",
                    "actual_base_rendering_id": "",
                    "screen_orientation": ("portrait", "landscape")[split_index % 2],
                    "orientation_degrees": (-45, 0, 45, 15)[split_index],
                    "presentation_distance": ("near", "medium", "far", "medium")[split_index],
                    "presentation_brightness": ("low", "medium", "high", "medium")[
                        split_index
                    ],
                    "presentation_glare": ("none", "glare")[kind_index % 2],
                    "presentation_background": ("room", "vehicle", "outdoor")[kind_index],
                    "scene_composition": "negative_screen",
                    "presentation_material": "screen",
                    "negative_kind": negative_kind,
                    "actual_session_id": "",
                    "privacy_consent_status": "pending",
                    "review_status": (
                        "locked_pending_collection" if split == "phase_e2" else "pending"
                    ),
                }
            )
    return rows


def build_phase_b_collection_plan(
    output_dir: str | Path,
    *,
    labels_path: str | Path = "models/exported/runtime/sign_classifier.labels.json",
) -> dict[str, Any]:
    """Write immutable planning files; this does not create accepted dataset rows."""
    destination = project_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    labels_file = project_path(labels_path)
    labels = _load_labels(labels_file)
    road_path = destination / "recovery_phase_b_route_session_plan_v1.csv"
    presentation_path = destination / "recovery_phase_b_presentation_plan_v1.csv"
    summary_path = destination / "recovery_phase_b_collection_plan_v1.json"
    existing = [path for path in (road_path, presentation_path, summary_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite collection plan files: {existing}")
    road_rows = _road_rows()
    presentation_rows = _presentation_sign_rows(labels) + _presentation_negative_rows()
    _write_csv(road_path, ROAD_PLAN_FIELDS, road_rows)
    _write_csv(presentation_path, PRESENTATION_PLAN_FIELDS, presentation_rows)
    development_road = [row for row in road_rows if row["split"] != "phase_e2"]
    summary: dict[str, Any] = {
        "plan_version": PLAN_VERSION,
        "status": "planned_no_data",
        "generated_at": datetime.now(UTC).isoformat(),
        "supported_labels_path": str(labels_file),
        "supported_labels_sha256": _sha256(labels_file),
        "supported_classes": len(labels),
        "road": {
            "development_sessions": len(development_road),
            "phase_e2_sessions": sum(row["split"] == "phase_e2" for row in road_rows),
            "planned_development_full_context_frames": sum(
                _planned_count(row, "planned_full_context_frames")
                for row in development_road
            ),
            "planned_development_negative_frames": sum(
                _planned_count(row, "planned_negative_frames") for row in development_road
            ),
            "planned_development_sign_instances": sum(
                _planned_count(row, "planned_sign_instances") for row in development_road
            ),
        },
        "presentation": {
            "tasks": len(presentation_rows),
            "screen_sign_tasks": sum(
                row["capture_domain"] == "electronic_screen"
                and row["expected_kind"] == "sign"
                for row in presentation_rows
            ),
            "print_sign_tasks": sum(
                row["capture_domain"] == "printed_card" for row in presentation_rows
            ),
            "negative_screen_tasks": sum(
                row["expected_kind"] == "no_sign" for row in presentation_rows
            ),
        },
        "files": {
            "road": {"path": str(road_path), "sha256": _sha256(road_path)},
            "presentation": {
                "path": str(presentation_path),
                "sha256": _sha256(presentation_path),
            },
        },
        "warning": (
            "Planned rows are not images, annotations, reviews, consents, or Gate B evidence. "
            "Only accepted rows in the canonical intake manifest receive audit credit."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**summary, "summary_path": str(summary_path)}


__all__ = [
    "PRESENTATION_PLAN_FIELDS",
    "ROAD_PLAN_FIELDS",
    "build_phase_b_collection_plan",
]
