from __future__ import annotations

import csv
import json
from pathlib import Path

from roadsign_assist.datasets.recovery_phase_b import (
    PHASE_B_FIELDS,
    PhaseBFloors,
    audit_existing_recovery_inputs,
    audit_phase_b_manifest,
    write_phase_b_template,
)


def _catalogue(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "semantic_sign_id": "curve_left",
                        "severity": "critical",
                        "parameter_type": "direction",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _row(index: int, split: str, domain: str = "road_scene") -> dict[str, str]:
    row = {field: "" for field in PHASE_B_FIELDS}
    row.update(
        {
            "schema_version": "1.0",
            "release_id": "fixture_v1",
            "sample_id": f"sample-{index}",
            "instance_id": f"instance-{index}",
            "split": split,
            "expected_kind": "sign",
            "semantic_sign_id": "curve_left",
            "capture_domain": domain,
            "image_path": f"images/{index}.jpg",
            "label_path": f"labels/{index}.txt",
            "image_sha256": f"{index + 1:064x}",
            "image_width": "100",
            "image_height": "100",
            "bbox_x1": "0",
            "bbox_y1": "0",
            "bbox_x2": "10",
            "bbox_y2": "10",
            "area_ratio": "0.01",
            "size_bucket": "small",
            "source_id": "fixture",
            "session_id": f"session-{index}",
            "related_capture_group_id": f"group-{index}",
            "camera_pipeline_id": f"pipeline-{index % 3}",
            "annotation_status": "approved",
            "review_decision": "accept",
            "primary_reviewer": "reviewer-a",
            "secondary_reviewer": "reviewer-b",
            "licence_status": "approved",
            "use_policy": "internal_research",
        }
    )
    if domain == "road_scene":
        row.update(
            {
                "route_id": f"route-{index}",
                "physical_sign_id": f"physical-{index}",
                "camera_id": f"camera-{index % 3}",
                "weather": ("clear", "rain", "haze")[index % 3],
                "timeofday": ("day", "dusk", "night")[index % 3],
                "lighting": ("normal", "glare", "shadow", "backlight")[index % 4],
                "road_type": "urban",
            }
        )
    elif domain == "electronic_screen":
        row.update(
            {
                "source_artwork_id": "screen-artwork",
                "base_rendering_id": "screen-base",
                "device_id": f"screen-{index % 3}",
                "orientation_degrees": ("-45", "0", "45")[index % 3],
                "screen_orientation": ("portrait", "landscape")[index % 2],
                "presentation_distance": ("near", "medium", "far")[index % 3],
                "presentation_brightness": ("low", "medium", "high")[index % 3],
                "presentation_glare": ("none", "glare")[index % 2],
                "presentation_background": ("room", "vehicle")[index % 2],
                "scene_composition": ("single_sign", "search_results", "multi_sign")[index % 3],
            }
        )
    else:
        row.update(
            {
                "source_artwork_id": "print-artwork",
                "base_rendering_id": "print-base",
                "camera_id": "camera-print",
                "orientation_degrees": "0",
                "presentation_material": ("matte", "glossy")[index % 2],
                "presentation_distance": ("near", "medium", "far")[index % 3],
                "presentation_glare": "none",
                "presentation_background": "room",
                "scene_composition": "single_sign",
            }
        )
    return row


def _screen_negative(index: int, negative_kind: str) -> dict[str, str]:
    row = _row(index, "train", "electronic_screen")
    for field in (
        "instance_id",
        "semantic_sign_id",
        "label_path",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "area_ratio",
        "size_bucket",
    ):
        row[field] = ""
    row["expected_kind"] = "no_sign"
    row["negative_kind"] = negative_kind
    return row


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[str(field) for field in PHASE_B_FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def test_template_is_explicitly_empty_and_versioned(tmp_path: Path) -> None:
    template = write_phase_b_template(tmp_path / "intake.csv", release_id="release-v1")
    fields, *rows = list(csv.reader(template.open(encoding="utf-8")))
    assert fields == list(PHASE_B_FIELDS)
    assert rows == []
    metadata = json.loads(template.with_suffix(".template.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "collection_not_started"


def test_complete_fixture_passes_all_manifest_and_collection_checks(tmp_path: Path) -> None:
    catalogue = _catalogue(tmp_path / "catalogue.json")
    rows = [_row(index, "train") for index in range(26)]
    for index in range(3):
        rows[index] = _row(index, "train", "electronic_screen")
    rows[0]["orientation_degrees"] = "-45"
    rows[1]["orientation_degrees"] = "45"
    rows[2]["orientation_degrees"] = "0"
    rows[0]["screen_orientation"] = "portrait"
    rows[1]["screen_orientation"] = "landscape"
    rows[3] = _row(3, "train", "printed_card")
    rows[3]["presentation_material"] = "matte"
    rows[4] = _row(4, "train", "printed_card")
    rows[4]["presentation_material"] = "glossy"
    rows[5] = _row(5, "train", "printed_card")
    rows[5]["presentation_material"] = "matte"
    rows[6] = _screen_negative(6, "icons_or_logos")
    rows[7] = _screen_negative(7, "yellow_diamond")
    rows[8] = _screen_negative(8, "unrelated_traffic_imagery")
    for index in range(18, 22):
        rows[index] = _row(index, "validation")
    for index in range(22, 26):
        rows[index] = _row(index, "internal_test")
    rows.append(_row(26, "phase_e2"))
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)

    report = audit_phase_b_manifest(
        manifest,
        catalogue_path=catalogue,
        supported_labels_path=None,
        floors=PhaseBFloors(
            full_context_images=15,
            annotated_sign_instances=15,
            negative_fraction_min=0.0,
            negative_fraction_max=1.0,
            independent_sessions=1,
            camera_pipelines=1,
            small_instances=1,
            very_small_instances=0,
            electronic_devices_per_class=3,
            independent_eval_groups_per_safety_class=1,
        ),
    )

    assert report["gate_b_passed"] is True
    assert report["cross_split_leakage"] == {}


def test_owner_policy_only_waives_training_double_review(tmp_path: Path) -> None:
    rows = [_row(0, "train"), _row(1, "validation")]
    for row in rows:
        row.update(
            primary_reviewer="benli",
            secondary_reviewer="",
            approval_policy_id="owner_training_acceptance_20260905_v1",
            approval_basis="owner_blanket_acceptance",
        )
    path = tmp_path / "owner.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args = dict(catalogue_path=_catalogue(tmp_path / "catalogue.json"), supported_labels_path=None)
    strict = audit_phase_b_manifest(path, **args)
    approved = audit_phase_b_manifest(
        path,
        **args,
        training_approval_policy="configs/recovery/owner_training_approval_20260905.json",
    )
    assert sum(e["code"] == "double_review_required" for e in strict["validation_errors"]) == 2
    assert [
        e["row"] for e in approved["validation_errors"] if e["code"] == "double_review_required"
    ] == [3]
    assert not approved["gate_b_passed"]


def test_synthetic_presentations_do_not_prove_camera_coverage(tmp_path: Path) -> None:
    rows = [_row(0, "train", "electronic_screen"), _row(1, "train", "printed_card")]
    for row in rows:
        row["source_id"] = "deterministic_synthetic"
    path = tmp_path / "synthetic.csv"
    _write_manifest(path, rows)
    report = audit_phase_b_manifest(
        path, catalogue_path=_catalogue(tmp_path / "catalogue.json"), supported_labels_path=None
    )
    assert report["counts"]["sign_instances"] == 2
    assert not report["checks"]["screen_all_classes"]
    assert not report["checks"]["print_all_classes"]


def test_cross_split_session_leakage_blocks_gate(tmp_path: Path) -> None:
    catalogue = _catalogue(tmp_path / "catalogue.json")
    rows = [_row(0, "train"), _row(1, "phase_e2")]
    rows[1]["session_id"] = rows[0]["session_id"]
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)

    report = audit_phase_b_manifest(
        manifest,
        catalogue_path=catalogue,
        supported_labels_path=None,
    )

    assert report["checks"]["zero_cross_split_leakage"] is False
    assert report["cross_split_leakage"]["session_id"][0]["splits"] == [
        "phase_e2",
        "train",
    ]


def test_existing_release_gap_audit_counts_available_evidence(tmp_path: Path) -> None:
    catalogue = _catalogue(tmp_path / "catalogue.json")
    label = tmp_path / "sign.txt"
    label.write_text("0 0.5 0.5 0.05 0.01\n", encoding="utf-8")
    detector = tmp_path / "detector.csv"
    with detector.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "bbox_count", "is_negative", "dataset_label_path", "group_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "bbox_count": "1",
                "is_negative": "false",
                "dataset_label_path": str(label),
                "group_id": "g1",
            }
        )
        writer.writerow(
            {
                "split": "test",
                "bbox_count": "0",
                "is_negative": "true",
                "dataset_label_path": "",
                "group_id": "g2",
            }
        )
    classifier = tmp_path / "classifier.csv"
    with classifier.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "semantic_sign_id", "source_group"])
        writer.writeheader()
        writer.writerow({"split": "train", "semantic_sign_id": "curve_left", "source_group": "c1"})
        writer.writerow(
            {"split": "validation", "semantic_sign_id": "curve_left", "source_group": "c2"}
        )

    report = audit_existing_recovery_inputs(
        detector,
        classifier,
        catalogue_path=catalogue,
        supported_labels_path=None,
    )

    assert report["gate_b_passed"] is False
    assert report["detector"]["annotated_sign_instances"] == 1
    assert report["detector"]["very_small_instances_at_most_0_1pct"] == 1
    assert report["classifier"]["minimum_train_examples_per_class"] == 1
