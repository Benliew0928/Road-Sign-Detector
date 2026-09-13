from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from roadsign_assist.datasets.final_classifier_release import (
    CONTRIBUTOR_MANIFEST_SPECS,
    ContributorManifestSpec,
    FinalClassifierReleaseConfig,
    build_final_classifier_release,
    is_accepted_contributor_decision,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_image(path: Path, colour: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), colour).save(path)
    return image_hash(path)


def create_inputs(tmp_path: Path, contributor_rows: list[dict[str, str]]) -> FinalClassifierReleaseConfig:
    root = tmp_path
    labels = [f"label_{number:02d}" for number in range(78)]
    frozen_rows: list[dict[str, str]] = []
    for index, label in enumerate(labels):
        split = "validation" if index == 0 else "test" if index == 1 else "train"
        image = root / "data/processed/v2" / split / label / f"base_{index}.png"
        digest = write_image(image, ((index * 13) % 256, (index * 29) % 256, (index * 47) % 256))
        frozen_rows.append(
            {
                "sample_id": f"base_{index}",
                "split": split,
                "semantic_sign_id": label,
                "dataset_image_path": image.relative_to(root).as_posix(),
                "source_crop_path": image.relative_to(root).as_posix(),
                "crop_sha256": digest,
                "name_en": label,
                "priority": "must",
                "required_for": "test",
                "leakage_group": f"base-{index}",
                "source_group": f"base-{index}",
                "review_decision": "accept",
            }
        )
    manifest_dir = root / "data/manifests"
    write_csv(manifest_dir / "classifier_release.csv", frozen_rows)
    write_csv(manifest_dir / "classifier_production_78_v2_20260826.csv", frozen_rows)
    write_csv(
        manifest_dir / "stage_c_test_contributors.csv",
        contributor_rows,
    )
    tracker = [
        {
            "semantic_sign_id": label,
            "priority": "must",
            "minimum_clean_crops": "1",
            "gap_to_minimum": "0",
            "status": "baseline",
            "next_action": "none",
        }
        for label in labels
    ]
    write_csv(manifest_dir / "PRODUCTION_78_DATA_PROGRESS.csv", tracker)
    return FinalClassifierReleaseConfig(
        root,
        release_id="v3_test",
        output_root=root / "data/processed/v3_test",
        output_manifest=manifest_dir / "v3_test.csv",
        audit_root=root / "outputs/audit/v3_test",
        v2_root=root / "data/processed/v2",
        contributor_manifest_specs={
            "stage_c_test_contributors.csv": ContributorManifestSpec(
                "stage_c_test_contributors.csv", "label_00"
            )
        },
        perceptual_hamming_threshold=-1,
    )


def contributor_row(root: Path, candidate_id: str, *, group: str, rights: str = "rights_unverified") -> dict[str, str]:
    crop = root / "data/raw/contributors" / f"{candidate_id}.png"
    original = root / "data/raw/originals" / f"{candidate_id}.png"
    colour = (20 + len(candidate_id), 130, 220)
    crop_hash = write_image(crop, colour)
    original_hash = write_image(original, colour)
    return {
        "candidate_id": candidate_id,
        "crop_path": crop.relative_to(root).as_posix(),
        "original_path": original.relative_to(root).as_posix(),
        "crop_sha256": crop_hash,
        "original_sha256": original_hash,
        "source_group": group,
        "source_video_group": "not_explicitly_named_video_sequence",
        "source_page_url": "https://example.test/source",
        "source_original_dataset": "fixture dataset",
        "rights_status": rights,
        "stated_license": "fixture licence",
        "small_object": "false",
        "final_review_decision": "manual_accept_after_review",
        "reviewer_notes": "approved fixture",
    }


def test_manifest_mapping_and_acceptance_codes_are_explicit() -> None:
    assert len(CONTRIBUTOR_MANIFEST_SPECS) == 21
    assert CONTRIBUTOR_MANIFEST_SPECS[
        "stage_c_turn_left_or_right_catsad_batch_01_20260828.csv"
    ].semantic_sign_id == "turn_left_or_right"
    assert is_accepted_contributor_decision("accept")
    assert is_accepted_contributor_decision("accept_user_manual_review")
    assert is_accepted_contributor_decision("manual_accept_exact_physical_target")
    assert not is_accepted_contributor_decision("hold_ambiguous")
    assert not is_accepted_contributor_decision("reject")


def test_freeze_preserves_locked_eval_and_tracks_internal_exception(tmp_path: Path) -> None:
    accepted = contributor_row(tmp_path, "contributor_001", group="scene-001")
    rejected = dict(accepted, candidate_id="contributor_rejected", final_review_decision="reject")
    config = create_inputs(tmp_path, [accepted, rejected])

    audit = build_final_classifier_release(config)

    assert audit["gates"]["frozen_validation_and_test_preserved"] is True
    assert audit["counts"]["retained_contributor_rows"] == 1
    assert audit["dvc_remote_push_allowed"] is False
    assert audit["external_data_or_model_release_allowed"] is False
    canonical = list(csv.DictReader(config.output_manifest.open(encoding="utf-8")))
    contributor = next(row for row in canonical if row["sample_id"] == "contributor_001")
    assert contributor["split"] == "train"
    assert contributor["licence_status"] == "internal_academic_exception"
    assert contributor["internal_academic_exception_reasons"]
    for split, sample in (("validation", "base_0"), ("test", "base_1")):
        row = next(item for item in canonical if item["sample_id"] == sample)
        source = tmp_path / "data/processed/v2" / split / row["semantic_sign_id"] / f"{sample}.png"
        assert image_hash(tmp_path / row["dataset_image_path"]) == image_hash(source)
    inventory = list(
        csv.DictReader((config.audit_root / "contributor_inventory.csv").open(encoding="utf-8"))
    )
    assert {row["inventory_status"] for row in inventory} == {"retained", "excluded"}
    progress = list(csv.DictReader(config.progress_tracker.open(encoding="utf-8")))
    assert progress[0]["last_release_id"] == "v3_test"
    assert progress[0]["production_78_v3_total_count"] == "2"
    assert json.loads((config.output_root / "dataset_metadata.json").read_text())[
        "internal_runtime_promotion_eligible"
    ] is True
    metadata = json.loads((config.output_root / "dataset_metadata.json").read_text())
    assert metadata["annotation_status"] == "approved_with_internal_academic_exception"
    assert metadata["coursework_images_included"] == 0
    assert metadata["publication_prohibited"] is True


def test_global_group_dedupe_prefers_documented_licence_row(tmp_path: Path) -> None:
    exception_row = contributor_row(tmp_path, "contributor_001", group="scene-001")
    documented_row = contributor_row(
        tmp_path,
        "contributor_002",
        group="scene-001",
        rights="stated_license",
    )
    config = create_inputs(tmp_path, [exception_row, documented_row])

    audit = build_final_classifier_release(config)

    assert audit["counts"]["retained_contributor_rows"] == 1
    canonical = list(csv.DictReader(config.output_manifest.open(encoding="utf-8")))
    retained = [row for row in canonical if row["sample_id"].startswith("contributor_")]
    assert [row["sample_id"] for row in retained] == ["contributor_002"]
    decisions = list(
        csv.DictReader((config.audit_root / "duplicate_decisions.csv").open(encoding="utf-8"))
    )
    assert any(row["dedupe_resolution"].startswith("excluded_component_winner") for row in decisions)


def test_hash_mismatch_is_audited_and_outputs_refuse_overwrite(tmp_path: Path) -> None:
    row = contributor_row(tmp_path, "contributor_001", group="scene-001")
    row["crop_sha256"] = "0" * 64
    config = create_inputs(tmp_path, [row])

    audit = build_final_classifier_release(config)

    assert audit["gates"]["invalid_contributor_rows"] == 1
    exclusions = list(csv.DictReader((config.audit_root / "exclusions.csv").open(encoding="utf-8")))
    assert exclusions[0]["exclusion_reason"].startswith("SHA-256 mismatch")
    with pytest.raises(FileExistsError, match="output already exists"):
        build_final_classifier_release(config)


def test_approved_coverage_waiver_authorizes_phase_b(tmp_path: Path) -> None:
    row = contributor_row(tmp_path, "contributor_001", group="scene-001")
    config = create_inputs(tmp_path, [row])
    tracker = list(csv.DictReader(config.progress_tracker.open(encoding="utf-8")))
    tracker[0]["minimum_clean_crops"] = "3"
    write_csv(config.progress_tracker, tracker)
    config = replace(
        config,
        allow_must_have_coverage_exception=True,
        coverage_exception_note="Owner approved 48/50 equivalent fixture waiver.",
    )

    audit = build_final_classifier_release(config)

    assert audit["gates"]["must_have_coverage"] is False
    assert audit["gates"]["must_have_coverage_exception_approved"] is True
    assert audit["gates"]["phase_a_training_ready"] is True
    metadata = json.loads((config.output_root / "dataset_metadata.json").read_text())
    assert metadata["phase_b_authorized"] is True
    assert metadata["coverage_exception"]["classes_below_minimum"] == ["label_00"]
