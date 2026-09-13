from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_owner_release_preserves_unreadable_and_unknown_capture_metadata() -> None:
    script = _script("promote_recovery_owner_training")
    row = script.real_row(
        {"expected_kind": "unreadable", "semantic_sign_id": "unknown_sign"},
        {
            "semantic_sign_id": "unknown_sign",
            "item_kind": "real_sign_proposal",
            "object_id": "one",
            "source_id": "emtd_zenodo_1217105",
        },
        {},
    )
    assert row["expected_kind"] == "unreadable"
    assert row["capture_evidence"] == "unverified_context"
    assert row["session_id"] == "unknown_not_recorded"
    assert row["physical_sign_id"] == "unknown_not_recorded"
    assert row["camera_pipeline_id"] == "unknown_not_recorded"


def test_checkpoint_quantifies_gaps_without_claiming_global_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script("close_recovery_collection_checkpoint")
    root = tmp_path / "archive"
    release = root / "releases/fixture"
    release.mkdir(parents=True)
    canonical = release / "canonical_intake.csv"
    canonical.write_text(
        "source_id,image_sha256,expected_kind\nsource,abc,unknown_sign\n", encoding="utf-8"
    )
    (release / "quality_exclusions.csv").write_text("instance_id\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "counts": {
                    "road_development_images": 2,
                    "road_development_sign_instances": 1,
                    "road_development_small_instances_at_most_1pct": 1,
                    "road_development_very_small_instances_at_most_0_1pct": 0,
                    "road_development_independent_sessions": 1,
                    "road_development_camera_pipelines": 1,
                    "road_development_negative_fraction": 0.5,
                },
                "floors": {
                    "full_context_images": 15,
                    "annotated_sign_instances": 20,
                    "small_instances": 3,
                    "very_small_instances": 1,
                    "independent_sessions": 3,
                    "camera_pipelines": 3,
                    "negative_fraction_min": 0.2,
                    "negative_fraction_max": 0.3,
                },
                "coverage_gaps": {"screen_classes": ["stop"]},
                "checks": {"row_validation": True, "full_context_image_floor": False},
                "gate_b_passed": False,
            }
        ),
        encoding="utf-8",
    )
    (release / "promotion_report.json").write_text(
        json.dumps(
            {
                "audit": {"json": str(audit)},
                "canonical_manifest": str(canonical),
                "pending_owner_reviews": 0,
            }
        ),
        encoding="utf-8",
    )
    cycles = root / "manifests/cycles"
    cycles.mkdir(parents=True)
    (cycles / "source.json").write_text(
        json.dumps(
            {
                "cycle_id": "source",
                "source_id": "source",
                "status": "listed_source_exhausted",
                "batches": [{"batch_id": "b001", "downloaded": 2, "unique_normalized": 1}],
            }
        ),
        encoding="utf-8",
    )
    downloads = root / "manifests/download_reports"
    downloads.mkdir()
    (downloads / "b001_download.json").write_text(
        json.dumps({"failed": [{"error": "archived"}], "skipped": []}), encoding="utf-8"
    )
    freezes = root / "manifests/freezes"
    freezes.mkdir()
    freeze = freezes / "fixture.json"
    freeze.write_text(
        json.dumps({"files": {str(canonical): script.sha256(canonical)}}), encoding="utf-8"
    )
    monkeypatch.setattr(script, "project_path", lambda _: root)
    monkeypatch.setattr(sys, "argv", ["checkpoint", "--release-id", "fixture"])
    script.main()
    result = json.loads((release / "collection_checkpoint.json").read_text(encoding="utf-8"))
    assert result["exact_numeric_gaps"]["real_full_context_images"]["missing"] == 13
    assert (
        result["cycles"][0]["interpreted_status"]
        == "finite_manifest_completed_with_recorded_failed_or_skipped_attempts"
    )
    assert result["pending_owner_batch_reviews"] == 0
    assert result["minor_plan_globally_complete"] is False
    assert result["major_plan_complete"] is False
    with pytest.raises(FileExistsError):
        script.main()
