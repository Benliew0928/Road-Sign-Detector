from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from roadsign_assist.datasets.recovery_collection_plan import build_phase_b_collection_plan


def test_collection_plan_predeclares_floors_without_claiming_data(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["curve_left", "curve_right"]), encoding="utf-8")

    report = build_phase_b_collection_plan(tmp_path / "plans", labels_path=labels)

    assert report["status"] == "planned_no_data"
    assert report["supported_classes"] == 2
    assert report["road"]["development_sessions"] == 30
    assert report["road"]["phase_e2_sessions"] == 6
    assert report["road"]["planned_development_full_context_frames"] == 18_000
    assert report["road"]["planned_development_negative_frames"] == 4_500
    assert report["road"]["planned_development_sign_instances"] == 21_000
    assert report["presentation"]["screen_sign_tasks"] == 8
    assert report["presentation"]["print_sign_tasks"] == 8
    assert report["presentation"]["negative_screen_tasks"] == 12
    presentation_path = tmp_path / "plans/recovery_phase_b_presentation_plan_v1.csv"
    with presentation_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["plan_status"] for row in rows} == {"planned_no_data"}
    assert {row["review_status"] for row in rows} == {
        "pending",
        "locked_pending_collection",
    }
    assert all(not row["actual_session_id"] for row in rows)


def test_collection_plan_is_immutable(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["curve_left"]), encoding="utf-8")
    output = tmp_path / "plans"
    build_phase_b_collection_plan(output, labels_path=labels)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_phase_b_collection_plan(output, labels_path=labels)
