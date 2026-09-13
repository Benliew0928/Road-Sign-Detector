from __future__ import annotations

import csv
from pathlib import Path

from roadsign_assist.datasets.phase_c_negative_candidates import (
    MANIFEST_FIELDS,
    REVIEW_FIELDS,
    TARGET,
    approve_phase_c_negative_candidates,
    candidate_from_row,
)


def test_only_annotated_no_sign_road_rows_are_eligible() -> None:
    base = {
        "image_id": "frame-001",
        "image_bytes": {"src": "https://example.invalid/image.jpg", "width": 1280, "height": 720},
        "scene": "highway",
        "timeofday": "daytime",
        "weather": "clear",
        "split": "train",
        "ann_categories": ["car", "truck"],
        "is_duplicate": False,
    }
    candidate = candidate_from_row(base, 4)
    assert candidate is not None
    assert candidate.stratum == ("highway", "daytime")

    with_sign = {**base, "ann_categories": ["car", "traffic sign"]}
    assert candidate_from_row(with_sign, 4) is None
    source_duplicate = {**base, "is_duplicate": True}
    assert candidate_from_row(source_duplicate, 4) is None
    not_a_road_scene = {**base, "scene": "parking lot"}
    assert candidate_from_row(not_a_road_scene, 4) is None


def test_owner_approval_updates_both_negative_manifests(tmp_path: Path) -> None:
    manifest = tmp_path / "candidates.csv"
    review = tmp_path / "review.csv"
    manifest_rows = [
        {field: "" for field in MANIFEST_FIELDS}
        | {"sample_id": f"sample-{index}", "source_kind": "negative"}
        for index in range(TARGET)
    ]
    review_rows = [
        {field: "" for field in REVIEW_FIELDS} | {"sample_id": f"sample-{index}"}
        for index in range(TARGET)
    ]
    for path, rows, fields in (
        (manifest, manifest_rows, MANIFEST_FIELDS),
        (review, review_rows, REVIEW_FIELDS),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    result = approve_phase_c_negative_candidates(
        project_root=tmp_path,
        manifest_path=manifest,
        review_path=review,
    )

    assert result["accepted_count"] == TARGET
    with review.open(newline="", encoding="utf-8") as handle:
        assert {row["review_decision"] for row in csv.DictReader(handle)} == {"accept"}
