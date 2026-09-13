from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from roadsign_assist.datasets.detector_review_package import (
    INDEX_FIELDS,
    build_emtd_detector_review_package,
    build_teammate_detector_review_package,
)


def test_review_package_renders_overlay_sheet_and_index(tmp_path: Path) -> None:
    image = tmp_path / "extracted/train/images/example.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (120, 80), (90, 120, 170)).save(image)
    label = tmp_path / "extracted/train/labels/example.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")
    queue = tmp_path / "queue.csv"
    fields = [
        "sample_id",
        "source_kind",
        "source_path",
        "layout_root_id",
        "source_image_id",
        "review_reasons",
        "review_decision",
        "reviewer_notes",
    ]
    with queue.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample-001",
                "source_kind": "teammate",
                "source_path": image.as_posix(),
                "layout_root_id": "source/utm/train",
                "source_image_id": "source-001",
                "review_reasons": "large_box",
                "review_decision": "",
                "reviewer_notes": "",
            }
        )

    output = tmp_path / "review_package"
    summary = build_teammate_detector_review_package(
        project_root=tmp_path,
        queue_path=queue,
        output_root=output,
    )

    assert summary["indexed_rows"] == 1
    assert summary["sheet_count"] == 1
    index_path = output / "teammate_review_index.csv"
    rows = list(csv.DictReader(index_path.open(encoding="utf-8")))
    assert list(rows[0]) == INDEX_FIELDS
    assert (output / rows[0]["review_sheet"]).is_file()


def test_emtd_review_package_renders_nested_image_layout(tmp_path: Path) -> None:
    image = tmp_path / "emtd/images/train/example.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (120, 80), (90, 120, 170)).save(image)
    label = tmp_path / "emtd/labels/train/example.txt"
    label.parent.mkdir(parents=True)
    label.write_text("5 0.5 0.5 0.4 0.5\n", encoding="utf-8")
    queue = tmp_path / "emtd_queue.csv"
    fields = [
        "sample_id",
        "source_kind",
        "source_path",
        "layout_root_id",
        "source_image_id",
        "review_reasons",
        "review_decision",
        "reviewer_notes",
    ]
    with queue.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "emtd-001",
                "source_kind": "emtd",
                "source_path": image.as_posix(),
                "layout_root_id": "emtd",
                "source_image_id": "source-001",
                "review_reasons": "tiny_box",
                "review_decision": "",
                "reviewer_notes": "",
            }
        )

    output = tmp_path / "emtd_review_package"
    summary = build_emtd_detector_review_package(
        project_root=tmp_path,
        queue_path=queue,
        output_root=output,
    )

    assert summary["indexed_rows"] == 1
    rows = list(csv.DictReader((output / "emtd_review_index.csv").open(encoding="utf-8")))
    assert (output / rows[0]["review_sheet"]).is_file()
