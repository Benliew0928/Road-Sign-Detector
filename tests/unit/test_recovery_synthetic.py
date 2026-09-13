from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

from roadsign_assist.datasets.recovery_synthetic import generate_presentation_candidates


def _base_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "bases.csv"
    fields = [
        "sample_id",
        "split",
        "semantic_sign_id",
        "dataset_image_path",
        "crop_sha256",
        "source_group",
        "source_dataset",
        "source_url",
        "licence_status",
        "review_decision",
    ]
    rows: list[dict[str, str]] = []
    for label_index in range(78):
        label = f"class_{label_index:02d}"
        for artwork_index in range(3):
            image = np.full(
                (48, 48, 3),
                (label_index * 3 + artwork_index) % 255,
                dtype=np.uint8,
            )
            cv2.circle(image, (24, 24), 12 + artwork_index, (0, 0, 255), 2)
            path = tmp_path / f"{label}_{artwork_index}.jpg"
            assert cv2.imwrite(str(path), image)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(
                {
                    "sample_id": f"{label}-{artwork_index}",
                    "split": "train",
                    "semantic_sign_id": label,
                    "dataset_image_path": str(path),
                    "crop_sha256": digest,
                    "source_group": f"group-{label}-{artwork_index}",
                    "source_dataset": "test",
                    "source_url": "",
                    "licence_status": "reviewed",
                    "review_decision": "accept",
                }
            )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_synthetic_generation_is_balanced_and_quarantined(tmp_path: Path) -> None:
    report = generate_presentation_candidates(
        batch_id="synthetic-v1",
        total_images=100,
        base_manifest=_base_manifest(tmp_path),
        acquisition_root=tmp_path / "archive",
        audit_root=tmp_path / "audit",
        minimum_free_gib=0,
        maximum_acquisition_gib=1,
    )

    assert report["generated_rows"] == 100
    assert report["class_balance_range"] == [1, 2]
    assert set(report["negative_counts"]) == {
        "icons_or_logos",
        "unrelated_traffic_imagery",
        "yellow_diamond",
    }
    assert report["all_rows_quarantined"] is True
    assert report["group_split_crossings"] == 0


def test_synthetic_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = generate_presentation_candidates(
        batch_id="dry-run",
        total_images=100,
        base_manifest=_base_manifest(tmp_path),
        acquisition_root=tmp_path / "archive",
        audit_root=tmp_path / "audit",
        minimum_free_gib=0,
        maximum_acquisition_gib=1,
        dry_run=True,
    )
    assert result["classes"] == 78
    assert not (tmp_path / "archive").exists()
