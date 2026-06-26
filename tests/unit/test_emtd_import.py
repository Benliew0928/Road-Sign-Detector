from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from roadsign_assist.datasets.emtd_import import import_emtd_subset


def test_emtd_import_builds_detection_and_classification_data(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    image_paths: list[Path] = []
    for index, color in enumerate(((220, 30, 30), (30, 80, 220), (220, 200, 20))):
        path = raw / f"sample-{index}.jpg"
        Image.new("RGB", (100, 80), color=color).save(path)
        image_paths.append(path)

    subset = tmp_path / "subset.csv"
    with subset.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "relative_path",
                "class_ids",
                "compressed_size",
                "uncompressed_size",
                "sha256",
                "status",
            ],
        )
        writer.writeheader()
        for index, path in enumerate(image_paths):
            writer.writerow(
                {
                    "filename": path.name,
                    "relative_path": path,
                    "class_ids": str(index + 1),
                    "compressed_size": path.stat().st_size,
                    "uncompressed_size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "status": "downloaded",
                }
            )

    ground_truth = tmp_path / "GT.csv"
    with ground_truth.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "Class ID", "xmin", "ymin", "xmax", "ymax"],
        )
        writer.writeheader()
        for index, path in enumerate(image_paths):
            writer.writerow(
                {
                    "filename": path.name,
                    "Class ID": index + 1,
                    "xmin": 10,
                    "ymin": 10,
                    "xmax": 60,
                    "ymax": 60,
                }
            )

    class_mapping = tmp_path / "mapping.json"
    class_mapping.write_text(
        json.dumps(
            {
                "review_status": "draft_single_review",
                "classes": {
                    str(index + 1): {
                        "semantic_sign_id": f"class_{index + 1}",
                        "parameter_value": None,
                        "confidence": 1.0,
                    }
                    for index in range(3)
                },
            }
        ),
        encoding="utf-8",
    )

    stats = import_emtd_subset(
        subset_manifest_path=subset,
        ground_truth_path=ground_truth,
        class_mapping_path=class_mapping,
        dataset_manifest_path=tmp_path / "dataset.csv",
        annotation_path=tmp_path / "boxes.csv",
        split_root=tmp_path / "splits",
        processed_root=tmp_path / "processed",
        duplicate_threshold=0,
    )
    assert stats["images"] == 3
    assert stats["instances"] == 3
    assert stats["classification_crops"] == 3
    assert (tmp_path / "processed/emtd_detection/data.yaml").exists()
    assert len(list((tmp_path / "processed/emtd_classification").rglob("*.jpg"))) == 3
    classifier_metadata = json.loads(
        (tmp_path / "processed/emtd_classification/dataset_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert classifier_metadata["split_strategy"] == "deterministic_group_stratified"
    assert "classification_split_label_counts" in classifier_metadata
