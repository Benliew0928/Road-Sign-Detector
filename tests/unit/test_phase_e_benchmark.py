from __future__ import annotations

# pyright: reportPrivateUsage=false
import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from roadsign_assist.datasets import phase_e_benchmark as benchmark


def test_bdd_candidate_bucketing_and_selection_are_deterministic() -> None:
    raw = {
        "image_id": "bdd-image-1",
        "image_bytes": {"src": "https://example.invalid/image.jpg", "width": 1280},
        "ann_categories": ["traffic sign", "car"],
        "ann_bboxes": [[1, 2, 20, 30], [4, 5, 40, 50]],
        "split": "train",
        "weather": "rainy",
        "scene": "city street",
        "timeofday": "daytime",
        "width": 1280,
        "height": 720,
        "is_duplicate": False,
    }

    first = benchmark._bdd_candidate(raw, 42)
    second = benchmark._bdd_candidate(raw, 42)

    assert first == second
    assert first is not None
    assert first.bucket == "adverse_weather"
    assert len(first.boxes) == 1
    assert benchmark._page_offsets(70_000) == benchmark._page_offsets(70_000)


def test_incomplete_owner_review_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "data" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(image_path)
    pixel, dhash_bits, _dhash_hex, width, height = benchmark._path_hashes(image_path)
    row: dict[str, object] = {
        "benchmark_id": benchmark.BENCHMARK_ID,
        "instance_id": "sample:box-001",
        "image_id": "sample",
        "image_path": "data/sample.jpg",
        "expected_kind": "sign",
        "domain": "in_domain_locked_test",
        "image_width": width,
        "image_height": height,
        "x1": 10,
        "y1": 10,
        "x2": 50,
        "y2": 50,
        "sha256": benchmark._sha256(image_path),
        "pixel_sha256": pixel,
        "dhash64": dhash_bits,
        "bbox_review_decision": "",
        "review_decision": "",
        "semantic_sign_id": "",
        "reviewer": "",
        "reviewed_at": "",
    }
    review_path = tmp_path / benchmark.REVIEW_PATH
    benchmark._write_csv(review_path, [row])
    benchmark._write_csv(tmp_path / benchmark.INVENTORY_PATH, [row])
    labels_path = tmp_path / benchmark.SUPPORTED_LABELS_PATH
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text(
        json.dumps([f"supported_label_{index}" for index in range(78)]),
        encoding="utf-8",
    )

    def empty_hashes(_root: Path) -> dict[str, object]:
        return {
            "detector_sha": set(),
            "detector_pixel": set(),
            "detector_dhash": [],
            "classifier_sha": set(),
            "classifier_pixel": set(),
            "classifier_dhash": [],
            "production_source_ids": set(),
        }

    monkeypatch.setattr(benchmark, "_existing_hashes", empty_hashes)

    with pytest.raises(ValueError, match="owner review is incomplete"):
        benchmark.validate_phase_e_review(project_root=tmp_path)

    failure = tmp_path / benchmark.AUDIT_ROOT / "review_validation_failure.json"
    assert failure.is_file()


def test_frozen_manifest_hash_change_is_detected(tmp_path: Path) -> None:
    canonical = tmp_path / benchmark.CANONICAL_PATH
    canonical.parent.mkdir(parents=True)
    with canonical.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id"])
        writer.writeheader()
        writer.writerow({"image_id": "image-1"})
    audit = tmp_path / benchmark.AUDIT_ROOT / "freeze_audit.json"
    benchmark._write_json(audit, {"canonical_sha256": benchmark._sha256(canonical)})

    assert benchmark.phase_e_benchmark_status(project_root=tmp_path)["status"] == "frozen"
    canonical.write_text(canonical.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash changed"):
        benchmark.phase_e_benchmark_status(project_root=tmp_path)
