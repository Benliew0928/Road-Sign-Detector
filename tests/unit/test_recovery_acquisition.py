from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import pytest
from PIL import Image

from roadsign_assist.datasets.recovery_acquisition import (
    build_spent_denylist,
    check_disk_budget,
    inventory_local_candidates,
)


def _image(path: Path, colour: tuple[int, int, int] = (10, 20, 30)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), colour).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_inventory_hashes_joins_deduplicates_and_denylist(tmp_path: Path) -> None:
    data = tmp_path / "data"
    first = data / "processed/detector/images/train/a.jpg"
    digest = _image(first)
    duplicate = data / "raw/copy.jpg"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(first.read_bytes())
    label = data / "processed/detector/labels/train/a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.05 0.01\n", encoding="utf-8")
    manifest = data / "manifests/release.csv"
    _csv(
        manifest,
        [
            "sample_id",
            "dataset_image_path",
            "source_id",
            "group_id",
            "split",
            "sha256",
            "review_decision",
            "licence_status",
            "use_policy",
        ],
        [
            {
                "sample_id": "a",
                "dataset_image_path": str(first),
                "source_id": "source_a",
                "group_id": "group_a",
                "split": "train",
                "sha256": digest,
                "review_decision": "accept",
                "licence_status": "accepted",
                "use_policy": "training",
            }
        ],
    )
    spent = data / "manifests/spent.csv"
    _csv(
        spent,
        ["benchmark_id", "image_id", "instance_id", "image_path", "sha256", "crop_sha256"],
        [
            {
                "benchmark_id": "spent",
                "image_id": "a",
                "instance_id": "a:1",
                "image_path": str(first),
                "sha256": digest,
                "crop_sha256": "f" * 64,
            }
        ],
    )
    acquisition = tmp_path / "archive"
    audit = tmp_path / "audit"

    report = inventory_local_candidates(
        scan_roots=(data,),
        manifest_roots=(data / "manifests",),
        acquisition_root=acquisition,
        audit_root=audit,
        spent_manifest=spent,
        minimum_free_gib=0,
        maximum_acquisition_gib=1,
    )

    assert report["exit_condition"]["passed"] is True
    assert report["counts"]["objects"] == 2
    assert report["counts"]["unique_sha256"] == 1
    assert report["counts"]["bbox_instances"] == 1
    assert report["counts"]["very_small_bbox_instances_at_most_0_1pct"] == 1
    parquet: Any = pq
    rows = cast(
        list[dict[str, Any]],
        parquet.read_table(acquisition / "manifests/local_inventory.parquet").to_pylist(),
    )
    assert {row["collection_state"] for row in rows} == {"spent_evaluation_denylisted"}
    assert all(row["exact_duplicate_path_count"] == 2 for row in rows)
    joined_sources = json.loads(
        next(row for row in rows if row["relative_path"].endswith("a.jpg"))["sources_json"]
    )
    assert "source_a" in joined_sources
    assert (acquisition / "manifests/schemas/object_ledger.schema.json").is_file()


def test_inventory_dry_run_and_resume_are_safe(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _image(data / "one.png")
    kwargs: dict[str, Any] = {
        "scan_roots": (data,),
        "manifest_roots": (data / "manifests",),
        "acquisition_root": tmp_path / "archive",
        "audit_root": tmp_path / "audit",
        "spent_manifest": tmp_path / "missing.csv",
        "minimum_free_gib": 0,
        "maximum_acquisition_gib": 1,
    }
    dry = inventory_local_candidates(**kwargs, dry_run=True)
    assert dry["candidate_objects"] == 1
    assert not (tmp_path / "archive/manifests/local_inventory.parquet").exists()

    inventory_local_candidates(**kwargs)
    with pytest.raises(FileExistsError, match="without --resume"):
        inventory_local_candidates(**kwargs)
    resumed = inventory_local_candidates(**kwargs, resume=True)
    assert resumed["counts"]["resume_rows_reused"] == 1


def test_disk_guard_and_spent_denylist_missing_manifest(tmp_path: Path) -> None:
    budget = check_disk_budget(
        tmp_path,
        acquisition_root=tmp_path / "archive",
        minimum_free_gib=0,
        maximum_acquisition_gib=1,
    )
    assert budget.allowed is True
    rows, hashes = build_spent_denylist(tmp_path / "missing.csv")
    assert rows == []
    assert hashes == set()
