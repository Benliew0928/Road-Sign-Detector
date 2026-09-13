from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from roadsign_assist.datasets.final_detector_release import (
    DetectorReleaseConfig,
    ReviewRequiredError,
    _assign_splits,
    audit_approved_negatives_against_teammate,
    build_final_detector_release,
    reassemble_and_extract_detector_archives,
    record_emtd_review,
    record_teammate_layout_review,
    verify_bundle_checksums,
)


def write_image(path: Path, colour: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), colour).save(path)


def tar_bytes(entries: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, contents in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    return stream.getvalue()


def encoded_image(colour: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (100, 80), colour).save(stream, format="JPEG")
    return stream.getvalue()


def write_checksums(bundle: Path, entries: dict[str, bytes]) -> None:
    rows = [f"{hashlib.sha256(contents).hexdigest()}  {name}" for name, contents in entries.items()]
    (bundle / "CHECKSUMS.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def create_bundle(tmp_path: Path, *, bad_label: bool = False) -> Path:
    bundle = tmp_path / "bundle"
    detector = bundle / "detector_generic_yolo"
    detector.mkdir(parents=True)
    canonical_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    checksum_entries: dict[str, bytes] = {"README.md": b"fixture"}
    colours = {"train": (220, 20, 20), "validation": (20, 220, 20), "test": (20, 20, 220)}
    for split, colour in colours.items():
        image = encoded_image(colour)
        name = f"{split}.jpg"
        label = "0 0.5 0.5 0.2 0.2\n" if not bad_label or split != "train" else "0 0.5 0.5 1.2 0.2\n"
        archive_bytes = tar_bytes({f"images/{name}": image, f"labels/{split}.txt": label.encode()})
        part_count = 10 if split == "train" else 2
        for index in range(part_count):
            start = len(archive_bytes) * index // part_count
            end = len(archive_bytes) * (index + 1) // part_count
            (detector / f"{split}.tar.part{index + 1:03d}").write_bytes(archive_bytes[start:end])
        checksum_entries[f"detector_generic_yolo/{split}.tar"] = archive_bytes
        digest = hashlib.sha256(image).hexdigest()
        sample_id = f"source-{split}"
        canonical_rows.append(
            {
                "task": "detector",
                "image_sha256": digest,
                "source_id": "owner_drive",
                "source_image_id": sample_id,
                "layout_root_id": f"layout-{split}",
            }
        )
        split_rows.append(
            {
                "task": "detector",
                "source_image_id": sample_id,
                "group": "connected-shared" if split in {"train", "validation"} else "connected-test",
                "split": split,
            }
        )
    (bundle / "manifests").mkdir()
    (bundle / "manifests/canonical_samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in canonical_rows) + "\n", encoding="utf-8"
    )
    (bundle / "manifests/release_splits.jsonl").write_text(
        "\n".join(json.dumps(row) for row in split_rows) + "\n", encoding="utf-8"
    )
    checksum_entries["manifests/canonical_samples.jsonl"] = (
        bundle / "manifests/canonical_samples.jsonl"
    ).read_bytes()
    checksum_entries["manifests/release_splits.jsonl"] = (
        bundle / "manifests/release_splits.jsonl"
    ).read_bytes()
    (bundle / "README.md").write_bytes(b"fixture")
    write_checksums(bundle, checksum_entries)
    return bundle


def write_emtd(root: Path) -> None:
    for index, colour in enumerate(((30, 30, 30), (70, 70, 70))):
        image = root / "images/train" / f"emtd-{index}.jpg"
        write_image(image, colour)
        label = root / "labels/train" / f"emtd-{index}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def write_negatives(root: Path, count: int) -> None:
    for index in range(count):
        write_image(root / f"negative-{index}.jpg", (index + 10, index + 40, index + 90))


def mark_queue_accepted(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["review_decision"] = "accept"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_id", "review_decision"])
        writer.writeheader()
        writer.writerows(rows)


def config_for(tmp_path: Path, bundle: Path) -> DetectorReleaseConfig:
    root = tmp_path / "project"
    emtd = root / "emtd"
    negatives = root / "negatives"
    write_emtd(emtd)
    write_negatives(negatives, 2)
    return DetectorReleaseConfig(
        project_root=root,
        bundle_root=bundle,
        quarantine_root=tmp_path / "quarantine",
        emtd_root=emtd,
        negative_root=negatives,
        output_root=root / "release",
        output_manifest=root / "manifest.csv",
        audit_root=root / "audit",
        review_root=root / "reviews",
        negative_target=2,
    )


def test_bundle_checksum_and_multipart_reassembly(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    report = verify_bundle_checksums(bundle)
    assert report["passed_materialized_entries"] == 3
    config = config_for(tmp_path, bundle)
    reassembly = reassemble_and_extract_detector_archives(config, overwrite=False)
    assert len(reassembly["archives"]) == 3
    assert reassembly["archives"][0]["parts"][-1] == "train.tar.part010"
    assert (config.quarantine_root / "extracted/train/images/train.jpg").is_file()


def test_checksum_mismatch_refuses_reassembly(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    (bundle / "README.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum validation failed"):
        verify_bundle_checksums(bundle)


def test_reuse_verified_quarantine_revalidates_archives_without_extraction(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    config = config_for(tmp_path, bundle)
    first = reassemble_and_extract_detector_archives(config, overwrite=False)

    reused = reassemble_and_extract_detector_archives(
        config,
        overwrite=False,
        reuse_existing=True,
    )

    assert [row["sha256"] for row in reused["archives"]] == [
        row["sha256"] for row in first["archives"]
    ]
    assert all("prior_audit_sha256" in row for row in reused["archives"])


def test_group_safe_split_assignment_balances_absolute_targets() -> None:
    candidates = [SimpleNamespace(group_id=f"group-{index:04d}") for index in range(1_000)]

    assignments = _assign_splits(candidates, seed=2513)
    counts = {split: list(assignments.values()).count(split) for split in ("train", "validation", "test")}

    assert counts == {"train": 700, "validation": 150, "test": 150}


def test_release_requires_reviews_then_builds_deterministically(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    config = config_for(tmp_path, bundle)
    with pytest.raises(ReviewRequiredError):
        build_final_detector_release(config)
    for queue in config.review_root.glob("*.csv"):
        mark_queue_accepted(queue)
    audit = build_final_detector_release(config, overwrite=True)
    assert audit["gates"]["review_complete"] is True
    assert audit["counts"]["negative_retained"] == 2
    assert audit["dvc_remote_push_allowed"] is False
    rows = list(csv.DictReader(config.output_manifest.open(encoding="utf-8")))
    assert {row["source_kind"] for row in rows} == {"teammate", "emtd", "negative"}
    assert {row["licence_status"] for row in rows} == {"assignment_only_waiver"}
    assert all(row["bbox_count"] == "0" for row in rows if row["is_negative"] == "true")
    by_group: dict[str, set[str]] = {}
    for row in rows:
        if row["is_negative"] == "false":
            by_group.setdefault(row["group_id"], set()).add(row["split"])
        label = tmp_path / "project" / row["dataset_label_path"]
        assert all(line.startswith("0 ") for line in label.read_text(encoding="utf-8").splitlines())
    assert all(len(splits) == 1 for splits in by_group.values())
    first_manifest = config.output_manifest.read_bytes()
    second = build_final_detector_release(config, overwrite=True)
    assert second["counts"] == audit["counts"]
    assert config.output_manifest.read_bytes() == first_manifest


def test_invalid_yolo_label_is_excluded_from_candidate_scan(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path, bad_label=True)
    config = config_for(tmp_path, bundle)
    with pytest.raises(ReviewRequiredError):
        build_final_detector_release(config)
    for queue in config.review_root.glob("*.csv"):
        mark_queue_accepted(queue)
    audit = build_final_detector_release(config, overwrite=True)
    assert audit["counts"]["teammate_excluded"] == 1


def test_rewriting_pending_queues_preserves_completed_negative_review(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    config = config_for(tmp_path, bundle)
    with pytest.raises(ReviewRequiredError):
        build_final_detector_release(config)
    negative_queue = config.review_root / "negative_no_sign_review.csv"
    mark_queue_accepted(negative_queue)

    with pytest.raises(ReviewRequiredError):
        build_final_detector_release(config, overwrite=True)

    rows = list(csv.DictReader(negative_queue.open(encoding="utf-8")))
    assert rows
    assert {row["review_decision"] for row in rows} == {"accept"}


def test_recording_teammate_layout_review_is_guarded_and_auditable(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    config = config_for(tmp_path, bundle)
    with pytest.raises(ReviewRequiredError):
        build_final_detector_release(config)

    queue = config.review_root / "teammate_visual_review.csv"
    result = record_teammate_layout_review(
        queue,
        layout_root_ids=["layout-train", "layout-validation"],
        decision="accept",
        reviewer_notes="Owner reviewed every rendered overlay in these layouts.",
    )

    assert result["updated_count"] == 2
    assert result["remaining_pending_count"] == 1
    assert Path(str(result["backup"])).is_file()
    rows = list(csv.DictReader(queue.open(encoding="utf-8")))
    assert {row["review_decision"] for row in rows if row["layout_root_id"] != "layout-test"} == {
        "accept"
    }
    with pytest.raises(ValueError, match="Refusing to replace"):
        record_teammate_layout_review(
            queue,
            layout_root_ids=["layout-train"],
            decision="reject",
            reviewer_notes="A second decision must not replace the first.",
        )


def test_recording_emtd_review_updates_only_emtd_queue(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    config = config_for(tmp_path, bundle)
    with pytest.raises(ReviewRequiredError):
        build_final_detector_release(config)

    queue = config.review_root / "emtd_box_review.csv"
    result = record_emtd_review(
        queue,
        decision="accept",
        reviewer_notes="Owner reviewed every EMTD overlay; all boxes were approved.",
    )

    assert result["updated_count"] == 2
    assert result["remaining_pending_count"] == 0
    assert Path(str(result["backup"])).is_file()
    rows = list(csv.DictReader(queue.open(encoding="utf-8")))
    assert {row["review_decision"] for row in rows} == {"accept"}


def test_negative_pre_review_audit_rejects_positive_hash_collision(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path)
    config = config_for(tmp_path, bundle)
    reassemble_and_extract_detector_archives(config, overwrite=False)
    source = config.quarantine_root / "extracted/train/images/train.jpg"
    collision = config.negative_root / "collision.jpg"
    collision.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="collide"):
        audit_approved_negatives_against_teammate(config)
