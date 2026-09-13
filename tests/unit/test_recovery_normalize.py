from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roadsign_assist.datasets.recovery_normalize import normalize_quarantine_batch


def _write_image(path: Path, value: int) -> str:
    image = np.full((256, 384, 3), value, dtype=np.uint8)
    cv2.line(image, (0, 220), (383, 150), (255 - value,) * 3, 8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event(
    path: Path, object_id: str, image: Path, sha256: str, batch_id: str = "download"
) -> None:
    row = {
        "event_id": object_id,
        "observed_at": "2026-09-04T00:00:00Z",
        "object_id": object_id,
        "source_id": "source",
        "batch_id": batch_id,
        "group_id": f"group-{object_id}",
        "sequence_id": "",
        "source_url": "https://example.test/image.jpg",
        "filename": image.name,
        "sha256": sha256,
        "byte_size": image.stat().st_size,
        "mime_type": "image/jpeg",
        "raw_path": str(image),
        "collection_state": "quarantined",
        "licence_status": "unreviewed",
        "permitted_use": "none_pending_review",
        "review_status": "pending",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def test_normalize_rejects_duplicate_and_is_resumable(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    first = tmp_path / "one.jpg"
    second = tmp_path / "two.jpg"
    sha256 = _write_image(first, 80)
    second.write_bytes(first.read_bytes())
    ledger = root / "manifests" / "object_events.jsonl"
    _event(ledger, "one", first, sha256)
    _event(ledger, "two", second, sha256)

    result = normalize_quarantine_batch(
        source_id="source",
        source_batch_id="download",
        normalization_batch_id="normalize-v1",
        acquisition_root=root,
        audit_root=tmp_path / "audit",
        spent_denylist=tmp_path / "missing.csv",
        minimum_free_gib=0,
        maximum_acquisition_gib=1,
        privacy_scan=False,
    )

    assert result["exit_condition"]["passed"] is True
    assert result["states"] == {"normalized_candidate": 1, "rejected_auto": 1}
    assert result["rejections"] == {"perceptual_near_duplicate": 1}
    resumed = normalize_quarantine_batch(
        source_id="source",
        source_batch_id="download",
        normalization_batch_id="normalize-v1",
        acquisition_root=root,
        audit_root=tmp_path / "audit",
        minimum_free_gib=0,
        maximum_acquisition_gib=1,
        resume=True,
    )
    assert resumed["resume_reused"] is True


def test_normalize_dry_run_does_not_write_outputs(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    image = tmp_path / "one.jpg"
    sha256 = _write_image(image, 120)
    _event(root / "manifests" / "object_events.jsonl", "one", image, sha256)

    result = normalize_quarantine_batch(
        source_id="source",
        source_batch_id="download",
        normalization_batch_id="dry-run",
        acquisition_root=root,
        audit_root=tmp_path / "audit",
        minimum_free_gib=0,
        maximum_acquisition_gib=1,
        privacy_scan=False,
        dry_run=True,
    )

    assert result["candidate_objects"] == 1
    assert not (root / "manifests" / "normalization").exists()


def test_normalize_accepts_official_ppm_road_image(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    image = tmp_path / "00000.ppm"
    pixels = np.full((256, 384, 3), 90, dtype=np.uint8)
    pixels[160:220, :] = 170
    Image.fromarray(pixels).save(image)
    sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    _event(root / "manifests" / "object_events.jsonl", "ppm-one", image, sha256)

    result = normalize_quarantine_batch(
        source_id="source",
        source_batch_id="download",
        normalization_batch_id="normalize-ppm",
        acquisition_root=root,
        audit_root=tmp_path / "audit",
        spent_denylist=tmp_path / "missing.csv",
        minimum_free_gib=0,
        maximum_acquisition_gib=1,
        privacy_scan=False,
    )

    assert result["states"] == {"normalized_candidate": 1}
    assert result["unique_normalized_sha256"] == 1


def test_normalize_rejects_near_duplicates_across_completed_batches(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    first = tmp_path / "one.jpg"
    first_sha = _write_image(first, 80)
    ledger = root / "manifests/object_events.jsonl"
    _event(ledger, "one", first, first_sha, "download-1")
    args = dict(
        source_id="source",
        acquisition_root=root,
        audit_root=tmp_path / "audit",
        spent_denylist=tmp_path / "missing.csv",
        minimum_free_gib=0,
        maximum_acquisition_gib=1,
        privacy_scan=False,
    )
    result = normalize_quarantine_batch(
        source_batch_id="download-1", normalization_batch_id="normal-1", **args
    )
    assert result["states"]["normalized_candidate"] == 1
    second = tmp_path / "two.jpg"
    second.write_bytes(first.read_bytes() + b"different-file-same-decoded-pixels")
    _event(ledger, "two", second, hashlib.sha256(second.read_bytes()).hexdigest(), "download-2")
    result = normalize_quarantine_batch(
        source_batch_id="download-2", normalization_batch_id="normal-2", **args
    )
    assert result["states"] == {"rejected_auto": 1}
    assert result["rejections"] == {"perceptual_near_duplicate": 1}
