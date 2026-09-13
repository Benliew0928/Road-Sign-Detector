from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from roadsign_assist.datasets.recovery_proposals import propose_normalized_batch
from roadsign_assist.inference.models import (
    BoundingBoxModel,
    ClassificationModel,
    DetectionModel,
)


class FakeDetector:
    def __init__(self, name: str, offset: float) -> None:
        self._name = name
        self.offset = offset

    @property
    def name(self) -> str:
        return self._name

    def detect(self, image: np.ndarray) -> list[DetectionModel]:
        del image
        return [
            DetectionModel(
                detection_id=self.name,
                bbox=BoundingBoxModel(
                    x1=40 + self.offset,
                    y1=30 + self.offset,
                    x2=120 + self.offset,
                    y2=110 + self.offset,
                ),
                confidence=0.8,
                detector=self.name,
            )
        ]


class FakeClassifier:
    @property
    def name(self) -> str:
        return "fake-classifier"

    def classify(self, crop: np.ndarray) -> ClassificationModel:
        assert crop.size
        return ClassificationModel(
            semantic_sign_id="turn_left",
            confidence=0.8,
            accepted=True,
            model_name=self.name,
            top_k=[("turn_left", 0.8), ("turn_right", 0.15)],
            unknown_score=0.2,
            embedding_distance=0.1,
            nearest_prototype="turn_left",
        )


def test_proposals_fuse_detectors_and_export_review_formats(tmp_path: Path) -> None:
    image = np.full((180, 240, 3), 100, dtype=np.uint8)
    image_path = tmp_path / "normalized.jpg"
    assert cv2.imwrite(str(image_path), image)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    progress = tmp_path / "normalization.jsonl"
    row = {
        "normalization_batch_id": "normalization",
        "processed_at": "2026-09-04T00:00:00Z",
        "object_id": "object-1",
        "source_id": "source",
        "source_batch_id": "download",
        "group_id": "group-1",
        "raw_path": str(image_path),
        "raw_sha256": image_sha256,
        "raw_bytes": image_path.stat().st_size,
        "normalized_path": str(image_path),
        "normalized_sha256": image_sha256,
        "normalized_bytes": image_path.stat().st_size,
        "width": 240,
        "height": 180,
        "mime_type": "image/jpeg",
        "perceptual_hash": "0" * 16,
        "embedding": [1.0, 0.0],
        "triage_state": "normalized_candidate",
    }
    progress.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = tmp_path / "normalization.json"
    manifest.write_text(
        json.dumps({"outputs": {"progress_jsonl": str(progress)}}), encoding="utf-8"
    )

    report = propose_normalized_batch(
        manifest,
        proposal_batch_id="proposals-v1",
        acquisition_root=tmp_path / "archive",
        audit_root=tmp_path / "audit",
        detectors=[FakeDetector("one", 0), FakeDetector("two", 2)],
        classifier=FakeClassifier(),
    )

    assert report["exit_condition"]["passed"] is True
    assert report["proposal_count"] == 1
    assert report["embedding_distance_available"] == 1
    assert Path(report["outputs"]["cvat_coco"]).is_file()
    assert Path(report["outputs"]["canonical_intake_proposals"]).is_file()


def test_proposals_dry_run_reports_model_and_candidate_state(tmp_path: Path) -> None:
    progress = tmp_path / "normalization.jsonl"
    progress.write_text("", encoding="utf-8")
    manifest = tmp_path / "normalization.json"
    manifest.write_text(
        json.dumps({"outputs": {"progress_jsonl": str(progress)}}), encoding="utf-8"
    )
    result = propose_normalized_batch(
        manifest,
        proposal_batch_id="dry-run",
        acquisition_root=tmp_path / "archive",
        audit_root=tmp_path / "audit",
        dry_run=True,
    )
    assert result["candidate_images"] == 0
    assert not (tmp_path / "archive").exists()
