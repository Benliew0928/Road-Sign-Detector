from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from roadsign_assist.datasets.recovery_review import (
    _real_queue,
    build_recovery_review_package,
    validate_promotion_readiness,
)


def test_failed_or_unclassified_scan_never_becomes_a_negative(tmp_path: Path) -> None:
    progress = tmp_path / "states.jsonl"
    rows = [
        {"object_id": "failed", "image_state": "inference_failure", "proposals": []},
        {"object_id": "unknown", "image_state": "", "proposals": []},
        {
            "object_id": "error",
            "image_state": "candidate_no_sign",
            "inference_error": "failed",
            "proposals": [],
        },
        {"object_id": "clear", "image_state": "candidate_no_sign", "proposals": []},
    ]
    progress.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    queue = _real_queue({"outputs": {"progress_jsonl": str(progress)}})
    assert [row["object_id"] for row in queue] == ["clear"]
    assert queue[0]["item_kind"] == "real_no_sign_proposal"


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (160, 120), (80, 100, 120)).save(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    proposal_progress = tmp_path / "proposals.jsonl"
    proposal_progress.write_text(
        json.dumps(
            {
                "proposal_batch_id": "proposals",
                "processed_at": "2026-09-04T00:00:00Z",
                "object_id": "object-1",
                "source_id": "source",
                "group_id": "group-1",
                "image_path": str(image),
                "image_sha256": digest,
                "image_width": 160,
                "image_height": 120,
                "image_state": "proposals_generated",
                "proposals": [
                    {
                        "proposal_id": "proposal-1",
                        "bbox_xyxy": [20, 20, 80, 80],
                        "semantic_sign_id": "stop",
                        "candidate_state": "candidate_positive_agreement",
                        "detector_confidence": 0.8,
                        "classifier_confidence": 0.9,
                        "priority_score": 90,
                        "licence_status": "unreviewed",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps(
            {
                "proposal_batch_id": "proposals",
                "outputs": {"progress_jsonl": str(proposal_progress)},
                "exit_condition": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    synthetic_progress = tmp_path / "synthetic.jsonl"
    synthetic_progress.write_text(
        json.dumps(
            {
                "sample_id": "synthetic-1",
                "instance_id": "synthetic-1:0",
                "expected_kind": "sign",
                "semantic_sign_id": "stop",
                "image_path": str(image),
                "image_sha256": digest,
                "bbox_x1": 20,
                "bbox_y1": 20,
                "bbox_x2": 80,
                "bbox_y2": 80,
                "presentation_material": "screen",
                "scene_composition": "single_sign",
                "presentation_distance": "near",
                "related_capture_group_id": "synthetic-group",
                "licence_status": "unreviewed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    synthetic = tmp_path / "synthetic.json"
    synthetic.write_text(
        json.dumps(
            {
                "batch_id": "synthetic",
                "outputs": {"manifest_jsonl": str(synthetic_progress)},
                "exit_condition": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({"exit_condition": {"passed": True}}), encoding="utf-8")
    return proposal, normalization, synthetic, image


def test_review_package_is_ready_but_promotion_is_blocked(tmp_path: Path) -> None:
    proposal, normalization, synthetic, _image = _artifacts(tmp_path)
    root = tmp_path / "archive"
    source_ledger = root / "manifests" / "source_events.jsonl"
    source_ledger.parent.mkdir(parents=True)
    source_ledger.write_text(
        json.dumps(
            {
                "source_id": "source",
                "source_name": "Source",
                "publisher": "Publisher",
                "owner": "Owner",
                "source_url": "https://example.test",
                "licence_status": "unreviewed",
                "state": "success",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_recovery_review_package(
        proposal,
        normalization,
        synthetic,
        review_batch_id="review-v1",
        acquisition_root=root,
        audit_root=tmp_path / "audit",
        contact_sheet_limit=2,
    )
    assert report["exit_condition"]["package_ready"] is True
    assert report["exit_condition"]["promotion_complete"] is False
    readiness = validate_promotion_readiness(
        report["outputs"]["review_decisions"], report["outputs"]["source_decisions"]
    )
    assert readiness["promotion_allowed"] is False
    assert readiness["error_counts"]["review_decision_pending"] == 2


def test_promotion_requires_independent_secondary_reviewer(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "review_requirement",
                "promotion_outcome",
                "primary_reviewer",
                "secondary_reviewer",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "source",
                "review_requirement": "full_review",
                "promotion_outcome": "accept_training",
                "primary_reviewer": "person-a",
                "secondary_reviewer": "person-a",
            }
        )
    sources = tmp_path / "sources.csv"
    with sources.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "source_review_outcome", "permitted_use", "reviewer_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "source",
                "source_review_outcome": "approved",
                "permitted_use": "internal_training",
                "reviewer_id": "person-a",
            }
        )
    readiness = validate_promotion_readiness(review, sources)
    assert readiness["promotion_allowed"] is False
    assert readiness["error_counts"]["reviewers_not_independent"] == 1
