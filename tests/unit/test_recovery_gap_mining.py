from __future__ import annotations

import json
from pathlib import Path

from roadsign_assist.datasets.recovery_gap_mining import mine_recovery_gaps


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    progress = tmp_path / "proposals.jsonl"
    progress.write_text(
        json.dumps(
            {
                "proposal_batch_id": "proposals",
                "processed_at": "2026-09-04T00:00:00Z",
                "object_id": "object-1",
                "source_id": "source",
                "group_id": "group-1",
                "image_path": "image.jpg",
                "image_sha256": "a" * 64,
                "image_width": 640,
                "image_height": 480,
                "image_state": "proposals_generated",
                "proposals": [
                    {
                        "semantic_sign_id": "stop",
                        "candidate_state": "candidate_positive_agreement",
                        "size_bucket": "small",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps({"outputs": {"progress_jsonl": str(progress)}}), encoding="utf-8"
    )
    normalization = tmp_path / "normalization.json"
    normalization.write_text(
        json.dumps(
            {
                "candidate_objects": 2,
                "states": {"normalized_candidate": 1, "rejected_auto": 1},
                "road_context_score_at_least_0_35": 1,
                "privacy_flags": {"possible_plate": 1},
            }
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["stop", "give_way"]), encoding="utf-8")
    return proposal, normalization, labels


def test_gap_mining_separates_reviewed_and_unreviewed_counts(tmp_path: Path) -> None:
    proposal, normalization, labels = _fixtures(tmp_path)
    report = mine_recovery_gaps(
        proposal,
        normalization,
        gap_batch_id="gaps-v1",
        acquisition_root=tmp_path / "archive",
        audit_root=tmp_path / "audit",
        local_inventory=tmp_path / "missing.parquet",
        labels_path=labels,
    )

    assert report["exit_condition"]["passed"] is True
    assert report["coverage"]["proposal_counts_are_gate_credit"] is False
    stop = next(
        row for row in report["coverage"]["classes"] if row["semantic_sign_id"] == "stop"
    )
    assert stop["reviewed_unique_images"] == 0
    assert stop["unreviewed_proposals"] == 1
    assert report["next_queries"][0]["safety_priority"] is True


def test_gap_mining_dry_run_has_no_outputs(tmp_path: Path) -> None:
    proposal, normalization, labels = _fixtures(tmp_path)
    result = mine_recovery_gaps(
        proposal,
        normalization,
        gap_batch_id="dry-run",
        acquisition_root=tmp_path / "archive",
        audit_root=tmp_path / "audit",
        labels_path=labels,
        dry_run=True,
    )
    assert result["proposal_images"] == 1
    assert not (tmp_path / "archive").exists()
