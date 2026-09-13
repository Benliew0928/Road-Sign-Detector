from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from roadsign_assist.datasets.recovery_finalize import finalize_recovery_cycle


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_finalize_freezes_candidates_without_granting_gate_credit(tmp_path: Path) -> None:
    normal_rows = tmp_path / "normal.jsonl"
    normal_rows.write_text(
        json.dumps(
            {
                "object_id": "candidate-1",
                "triage_state": "normalized_candidate",
                "perceptual_hash": "0000000000000001",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    normal_parquet = tmp_path / "normal.parquet"
    proposal_parquet = tmp_path / "proposal.parquet"
    synthetic_parquet = tmp_path / "synthetic.parquet"
    review_decisions = tmp_path / "review.csv"
    source_decisions = tmp_path / "sources.csv"
    for path in (
        normal_parquet,
        proposal_parquet,
        synthetic_parquet,
        review_decisions,
        source_decisions,
    ):
        path.write_bytes(b"frozen")

    normal = _write_json(
        tmp_path / "normal.json",
        {
            "candidate_objects": 10,
            "states": {"normalized_candidate": 8},
            "rejections": {"spent_evaluation_denylisted": 1},
            "outputs": {
                "progress_jsonl": str(normal_rows),
                "manifest_parquet": str(normal_parquet),
            },
            "exit_condition": {"passed": True},
        },
    )
    proposal = _write_json(
        tmp_path / "proposal.json",
        {
            "proposal_count": 6,
            "image_states": {"candidate_no_sign": 2},
            "outputs": {"proposal_parquet": str(proposal_parquet)},
            "exit_condition": {"passed": True},
        },
    )
    gaps = _write_json(tmp_path / "gaps.json", {"exit_condition": {"passed": True}})
    synthetic = _write_json(
        tmp_path / "synthetic.json",
        {
            "generated_rows": 100,
            "per_class_floor": 1,
            "group_split_crossings": 0,
            "outputs": {"manifest_parquet": str(synthetic_parquet)},
            "exit_condition": {"passed": True},
        },
    )
    review = _write_json(
        tmp_path / "review.json",
        {
            "total_review_items": 108,
            "source_rows": 2,
            "outputs": {
                "review_decisions": str(review_decisions),
                "source_decisions": str(source_decisions),
            },
            "exit_condition": {"package_ready": True, "promotion_complete": False},
        },
    )
    intake = tmp_path / "intake.csv"
    intake.write_text("instance_id\n", encoding="utf-8")
    prior = tmp_path / "manifests" / "prior.csv"
    prior.parent.mkdir()
    prior.write_text("phash\n0000000000000000\n", encoding="utf-8")
    gate = _write_json(
        tmp_path / "gate.json",
        {
            "gate_b_passed": False,
            "counts": {"rows": 0, "road_development_images": 0},
            "checks": {"phase_e2_present": False},
            "coverage_gaps": {
                "screen_classes": ["stop"],
                "print_classes": ["stop"],
                "screen_negative_kinds": ["icon"],
                "safety_evaluation_groups": {"stop": 0},
            },
        },
    )

    report = finalize_recovery_cycle(
        freeze_id="cycle-v1",
        normalization_report=normal,
        proposal_report=proposal,
        gap_report=gaps,
        synthetic_report=synthetic,
        review_report=review,
        gate_audit=gate,
        canonical_intake=intake,
        manifest_root=prior.parent,
        acquisition_root=tmp_path / "archive",
        audit_root=tmp_path / "audit",
    )

    assert report["candidate_pipeline_complete"] is True
    assert report["gate_b_passed"] is False
    assert report["promotion_allowed"] is False
    assert report["training_allowed"] is False
    missing = cast(Mapping[str, object], report["exact_missing_counts"])
    leakage = cast(Mapping[str, object], report["leakage_audit"])
    snapshots = cast(list[Mapping[str, object]], report["immutable_snapshots"])
    assert missing["pending_review_items"] == 108
    assert leakage["candidate_near_matches_to_known_prior_hashes"] == 1
    assert all(snapshot["sha256"] for snapshot in snapshots)
    resumed = finalize_recovery_cycle(
        freeze_id="cycle-v1",
        normalization_report=normal,
        proposal_report=proposal,
        gap_report=gaps,
        synthetic_report=synthetic,
        review_report=review,
        gate_audit=gate,
        canonical_intake=intake,
        manifest_root=prior.parent,
        acquisition_root=tmp_path / "archive",
        audit_root=tmp_path / "audit",
        resume=True,
    )
    assert resumed["resume_reused"] is True
    review_decisions.write_text(
        "source_id,promotion_outcome,review_requirement,primary_reviewer,secondary_reviewer,split,approval_policy_id,approval_basis\n"
        "fixture,accept_training,full_review,benli,,train,owner_training_acceptance_20260905_v1,owner_blanket_acceptance\n",
        encoding="utf-8",
    )
    source_decisions.write_text(
        "source_id,source_review_outcome,permitted_use,reviewer_id\nfixture,approved,internal_academic,benli\n",
        encoding="utf-8",
    )
    accepted = finalize_recovery_cycle(
        freeze_id="cycle-owner",
        normalization_report=normal,
        proposal_report=proposal,
        gap_report=gaps,
        synthetic_report=synthetic,
        review_report=review,
        gate_audit=gate,
        canonical_intake=intake,
        manifest_root=prior.parent,
        acquisition_root=tmp_path / "archive",
        audit_root=tmp_path / "audit",
        training_approval_policy="configs/recovery/owner_training_approval_20260905.json",
    )
    assert accepted["promotion_allowed"] is True
    assert accepted["training_allowed"] is False
    assert accepted["exact_missing_counts"]["pending_review_items"] == 0
    assert accepted["exact_missing_counts"]["pending_source_decisions"] == 0
