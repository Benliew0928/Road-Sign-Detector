from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from roadsign_assist.datasets.recovery_approval import has_training_approval, load_training_approval
from roadsign_assist.datasets.recovery_review import validate_promotion_readiness
from roadsign_assist.paths import project_path

POLICY = "configs/recovery/owner_training_approval_20260905.json"


def _row() -> dict[str, str]:
    return dict(
        split="train",
        approval_policy_id="owner_training_acceptance_20260905_v1",
        approval_basis="owner_blanket_acceptance",
        primary_reviewer="benli",
        secondary_reviewer="",
        source_id="fixture",
        review_requirement="full_review",
        promotion_outcome="accept_training",
    )


def test_approval_requires_explicit_matching_authority() -> None:
    row = _row()
    policy = load_training_approval(POLICY)
    assert has_training_approval(row, policy)
    assert not has_training_approval(row, None)
    for update in (
        {"split": "validation"},
        {"split": "phase_e2"},
        {"approval_policy_id": "wrong"},
        {"primary_reviewer": "someone_else"},
        {"secondary_reviewer": "benli"},
        {"quality_gate_status": "content_conflict"},
    ):
        assert not has_training_approval(row | update, policy)


def test_policy_cannot_expand_to_evaluation(tmp_path: Path) -> None:
    policy = json.loads(project_path(POLICY).read_text(encoding="utf-8"))
    policy["allowed_splits"].append("validation")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation cannot be waived"):
        load_training_approval(path)


def test_review_validator_waiver_is_opt_in_and_training_only(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    sources = tmp_path / "sources.csv"
    sources.write_text(
        "source_id,source_review_outcome,permitted_use,reviewer_id\nfixture,approved,internal_training,benli\n",
        encoding="utf-8",
    )
    for outcome, allowed in (
        ("accept_training", True),
        ("accept_validation", False),
        ("accept_internal_test", False),
    ):
        row = _row() | {"promotion_outcome": outcome}
        with review.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        assert not validate_promotion_readiness(review, sources)["promotion_allowed"]
        result = validate_promotion_readiness(review, sources, training_approval_policy=POLICY)
        assert result["promotion_allowed"] is allowed
