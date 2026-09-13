"""Explicit owner-authorized training approval; never an evaluation waiver."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from roadsign_assist.paths import project_path


def load_training_approval(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    raw: object = json.loads(project_path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Invalid owner training approval policy; expected an object")
    value = cast(dict[str, object], raw)
    required = {
        "schema_version": "1.0",
        "scope": "current_and_future_training_candidates",
        "allowed_splits": ["train"],
        "approval_basis": "owner_blanket_acceptance",
        "waive_individual_training_review": True,
        "waive_secondary_training_review": True,
        "preserve_quality_exclusions": True,
        "preserve_evaluation_requirements": True,
        "preserve_collection_floors": True,
        "permitted_use": "internal_academic_only_no_redistribution",
    }
    valid_strings = all(
        isinstance(item, str) and bool(item.strip())
        for item in (value.get(key) for key in ("policy_id", "owner_id", "authority", "authorized_on"))
    )
    if not valid_strings or any(
        value.get(key) != expected or (isinstance(expected, bool) and type(value.get(key)) is not bool)
        for key, expected in required.items()
    ):
        raise ValueError("Invalid owner training approval policy; evaluation cannot be waived")
    return value


def has_training_approval(row: Mapping[str, object], policy: Mapping[str, object] | None) -> bool:
    """A row must name the authority and retain truthful reviewer attribution."""
    return bool(
        policy
        and row.get("split") == "train"
        and row.get("approval_policy_id") == policy["policy_id"]
        and row.get("approval_basis") == policy["approval_basis"]
        and row.get("primary_reviewer") == policy["owner_id"]
        and not row.get("secondary_reviewer")
        and row.get("quality_gate_status")
        not in {"fail", "content_conflict", "rejected", "hold_visible_sign_content_conflict"}
    )
