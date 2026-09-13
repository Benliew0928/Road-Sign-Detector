"""Phase B7 candidate freeze and residual Gate-B reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from roadsign_assist.datasets.recovery_acquisition import (
    DEFAULT_ACQUISITION_ROOT,
    DEFAULT_AUDIT_ROOT,
)
from roadsign_assist.datasets.recovery_review import validate_promotion_readiness
from roadsign_assist.paths import project_path

FINALIZER_VERSION = "recovery_candidate_freeze_v1"


def _now() -> datetime:
    return datetime.now(UTC)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _load_json(path: Path) -> Mapping[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value)


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                rows.append(cast(Mapping[str, object], value))
    return rows


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)) and value != "":
        return int(value)
    return 0


def _list_length(value: object) -> int:
    return len(cast(list[object], value)) if isinstance(value, list) else 0


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _known_prior_phashes(manifest_root: Path) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for path in sorted(manifest_root.rglob("*.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                hash_field = next(
                    (field for field in ("perceptual_hash", "phash") if field in fields), None
                )
                if hash_field is None:
                    continue
                for row_number, row in enumerate(reader, start=2):
                    value = row.get(hash_field, "").strip().lower()
                    if len(value) == 16 and all(
                        character in "0123456789abcdef" for character in value
                    ):
                        values.setdefault(value, set()).add(f"{path}:{row_number}")
        except (OSError, csv.Error, UnicodeError):
            continue
    return values


def _snapshot(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _hash(path) if path.is_file() else "",
    }


def _markdown(report: Mapping[str, object]) -> str:
    missing = _mapping(report.get("exact_missing_counts"))
    phases = _mapping(report.get("phase_results"))
    lines = [
        "# Autonomous Recovery Candidate Freeze",
        "",
        f"- Candidate pipeline complete: `{str(report.get('candidate_pipeline_complete')).upper()}`",
        f"- Gate B passed: `{str(report.get('gate_b_passed')).upper()}`",
        f"- Promotion allowed: `{str(report.get('promotion_allowed')).upper()}`",
        "- Public-source proxy counts as Phase E2: `FALSE`",
        "",
        "## Phase results",
        "",
        "| Phase | Result |",
        "| --- | --- |",
    ]
    lines.extend(f"| `{phase}` | `{value}` |" for phase, value in phases.items())
    lines.extend(
        [
            "",
            "## Exact remaining evidence debt",
            "",
            "| Requirement | Missing |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| `{name}` | {value} |" for name, value in missing.items())
    lines.extend(
        [
            "",
            "Candidate totals are not Gate-B credit. Review/source approval status is reported above; independent physical Phase E2 evidence remains a separate requirement.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_recovery_cycle(
    *,
    freeze_id: str,
    normalization_report: str | Path,
    proposal_report: str | Path,
    gap_report: str | Path,
    synthetic_report: str | Path,
    review_report: str | Path,
    gate_audit: str | Path,
    canonical_intake: str | Path = "data/manifests/recovery_phase_b_intake_v1.csv",
    manifest_root: str | Path = "data/manifests",
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    training_approval_policy: str | Path | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Freeze autonomous candidate artifacts and report every remaining Gate-B blocker."""
    root = project_path(acquisition_root)
    output = project_path(audit_root) / "final" / f"{freeze_id}.json"
    markdown = output.with_suffix(".md")
    freeze_manifest = root / "manifests" / "freezes" / f"{freeze_id}.json"
    if output.exists():
        if resume:
            return dict(_load_json(output)) | {"resume_reused": True}
        raise FileExistsError(f"Refusing to overwrite final report: {output}")
    normal = _load_json(project_path(normalization_report))
    proposal = _load_json(project_path(proposal_report))
    gaps = _load_json(project_path(gap_report))
    synthetic = _load_json(project_path(synthetic_report))
    review = _load_json(project_path(review_report))
    gate = _load_json(project_path(gate_audit))
    normal_outputs = _mapping(normal.get("outputs"))
    proposal_outputs = _mapping(proposal.get("outputs"))
    synthetic_outputs = _mapping(synthetic.get("outputs"))
    review_outputs = _mapping(review.get("outputs"))
    normal_rows = _load_jsonl(project_path(str(normal_outputs.get("progress_jsonl", ""))))
    prior_phashes = _known_prior_phashes(project_path(manifest_root))
    near_matches: list[dict[str, object]] = []
    for row in normal_rows:
        if row.get("triage_state") != "normalized_candidate":
            continue
        value = str(row.get("perceptual_hash", ""))
        if len(value) != 16:
            continue
        nearest = sorted(
            (
                (_hamming(value, prior), prior)
                for prior in prior_phashes
                if _hamming(value, prior) <= 4
            ),
            key=lambda item: (item[0], item[1]),
        )
        if nearest:
            distance, prior = nearest[0]
            near_matches.append(
                {
                    "object_id": row.get("object_id", ""),
                    "perceptual_hash": value,
                    "nearest_prior_hash": prior,
                    "hamming_distance": distance,
                    "prior_references": sorted(prior_phashes[prior])[:10],
                }
            )
    checks = _mapping(gate.get("checks"))
    gate_counts = _mapping(gate.get("counts"))
    gate_gaps = _mapping(gate.get("coverage_gaps"))
    safety_gaps = _mapping(gate_gaps.get("safety_evaluation_groups"))
    screen_classes = gate_gaps.get("screen_classes", [])
    print_classes = gate_gaps.get("print_classes", [])
    screen_negative = gate_gaps.get("screen_negative_kinds", [])
    review_items = _integer(review.get("total_review_items"))
    source_rows = _integer(review.get("source_rows"))
    readiness = None
    if training_approval_policy is not None:
        readiness = validate_promotion_readiness(
            str(review_outputs["review_decisions"]),
            str(review_outputs["source_decisions"]),
            training_approval_policy=training_approval_policy,
        )
        review_items = _integer(
            _mapping(readiness.get("error_counts")).get("review_decision_pending")
        )
        source_rows = _integer(readiness.get("source_rows")) - _integer(
            readiness.get("approved_sources")
        )
    missing = {
        "canonical_rows": 1 if _integer(gate_counts.get("rows")) == 0 else 0,
        "full_context_images_to_15000": max(
            0, 15_000 - _integer(gate_counts.get("road_development_images"))
        ),
        "sign_instances_to_20000": max(
            0, 20_000 - _integer(gate_counts.get("road_development_sign_instances"))
        ),
        "reviewed_negative_images_minimum_at_15000_images": max(
            0, 3_000 - _integer(gate_counts.get("road_development_accepted_negative_images"))
        ),
        "independent_sessions_to_30": max(
            0, 30 - _integer(gate_counts.get("road_development_independent_sessions"))
        ),
        "camera_pipelines_to_3": max(
            0, 3 - _integer(gate_counts.get("road_development_camera_pipelines"))
        ),
        "small_instances_to_3000": max(
            0, 3_000 - _integer(gate_counts.get("road_development_small_instances_at_most_1pct"))
        ),
        "very_small_instances_to_1000": max(
            0,
            1_000
            - _integer(gate_counts.get("road_development_very_small_instances_at_most_0_1pct")),
        ),
        "screen_classes_missing": _list_length(screen_classes),
        "print_classes_missing": _list_length(print_classes),
        "screen_negative_kinds_missing": _list_length(screen_negative),
        "safety_class_eval_groups_missing": sum(
            max(0, 30 - _integer(value)) for value in safety_gaps.values()
        ),
        "pending_review_items": review_items,
        "pending_source_decisions": source_rows,
        "independent_phase_e2_sets": 1 if not bool(checks.get("phase_e2_present")) else 0,
    }
    snapshots = [
        _snapshot(project_path(canonical_intake)),
        _snapshot(project_path(str(normal_outputs.get("manifest_parquet", "")))),
        _snapshot(project_path(str(proposal_outputs.get("proposal_parquet", "")))),
        _snapshot(project_path(str(synthetic_outputs.get("manifest_parquet", "")))),
        _snapshot(project_path(str(review_outputs.get("review_decisions", "")))),
        _snapshot(project_path(str(review_outputs.get("source_decisions", "")))),
        _snapshot(project_path(gate_audit)),
    ]
    phase_results = {
        "B0_inventory": "pass",
        "B1_first_bounded_acquisition": "partial_source_batch_complete",
        "B2_normalization": "pass"
        if _mapping(normal.get("exit_condition")).get("passed")
        else "fail",
        "B3_proposals": "pass"
        if _mapping(proposal.get("exit_condition")).get("passed")
        else "fail",
        "B4_gap_mining": "pass" if _mapping(gaps.get("exit_condition")).get("passed") else "fail",
        "B5_synthetic": "pass"
        if _mapping(synthetic.get("exit_condition")).get("passed")
        else "fail",
        "B6_review_package": "ready_blocked_on_human_decisions"
        if _mapping(review.get("exit_condition")).get("package_ready")
        else "fail",
        "B7_gate_audit": "blocked_reported" if not gate.get("gate_b_passed") else "pass",
    }
    candidate_complete = (
        all(
            phase_results[key] == "pass"
            for key in ("B2_normalization", "B3_proposals", "B4_gap_mining", "B5_synthetic")
        )
        and phase_results["B6_review_package"] == "ready_blocked_on_human_decisions"
    )
    if readiness and readiness["promotion_allowed"]:
        phase_results["B6_review_package"] = "owner_policy_accepted"
        candidate_complete = all(
            phase_results[key] == "pass"
            for key in ("B2_normalization", "B3_proposals", "B4_gap_mining", "B5_synthetic")
        )
    report: dict[str, object] = {
        "freeze_id": freeze_id,
        "generated_at": _now().isoformat(),
        "finalizer_version": FINALIZER_VERSION,
        "candidate_pipeline_complete": candidate_complete,
        "gate_b_passed": bool(gate.get("gate_b_passed")),
        "promotion_allowed": bool(readiness and readiness["promotion_allowed"]),
        "promotion_readiness": readiness,
        "training_allowed": False,
        "phase_results": phase_results,
        "candidate_counts": {
            "raw_downloaded_objects": normal.get("candidate_objects", 0),
            "unique_normalized_candidates": _mapping(normal.get("states")).get(
                "normalized_candidate", 0
            ),
            "automated_proposals": proposal.get("proposal_count", 0),
            "candidate_no_sign_images": _mapping(proposal.get("image_states")).get(
                "candidate_no_sign", 0
            ),
            "synthetic_images": synthetic.get("generated_rows", 0),
            "synthetic_images_per_class": synthetic.get("per_class_floor", 0),
        },
        "canonical_gate_counts": dict(gate_counts),
        "exact_missing_counts": missing,
        "gate_checks": dict(checks),
        "leakage_audit": {
            "spent_hash_rejections": _mapping(normal.get("rejections")).get(
                "spent_evaluation_denylisted", 0
            ),
            "exact_prior_object_rejections": _mapping(normal.get("rejections")).get(
                "exact_duplicate_of_existing_object", 0
            ),
            "within_batch_near_duplicate_rejections": _mapping(normal.get("rejections")).get(
                "perceptual_near_duplicate", 0
            ),
            "known_prior_perceptual_hashes": len(prior_phashes),
            "candidate_near_matches_to_known_prior_hashes": len(near_matches),
            "candidate_near_match_details": near_matches,
            "coverage_note": "Near-match comparison covers perceptual hashes recorded in prior manifests; exact SHA comparison covers the full local B0 inventory and spent denylist.",
        },
        "split_freeze": {
            "canonical_intake_frozen_rows": gate_counts.get("rows", 0),
            "candidate_groups_are_not_release_splits": True,
            "accepted_split_assignment": "train_only_owner_policy"
            if readiness and readiness["promotion_allowed"]
            else "blocked_until_promotion",
            "synthetic_group_split_crossings": synthetic.get("group_split_crossings", 0),
        },
        "public_source_evaluation_proxy": {
            "status": "train_only_no_independent_evaluation_credit"
            if readiness and readiness["promotion_allowed"]
            else "not_frozen_no_reviewed_accepted_rows",
            "phase_e2_credit": False,
            "reason": "Public KartaView candidates are neither target-device captures nor independently locked Phase E2 evidence.",
        },
        "immutable_snapshots": snapshots,
        "remaining_failures_reported": True,
        "resume_reused": False,
        "outputs": {
            "json": str(output),
            "markdown": str(markdown),
            "freeze_manifest": str(freeze_manifest),
        },
    }
    freeze_payload = {
        "freeze_id": freeze_id,
        "generated_at": report["generated_at"],
        "candidate_only": True,
        "gate_b_passed": report["gate_b_passed"],
        "training_allowed": False,
        "snapshots": snapshots,
    }
    for path, payload in ((output, report), (freeze_manifest, freeze_payload)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    markdown.write_text(_markdown(report), encoding="utf-8")
    return report


__all__ = ["FINALIZER_VERSION", "finalize_recovery_cycle"]
