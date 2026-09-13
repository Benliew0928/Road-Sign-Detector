"""Phase B6 deferred review packaging and fail-closed promotion validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from roadsign_assist.datasets.contact_sheet import ReviewTile, render_contact_sheet
from roadsign_assist.datasets.recovery_acquisition import (
    DEFAULT_ACQUISITION_ROOT,
    DEFAULT_AUDIT_ROOT,
)
from roadsign_assist.paths import project_path

REVIEW_PACKAGE_VERSION = "recovery_review_package_v1"
PROMOTION_OUTCOMES = frozenset(
    {
        "accept_training",
        "accept_validation",
        "accept_internal_test",
        "accept_detector_only",
        "accept_unknown_negative",
        "hold_internal_academic",
        "reject_licence",
        "reject_privacy",
        "reject_label",
        "reject_duplicate",
        "reject_quality",
    }
)
ACCEPT_OUTCOMES = frozenset(
    {
        "accept_training",
        "accept_validation",
        "accept_internal_test",
        "accept_detector_only",
        "accept_unknown_negative",
    }
)
QUEUE_FIELDS = (
    "review_item_id",
    "item_kind",
    "source_id",
    "source_batch_id",
    "object_id",
    "instance_id",
    "group_id",
    "image_path",
    "image_sha256",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "semantic_sign_id",
    "candidate_state",
    "detector_confidence",
    "classifier_confidence",
    "priority_score",
    "review_requirement",
    "licence_status",
    "source_review_required",
    "primary_reviewer",
    "secondary_reviewer",
    "promotion_outcome",
    "review_notes",
)
SOURCE_FIELDS = (
    "source_id",
    "source_name",
    "publisher",
    "owner",
    "source_url",
    "licence_name",
    "licence_url",
    "licence_status_observed",
    "attribution",
    "restrictions_json",
    "redistribution_review_required",
    "latest_state",
    "latest_state_reason",
    "source_review_outcome",
    "permitted_use",
    "redistribution_allowed",
    "reviewer_id",
    "review_notes",
)


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()


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
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                rows.append(cast(Mapping[str, object], value))
    return rows


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer: Any = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _number(value: object) -> float:
    if isinstance(value, (int, float, str)) and value != "":
        return float(value)
    return 0.0


def _integer(value: object) -> int:
    return round(_number(value))


def _real_queue(proposal_report: Mapping[str, object]) -> list[dict[str, object]]:
    outputs = _mapping(proposal_report.get("outputs"))
    rows = _load_jsonl(project_path(str(outputs.get("progress_jsonl", ""))))
    queue: list[dict[str, object]] = []
    for image in rows:
        # A failed scan is not evidence that the image contains no sign.
        if image.get("inference_error") or image.get("image_state") == "inference_failure":
            continue
        object_id = str(image.get("object_id", ""))
        proposals = image.get("proposals", [])
        if isinstance(proposals, list) and proposals:
            for proposal_value in cast(list[object], proposals):
                proposal = _mapping(proposal_value)
                semantic = str(proposal.get("semantic_sign_id", "unknown_sign"))
                state = str(proposal.get("candidate_state", ""))
                priority = _integer(proposal.get("priority_score"))
                requirement = "full_review"
                bbox_value = proposal.get("bbox_xyxy", [])
                bbox = cast(list[object], bbox_value) if isinstance(bbox_value, list) else []
                padded = [_integer(value) for value in bbox[:4]] + [0, 0, 0, 0]
                instance_id = str(proposal.get("proposal_id", ""))
                queue.append(
                    {
                        "review_item_id": _stable_id("review", instance_id),
                        "item_kind": "real_sign_proposal",
                        "source_id": image.get("source_id", ""),
                        "source_batch_id": proposal_report.get("proposal_batch_id", ""),
                        "object_id": object_id,
                        "instance_id": instance_id,
                        "group_id": image.get("group_id", ""),
                        "image_path": image.get("image_path", ""),
                        "image_sha256": image.get("image_sha256", ""),
                        "bbox_x1": padded[0],
                        "bbox_y1": padded[1],
                        "bbox_x2": padded[2],
                        "bbox_y2": padded[3],
                        "semantic_sign_id": semantic,
                        "candidate_state": state,
                        "detector_confidence": proposal.get("detector_confidence", 0),
                        "classifier_confidence": proposal.get("classifier_confidence", 0),
                        "priority_score": priority,
                        "review_requirement": requirement,
                        "licence_status": proposal.get("licence_status", "unreviewed"),
                        "source_review_required": "true",
                        "primary_reviewer": "",
                        "secondary_reviewer": "",
                        "promotion_outcome": "",
                        "review_notes": "",
                    }
                )
        elif image.get("image_state") == "candidate_no_sign":
            instance_id = f"{object_id}:no-sign"
            queue.append(
                {
                    "review_item_id": _stable_id("review", instance_id),
                    "item_kind": "real_no_sign_proposal",
                    "source_id": image.get("source_id", ""),
                    "source_batch_id": proposal_report.get("proposal_batch_id", ""),
                    "object_id": object_id,
                    "instance_id": instance_id,
                    "group_id": image.get("group_id", ""),
                    "image_path": image.get("image_path", ""),
                    "image_sha256": image.get("image_sha256", ""),
                    "bbox_x1": 0,
                    "bbox_y1": 0,
                    "bbox_x2": 0,
                    "bbox_y2": 0,
                    "semantic_sign_id": "",
                    "candidate_state": "candidate_no_sign",
                    "detector_confidence": 0,
                    "classifier_confidence": 0,
                    "priority_score": 75,
                    "review_requirement": "sensitive_detector_missing_sign_audit",
                    "licence_status": image.get("licence_status", "unreviewed"),
                    "source_review_required": "true",
                    "primary_reviewer": "",
                    "secondary_reviewer": "",
                    "promotion_outcome": "",
                    "review_notes": "",
                }
            )
    return queue


def _synthetic_queue(synthetic_report: Mapping[str, object]) -> list[dict[str, object]]:
    outputs = _mapping(synthetic_report.get("outputs"))
    rows = _load_jsonl(project_path(str(outputs.get("manifest_jsonl", ""))))
    first_strata: set[tuple[str, ...]] = set()
    queue: list[dict[str, object]] = []
    for row in rows:
        expected = str(row.get("expected_kind", ""))
        if expected == "sign":
            stratum = (
                str(row.get("semantic_sign_id", "")),
                str(row.get("presentation_material", "")),
                str(row.get("scene_composition", "")),
                str(row.get("presentation_distance", "")),
            )
        else:
            stratum = ("negative", str(row.get("negative_kind", "")))
        selected = stratum not in first_strata
        first_strata.add(stratum)
        bbox = [
            _integer(row.get("bbox_x1")),
            _integer(row.get("bbox_y1")),
            _integer(row.get("bbox_x2")),
            _integer(row.get("bbox_y2")),
        ]
        instance_id = str(row.get("instance_id", ""))
        queue.append(
            {
                "review_item_id": _stable_id("review", instance_id),
                "item_kind": "synthetic_sign" if expected == "sign" else "synthetic_negative",
                "source_id": "deterministic_synthetic",
                "source_batch_id": synthetic_report.get("batch_id", ""),
                "object_id": row.get("sample_id", ""),
                "instance_id": instance_id,
                "group_id": row.get("related_capture_group_id", ""),
                "image_path": row.get("image_path", ""),
                "image_sha256": row.get("image_sha256", ""),
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "semantic_sign_id": row.get("semantic_sign_id", ""),
                "candidate_state": "synthetic_exact_geometry_candidate",
                "detector_confidence": "",
                "classifier_confidence": "",
                "priority_score": 55 if selected else 20,
                "review_requirement": (
                    "stratified_synthetic_qc" if selected else "deferred_until_stratum_qc_passes"
                ),
                "licence_status": row.get("licence_status", "unreviewed"),
                "source_review_required": "true",
                "primary_reviewer": "",
                "secondary_reviewer": "",
                "promotion_outcome": "",
                "review_notes": "",
            }
        )
    return queue


def _source_template(source_ledger: Path) -> list[dict[str, object]]:
    latest: dict[str, Mapping[str, object]] = {}
    for row in _load_jsonl(source_ledger):
        source_id = str(row.get("source_id", ""))
        if source_id:
            latest[source_id] = row
    rows: list[dict[str, object]] = []
    for source_id, source in sorted(latest.items()):
        rows.append(
            {
                "source_id": source_id,
                "source_name": source.get("source_name", ""),
                "publisher": source.get("publisher", ""),
                "owner": source.get("owner", ""),
                "source_url": source.get("source_url", ""),
                "licence_name": source.get("licence_name", ""),
                "licence_url": source.get("licence_url", ""),
                "licence_status_observed": source.get("licence_status", ""),
                "attribution": source.get("attribution", ""),
                "restrictions_json": json.dumps(source.get("restrictions", []), sort_keys=True),
                "redistribution_review_required": source.get(
                    "redistribution_review_required", True
                ),
                "latest_state": source.get("state", ""),
                "latest_state_reason": source.get("state_reason", ""),
                "source_review_outcome": "",
                "permitted_use": "none_pending_review",
                "redistribution_allowed": "",
                "reviewer_id": "",
                "review_notes": "",
            }
        )
    rows.append(
        {
            "source_id": "deterministic_synthetic",
            "source_name": "Deterministic derivatives of reviewed classifier train artwork",
            "publisher": "MiniProject internal",
            "owner": "derived; see per-row source artwork",
            "source_url": "data/manifests/classifier_production_78_v3_20260829.csv",
            "licence_name": "per-source inherited terms",
            "licence_url": "",
            "licence_status_observed": "pending_batch_review",
            "attribution": "per-row inherited source metadata",
            "restrictions_json": json.dumps(["internal_academic_only", "publication_not_approved"]),
            "redistribution_review_required": True,
            "latest_state": "generated_quarantined",
            "latest_state_reason": "",
            "source_review_outcome": "",
            "permitted_use": "none_pending_review",
            "redistribution_allowed": "",
            "reviewer_id": "",
            "review_notes": "",
        }
    )
    return rows


def _contact_sheets(
    package: Path, queue: Sequence[Mapping[str, object]], *, limit: int
) -> tuple[int, list[dict[str, object]]]:
    selected = sorted(
        queue,
        key=lambda row: (-_integer(row.get("priority_score")), str(row.get("review_item_id", ""))),
    )[:limit]
    index: list[dict[str, object]] = []
    count = 0
    for start in range(0, len(selected), 48):
        page = selected[start : start + 48]
        tiles: list[ReviewTile] = []
        page_rows: list[Mapping[str, object]] = []
        for row in page:
            image_path = project_path(str(row.get("image_path", "")))
            if not image_path.is_file():
                continue
            bbox = (
                _integer(row.get("bbox_x1")),
                _integer(row.get("bbox_y1")),
                _integer(row.get("bbox_x2")),
                _integer(row.get("bbox_y2")),
            )
            crop = bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None
            label = f"{row.get('semantic_sign_id') or row.get('candidate_state')} p{row.get('priority_score')}"
            tiles.append(ReviewTile(label=label[:30], image_path=image_path, crop=crop))
            page_rows.append(row)
        if not tiles:
            continue
        sheet = package / "contact_sheets" / f"ranked_{start // 48 + 1:03d}.jpg"
        render_contact_sheet(tiles, sheet, columns=8)
        count += 1
        for position, row in enumerate(page_rows, start=1):
            index.append(
                {
                    "review_item_id": row.get("review_item_id", ""),
                    "sheet": str(sheet),
                    "position": position,
                }
            )
    return count, index


def build_recovery_review_package(
    proposal_report: str | Path,
    normalization_manifest: str | Path,
    synthetic_report: str | Path,
    *,
    review_batch_id: str,
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    contact_sheet_limit: int = 960,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    """Build a ranked review package without making or implying review decisions."""
    root = project_path(acquisition_root)
    proposal = _load_json(project_path(proposal_report))
    normalization = _load_json(project_path(normalization_manifest))
    synthetic = _load_json(project_path(synthetic_report))
    package = root / "review_packages" / review_batch_id
    report_path = root / "manifests" / "reviews" / f"{review_batch_id}.json"
    audit_path = project_path(audit_root) / "reviews" / f"{review_batch_id}.json"
    if report_path.exists():
        if resume:
            return dict(_load_json(report_path)) | {"resume_reused": True}
        raise FileExistsError(f"Refusing to overwrite review package report: {report_path}")
    real_queue = _real_queue(proposal)
    synthetic_queue = _synthetic_queue(synthetic)
    queue = real_queue + synthetic_queue
    preflight: dict[str, object] = {
        "review_batch_id": review_batch_id,
        "real_review_items": len(real_queue),
        "synthetic_review_items": len(synthetic_queue),
        "total_review_items": len(queue),
        "contact_sheet_limit": contact_sheet_limit,
        "dry_run": dry_run,
    }
    if dry_run:
        return preflight
    package.mkdir(parents=True, exist_ok=False)
    queue_path = package / "review_decisions.csv"
    source_path = package / "source_decisions.csv"
    contact_index_path = package / "contact_sheet_index.csv"
    _write_csv(queue_path, QUEUE_FIELDS, queue)
    source_rows = _source_template(root / "manifests" / "source_events.jsonl")
    _write_csv(source_path, SOURCE_FIELDS, source_rows)
    sheet_count, contact_index = _contact_sheets(package, queue, limit=contact_sheet_limit)
    _write_csv(contact_index_path, ("review_item_id", "sheet", "position"), contact_index)

    missing_files = 0
    hash_mismatches = 0
    duplicate_hashes: Counter[str] = Counter()
    group_splits: dict[str, set[str]] = {}
    checked_images: set[str] = set()
    for row in queue:
        image_path = project_path(str(row.get("image_path", "")))
        image_key = str(image_path.resolve())
        if image_key in checked_images:
            continue
        checked_images.add(image_key)
        if not image_path.is_file():
            missing_files += 1
            continue
        digest = str(row.get("image_sha256", ""))
        duplicate_hashes[digest] += 1
        if _hash(image_path) != digest:
            hash_mismatches += 1
        group = str(row.get("group_id", ""))
        if group:
            group_splits.setdefault(group, set()).add("train_candidate")
    exact_duplicate_hashes = sum(count > 1 for count in duplicate_hashes.values())
    requirement_counts = Counter(str(row["review_requirement"]) for row in queue)
    report: dict[str, object] = preflight | {
        "dry_run": False,
        "review_package_version": REVIEW_PACKAGE_VERSION,
        "source_rows": len(source_rows),
        "contact_sheets": sheet_count,
        "automatic_checks": {
            "missing_files": missing_files,
            "hash_mismatches": hash_mismatches,
            "exact_duplicate_hashes": exact_duplicate_hashes,
            "group_split_crossings": sum(len(splits) > 1 for splits in group_splits.values()),
            "normalization_exit_passed": bool(
                _mapping(normalization.get("exit_condition")).get("passed")
            ),
            "proposal_exit_passed": bool(_mapping(proposal.get("exit_condition")).get("passed")),
            "synthetic_exit_passed": bool(_mapping(synthetic.get("exit_condition")).get("passed")),
        },
        "review_requirements": dict(sorted(requirement_counts.items())),
        "recorded_decisions": 0,
        "promoted_items": 0,
        "blocked_items": len(queue),
        "blockers": [
            "source terms/permitted use decisions are pending",
            "content and label decisions are pending",
            "independent secondary review is unavailable for rows that require it",
        ],
        "outputs": {
            "report": str(report_path),
            "audit_report": str(audit_path),
            "review_decisions": str(queue_path),
            "source_decisions": str(source_path),
            "contact_sheet_index": str(contact_index_path),
            "contact_sheet_directory": str(package / "contact_sheets"),
        },
        "exit_condition": {
            "package_ready": missing_files == 0
            and hash_mismatches == 0
            and bool(queue)
            and bool(source_rows),
            "promotion_complete": False,
            "basis": "Evidence is ranked and integrity-checked, but promotion remains fail-closed pending real review decisions.",
        },
        "resume_reused": False,
    }
    for path in (report_path, audit_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    return report


def validate_promotion_readiness(
    review_decisions: str | Path,
    source_decisions: str | Path,
    *,
    training_approval_policy: str | Path | None = None,
) -> dict[str, object]:
    """Validate edited B6 decisions and refuse promotion while any authority is missing."""
    from roadsign_assist.datasets.recovery_approval import (
        has_training_approval,
        load_training_approval,
    )

    policy = load_training_approval(training_approval_policy)
    with project_path(review_decisions).open(encoding="utf-8-sig", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    with project_path(source_decisions).open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    source_approved = {
        row["source_id"]
        for row in source_rows
        if row.get("source_review_outcome") == "approved"
        and row.get("permitted_use") not in {"", "none_pending_review"}
        and row.get("reviewer_id")
    }
    errors: list[dict[str, object]] = []
    accepted = 0
    for index, row in enumerate(review_rows, start=2):
        outcome = row.get("promotion_outcome", "")
        if not outcome:
            errors.append({"row": index, "code": "review_decision_pending"})
            continue
        if outcome not in PROMOTION_OUTCOMES:
            errors.append({"row": index, "code": "invalid_promotion_outcome"})
            continue
        if not row.get("primary_reviewer"):
            errors.append({"row": index, "code": "primary_reviewer_missing"})
        if outcome in ACCEPT_OUTCOMES:
            accepted += 1
            if row.get("source_id") not in source_approved:
                errors.append({"row": index, "code": "source_not_approved"})
            owner_accepted = has_training_approval(row, policy) and outcome not in {
                "accept_validation",
                "accept_internal_test",
            }
            if (
                row.get("review_requirement") == "full_review"
                and not row.get("secondary_reviewer")
                and not owner_accepted
            ):
                errors.append({"row": index, "code": "secondary_reviewer_missing"})
            if row.get("primary_reviewer") == row.get("secondary_reviewer"):
                errors.append({"row": index, "code": "reviewers_not_independent"})
    return {
        "review_items": len(review_rows),
        "training_approval_policy": policy,
        "accepted_items": accepted,
        "source_rows": len(source_rows),
        "approved_sources": len(source_approved),
        "errors": errors,
        "error_counts": dict(sorted(Counter(str(error["code"]) for error in errors).items())),
        "promotion_allowed": bool(accepted) and not errors,
    }


__all__ = [
    "ACCEPT_OUTCOMES",
    "PROMOTION_OUTCOMES",
    "REVIEW_PACKAGE_VERSION",
    "build_recovery_review_package",
    "validate_promotion_readiness",
]
