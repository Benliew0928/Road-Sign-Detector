"""Phase B4 closed-loop coverage reporting and next-batch query generation."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from roadsign_assist.datasets.recovery_acquisition import (
    DEFAULT_ACQUISITION_ROOT,
    DEFAULT_AUDIT_ROOT,
)
from roadsign_assist.paths import project_path

GAP_MINER_VERSION = "recovery_gap_miner_v1"
DEFAULT_CLASS_TARGET_DISTINCT_GROUPS = 30
SAFETY_PRIORITY = frozenset(
    {
        "give_way",
        "maximum_speed",
        "no_entry",
        "no_left_turn",
        "no_right_turn",
        "no_straight_ahead",
        "no_u_turn",
        "pedestrian_crossing",
        "stop",
        "traffic_signal_ahead",
    }
)


def _now() -> datetime:
    return datetime.now(UTC)


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value)


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)) and value != "":
        return int(value)
    return 0


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file():
        return []
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


def _json_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    items = cast(list[object], parsed)
    return [str(item) for item in items if str(item)]


def _family(label: str) -> str:
    if label.startswith("no_"):
        return "prohibition"
    if any(token in label for token in ("turn", "straight", "keep_", "one_way", "pass_")):
        return "directional"
    if any(token in label for token in ("maximum_speed", "height_", "weight_", "width_")):
        return "numeric_restriction"
    if label in {"stop", "give_way", "no_entry"}:
        return "priority_regulatory"
    if label == "unknown_sign":
        return "unknown"
    return "warning_or_information"


def _local_reviewed_coverage(path: Path) -> tuple[Counter[str], dict[str, set[str]]]:
    class_images: Counter[str] = Counter()
    class_groups: dict[str, set[str]] = defaultdict(set)
    if not path.is_file():
        return class_images, class_groups
    parquet: Any = pq
    table: Any = parquet.read_table(
        path,
        columns=["sha256", "classes_json", "groups_json", "collection_state"],
    )
    seen: set[tuple[str, str]] = set()
    for row in table.to_pylist():
        if str(row.get("collection_state", "")) != "existing_reviewed_release":
            continue
        sha256 = str(row.get("sha256", ""))
        groups = _json_list(row.get("groups_json"))
        for label in _json_list(row.get("classes_json")):
            identity = (sha256, label)
            if identity in seen:
                continue
            seen.add(identity)
            class_images[label] += 1
            class_groups[label].update(groups)
    return class_images, class_groups


def _source_metadata(object_ledger: Path) -> dict[str, Mapping[str, object]]:
    values: dict[str, Mapping[str, object]] = {}
    for row in _load_jsonl(object_ledger):
        object_id = str(row.get("object_id", ""))
        if object_id and row.get("coarse_location"):
            values[object_id] = row
    return values


def mine_recovery_gaps(
    proposal_report: str | Path,
    normalization_manifest: str | Path,
    *,
    gap_batch_id: str,
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    local_inventory: str | Path = "_archive/recovery_acquisition_v1/manifests/local_inventory.parquet",
    labels_path: str | Path = "models/exported/runtime/sign_classifier.labels.json",
    class_target_distinct_groups: int = DEFAULT_CLASS_TARGET_DISTINCT_GROUPS,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    """Compute separate reviewed/proposed coverage and ranked next-source queries."""
    if class_target_distinct_groups <= 0:
        raise ValueError("class_target_distinct_groups must be positive")
    root = project_path(acquisition_root)
    proposal_path = project_path(proposal_report)
    normalization_path = project_path(normalization_manifest)
    labels_file = project_path(labels_path)
    if not proposal_path.is_file() or not normalization_path.is_file() or not labels_file.is_file():
        raise FileNotFoundError("proposal, normalization, and labels artifacts must exist")
    proposal = _load_json(proposal_path)
    normalization = _load_json(normalization_path)
    proposal_outputs = _mapping(proposal.get("outputs"))
    progress_path = project_path(str(proposal_outputs.get("progress_jsonl", "")))
    proposal_rows = _load_jsonl(progress_path)
    labels_value: object = json.loads(labels_file.read_text(encoding="utf-8"))
    labels = (
        sorted(str(value) for value in cast(list[object], labels_value))
        if isinstance(labels_value, list)
        else []
    )
    preflight: dict[str, object] = {
        "gap_batch_id": gap_batch_id,
        "proposal_report": str(proposal_path),
        "normalization_manifest": str(normalization_path),
        "proposal_images": len(proposal_rows),
        "catalogue_classes": len(labels),
        "dry_run": dry_run,
    }
    if dry_run:
        return preflight
    manifest_path = root / "manifests" / "gap_mining" / f"{gap_batch_id}.json"
    if manifest_path.exists():
        if resume:
            return dict(_load_json(manifest_path)) | {"resume_reused": True}
        raise FileExistsError(f"Refusing to overwrite gap report: {manifest_path}")

    reviewed_images, reviewed_groups = _local_reviewed_coverage(project_path(local_inventory))
    proposed_images: Counter[str] = Counter()
    proposed_groups: dict[str, set[str]] = defaultdict(set)
    families: Counter[str] = Counter()
    size_buckets: Counter[str] = Counter()
    proposal_states: Counter[str] = Counter()
    negative_types: Counter[str] = Counter()
    geographic_images: Counter[str] = Counter()
    metadata = _source_metadata(root / "manifests" / "object_events.jsonl")
    source_groups: set[str] = set()
    processed_objects: set[str] = set()
    for image_row in proposal_rows:
        object_id = str(image_row.get("object_id", ""))
        group_id = str(image_row.get("group_id", ""))
        processed_objects.add(object_id)
        if group_id:
            source_groups.add(group_id)
        location = str(metadata.get(object_id, {}).get("coarse_location", "")) or "unknown"
        geographic_images[location] += 1
        image_state = str(image_row.get("image_state", ""))
        if image_state == "candidate_no_sign":
            negative_types["candidate_no_sign"] += 1
        raw_proposals = image_row.get("proposals", [])
        if not isinstance(raw_proposals, list):
            continue
        for raw_proposal in cast(list[object], raw_proposals):
            current = _mapping(raw_proposal)
            label = str(current.get("semantic_sign_id", "unknown_sign"))
            state = str(current.get("candidate_state", ""))
            proposed_images[label] += 1
            if group_id:
                proposed_groups[label].add(group_id)
            families[_family(label)] += 1
            size_buckets[str(current.get("size_bucket", "unknown"))] += 1
            proposal_states[state] += 1
            if state in {"candidate_unknown", "candidate_unreadable", "candidate_hard_negative"}:
                negative_types[state] += 1

    class_rows: list[dict[str, object]] = []
    for label in labels:
        reviewed_group_count = len(reviewed_groups[label])
        proposed_group_count = len(proposed_groups[label])
        effective_distinct_groups = reviewed_group_count + len(
            proposed_groups[label] - reviewed_groups[label]
        )
        gap = max(0, class_target_distinct_groups - effective_distinct_groups)
        class_rows.append(
            {
                "semantic_sign_id": label,
                "family": _family(label),
                "reviewed_unique_images": reviewed_images[label],
                "reviewed_distinct_groups": reviewed_group_count,
                "unreviewed_proposals": proposed_images[label],
                "unreviewed_distinct_groups": proposed_group_count,
                "effective_distinct_groups_for_acquisition_planning_only": effective_distinct_groups,
                "target_distinct_groups": class_target_distinct_groups,
                "gap_distinct_groups": gap,
                "safety_priority": label in SAFETY_PRIORITY,
            }
        )
    class_rows.sort(
        key=lambda row: (
            not bool(row["safety_priority"]),
            -_integer(row["gap_distinct_groups"]),
            str(row["semantic_sign_id"]),
        )
    )
    next_queries: list[dict[str, object]] = []
    for rank, row in enumerate((row for row in class_rows if row["gap_distinct_groups"]), start=1):
        if rank > 30:
            break
        label = str(row["semantic_sign_id"])
        phrase = label.replace("_", " ")
        next_queries.append(
            {
                "rank": rank,
                "semantic_sign_id": label,
                "family": row["family"],
                "gap_distinct_groups": row["gap_distinct_groups"],
                "safety_priority": row["safety_priority"],
                "wikimedia_query": f'Malaysia "{phrase}" traffic sign road',
                "repository_query": f'"{phrase}" road sign dataset Malaysia OR Southeast Asia',
                "kartaview_strategy": "sample additional distinct Malaysian sequences; semantic filtering occurs after proposals",
                "required_candidate_state": "quarantined_unreviewed",
            }
        )

    raw_downloaded = _integer(normalization.get("candidate_objects", 0))
    normalized_unique = _integer(
        _mapping(normalization.get("states")).get("normalized_candidate", 0)
    )
    relevant = _integer(normalization.get("road_context_score_at_least_0_35", 0))
    unique_yield = normalized_unique / raw_downloaded if raw_downloaded else 0.0
    relevant_unique_yield = min(normalized_unique, relevant) / raw_downloaded if raw_downloaded else 0.0
    review_debt = {
        "proposal_images": len(proposal_rows),
        "proposal_instances": sum(proposed_images.values()),
        "candidate_no_sign_images": negative_types["candidate_no_sign"],
        "unknown_or_unreadable_proposals": negative_types["candidate_unknown"]
        + negative_types["candidate_unreadable"],
        "source_licence_review_required": True,
        "privacy_flagged_normalized_objects": sum(
            _integer(value) for value in _mapping(normalization.get("privacy_flags")).values()
        ),
        "promotion_allowed": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    acquisition_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    storage = {
        "free_bytes": usage.free,
        "free_gib": round(usage.free / 1024**3, 3),
        "acquisition_bytes": acquisition_bytes,
        "acquisition_gib": round(acquisition_bytes / 1024**3, 3),
        "minimum_free_gib": 15.0,
        "maximum_acquisition_gib": 10.0,
        "within_guard": usage.free >= 15 * 1024**3 and acquisition_bytes <= 10 * 1024**3,
    }
    coverage = {
        "generated_at": _now().isoformat(),
        "gap_batch_id": gap_batch_id,
        "reviewed_counts_are_gate_credit": True,
        "proposal_counts_are_gate_credit": False,
        "classes": class_rows,
        "families_unreviewed": dict(sorted(families.items())),
        "apparent_size_unreviewed": dict(sorted(size_buckets.items())),
        "proposal_states": dict(sorted(proposal_states.items())),
        "negative_types_unreviewed": dict(sorted(negative_types.items())),
        "source_groups_in_batch": len(source_groups),
        "geography_images": dict(sorted(geographic_images.items())),
        "camera": {"unknown": len(processed_objects)},
        "weather": {"unknown": len(processed_objects)},
        "timeofday": {"unknown": len(processed_objects)},
        "lighting": {"unknown": len(processed_objects)},
        "road_type": {"unknown": len(processed_objects)},
        "real_normalized_images": normalized_unique,
        "synthetic_images": 0,
    }
    audit = project_path(audit_root)
    gap_audit = audit / "gap_mining" / f"{gap_batch_id}.json"
    coverage_path = audit / "coverage.json"
    review_debt_path = audit / "review_debt.json"
    storage_path = audit / "storage_report.json"
    query_path = root / "manifests" / "gap_mining" / f"{gap_batch_id}_next_queries.json"
    report: dict[str, object] = preflight | {
        "dry_run": False,
        "gap_miner_version": GAP_MINER_VERSION,
        "coverage": coverage,
        "review_debt": review_debt,
        "storage": storage,
        "batch_yield": {
            "raw_downloaded": raw_downloaded,
            "unique_normalized": normalized_unique,
            "road_context_scored_relevant": relevant,
            "unique_yield": round(unique_yield, 6),
            "relevant_unique_yield": round(relevant_unique_yield, 6),
            "below_five_percent": relevant_unique_yield < 0.05,
            "consecutive_below_five_percent": 0 if relevant_unique_yield >= 0.05 else 1,
        },
        "next_queries": next_queries,
        "suppressed_sources": [],
        "next_action": "Run the highest-ranked source queries, then normalize and propose the next bounded batch.",
        "outputs": {
            "manifest": str(manifest_path),
            "gap_audit": str(gap_audit),
            "coverage": str(coverage_path),
            "review_debt": str(review_debt_path),
            "storage": str(storage_path),
            "next_queries": str(query_path),
        },
        "exit_condition": {
            "passed": bool(class_rows) and len(processed_objects) == len(proposal_rows),
            "basis": "Coverage, review debt, storage state, marginal yield, and ranked source queries are computable from the batch.",
        },
        "resume_reused": False,
    }
    payloads = {
        manifest_path: report,
        gap_audit: report,
        coverage_path: coverage,
        review_debt_path: review_debt,
        storage_path: storage,
        query_path: {"gap_batch_id": gap_batch_id, "queries": next_queries},
    }
    for path, payload in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    return report


__all__ = ["GAP_MINER_VERSION", "mine_recovery_gaps"]
