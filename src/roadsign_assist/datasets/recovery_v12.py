"""Derive, audit, materialize, and freeze the recovery V12 training release.

V12 is a split-only child of immutable V11.  It authorizes available-data
experiments, never a Phase-E2, safety, deployment, or runtime-promotion claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from PIL import Image

from roadsign_assist.datasets.recovery_phase_b import (
    GROUP_FIELDS,
    PHASE_B_FIELDS,
    audit_phase_b_manifest,
)
from roadsign_assist.paths import project_path

PARENT_RELEASE_ID = "recovery_phase_b_intake_20260906_v11"
RELEASE_ID = "recovery_phase_b_intake_20260906_v12"
PARENT_MANIFEST = Path("data/manifests/recovery_phase_b_intake_v11.csv")
MANIFEST = Path("data/manifests/recovery_phase_b_intake_v12.csv")
PARENT_SHA256 = "a0dceb8ce6ae9a1f83a3fe8138d7782be84c4697a82abd15fac607a032cdce4d"
POLICY = Path("configs/recovery/available_data_experimental_training_20260906_v1.json")
POLICY_ID = "available_data_experimental_training_20260906_v1"
SPLIT_SEED = 260906
RELEASE_ROOT = Path("_archive/recovery_acquisition_v1/releases") / RELEASE_ID
TRAINING_ROOT = Path("_archive/recovery_acquisition_v1/training_preparation") / RELEASE_ID
AUDIT_ROOT = Path("outputs/audit") / RELEASE_ID
FREEZE = Path("_archive/recovery_acquisition_v1/manifests/freezes") / f"{RELEASE_ID}.json"
SPENT_DENYLIST = Path("_archive/recovery_acquisition_v1/manifests/spent_data_denylist.csv")
EXTRA_FIELDS = (
    "parent_release_id",
    "parent_manifest_sha256",
    "split_component_id",
    "experimental_split_policy_id",
    "experimental_evaluation_eligibility",
    "experimental_evaluation_basis",
    "experimental_evaluation_limitations",
)
PLACEHOLDERS = frozenset(
    {
        "unknown",
        "unknown_not_recorded",
        "not_recorded",
        "not_applicable",
        "n/a",
        "na",
        "none",
    }
)
EVIDENCE_PATTERNS = (
    "accepted corrected full-context sheet",
    "passed the full-context no-sign audit",
    "passed the proposal and full-context image",
)
SPLIT_MAP = {"train": "train", "validation": "validation", "internal_test": "test"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _normal(value: object) -> str:
    return str(value or "").strip()


def _is_placeholder(value: object) -> bool:
    normalized = _normal(value).casefold()
    return not normalized or normalized in PLACEHOLDERS or normalized.startswith("unknown_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_fraction(value: str, seed: int = SPLIT_SEED) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(project_path(".").resolve()).as_posix()


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left == right:
            return
        if left < right:
            self.parent[right] = left
        else:
            self.parent[left] = right


@dataclass(frozen=True)
class _Component:
    component_id: str
    samples: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    eligible: bool
    eligibility_reason: str
    stratum: str


def _component_id(samples: Sequence[str], rows: Sequence[Mapping[str, str]]) -> str:
    tokens = [f"sample_id:{sample}" for sample in samples]
    for field in GROUP_FIELDS:
        tokens.extend(
            f"{field}:{value}"
            for value in sorted({_normal(row.get(field)) for row in rows})
            if not _is_placeholder(value)
        )
    digest = hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()[:20]
    return f"v12-component:{digest}"


def _component_eligibility(rows: Sequence[Mapping[str, str]]) -> tuple[bool, str]:
    if not rows or {row.get("source_id") for row in rows} != {"kartaview_malaysia"}:
        return False, "source_not_eligible_for_existing_data_evaluation"
    if any(row.get("capture_evidence") != "public_road_image" for row in rows):
        return False, "capture_evidence_not_public_road_image"
    for field in ("session_id", "route_id", "related_capture_group_id"):
        if any(_is_placeholder(row.get(field)) for row in rows):
            return False, f"{field}_not_concrete"
    notes = " ".join(_normal(row.get("review_notes")).casefold() for row in rows)
    if not any(pattern in notes for pattern in EVIDENCE_PATTERNS):
        return False, "no_recorded_full_context_owner_review"
    return True, "eligible_primary_full_context_review_with_concrete_route_session_group"


def _stratum(rows: Sequence[Mapping[str, str]]) -> str:
    kinds = {row.get("expected_kind") for row in rows}
    if "sign" in kinds:
        return "known_sign"
    if kinds & {"unknown_sign", "unreadable"}:
        return "unknown_or_unreadable_sign"
    return "no_sign"


def _components(rows: Sequence[dict[str, str]]) -> list[_Component]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sample[_normal(row.get("sample_id"))].append(row)
    if "" in by_sample:
        raise ValueError("V11 contains a blank sample_id")
    union = _UnionFind(by_sample)
    owners: dict[tuple[str, str], str] = {}
    for sample_id, sample_rows in by_sample.items():
        for field in GROUP_FIELDS:
            for value in {_normal(row.get(field)) for row in sample_rows}:
                if _is_placeholder(value):
                    continue
                token = (field, value)
                prior = owners.setdefault(token, sample_id)
                union.union(sample_id, prior)
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id in by_sample:
        grouped[union.find(sample_id)].append(sample_id)
    result: list[_Component] = []
    for samples in grouped.values():
        ordered_samples = tuple(sorted(samples))
        component_rows = tuple(row for sample in ordered_samples for row in by_sample[sample])
        eligible, reason = _component_eligibility(component_rows)
        result.append(
            _Component(
                component_id=_component_id(ordered_samples, component_rows),
                samples=ordered_samples,
                rows=component_rows,
                eligible=eligible,
                eligibility_reason=reason,
                stratum=_stratum(component_rows),
            )
        )
    return sorted(result, key=lambda item: item.component_id)


def _assign_components(components: Sequence[_Component]) -> dict[str, str]:
    assignments = {component.component_id: "train" for component in components}
    strata: dict[str, list[_Component]] = defaultdict(list)
    for component in components:
        if component.eligible:
            strata[component.stratum].append(component)
    for name, members in sorted(strata.items()):
        ordered = sorted(
            members,
            key=lambda item: (_stable_fraction(f"{name}:{item.component_id}"), item.component_id),
        )
        validation_count = round(len(ordered) * 0.15)
        internal_count = round(len(ordered) * 0.15)
        for component in ordered[:validation_count]:
            assignments[component.component_id] = "validation"
        for component in ordered[validation_count : validation_count + internal_count]:
            assignments[component.component_id] = "internal_test"
    return assignments


def verify_freezes(*, include_v12: bool = True) -> dict[str, Any]:
    freeze_root = project_path("_archive/recovery_acquisition_v1/manifests/freezes")
    names = [
        "recovery_phase_b_intake_20260905_v6",
        "recovery_phase_b_intake_20260905_v7",
        "recovery_phase_b_intake_20260906_v8",
        "recovery_phase_b_intake_20260906_v9",
        "recovery_phase_b_intake_20260906_v10",
        "recovery_phase_b_intake_20260906_v11",
    ]
    if include_v12:
        names.append(RELEASE_ID)
    releases: dict[str, Any] = {}
    all_passed = True
    for name in names:
        freeze_path = freeze_root / f"{name}.json"
        findings: list[dict[str, object]] = []
        if not freeze_path.is_file():
            findings.append({"path": str(freeze_path), "status": "missing_freeze"})
        else:
            payload = cast(dict[str, Any], json.loads(freeze_path.read_text(encoding="utf-8")))
            files = cast(Mapping[str, object], payload.get("files", {}))
            for raw_path, raw_expected in files.items():
                path = Path(raw_path)
                if not path.is_file():
                    findings.append({"path": str(path), "status": "missing"})
                    continue
                actual = _sha256(path)
                expected = str(raw_expected)
                if actual != expected:
                    findings.append(
                        {
                            "path": str(path),
                            "status": "hash_mismatch",
                            "expected": expected,
                            "actual": actual,
                        }
                    )
        passed = not findings
        all_passed = all_passed and passed
        releases[name] = {"passed": passed, "findings": findings}
    return {"checked_at": _utc_now(), "passed": all_passed, "releases": releases}


def derive_v12(*, reuse_existing: bool = False) -> dict[str, Any]:
    parent = project_path(PARENT_MANIFEST)
    parent_exclusions = project_path(
        Path("_archive/recovery_acquisition_v1/releases")
        / PARENT_RELEASE_ID
        / "quality_exclusions.csv"
    )
    manifest = project_path(MANIFEST)
    canonical = project_path(RELEASE_ROOT / "canonical_intake.csv")
    exclusions = project_path(RELEASE_ROOT / "quality_exclusions.csv")
    report_path = project_path(RELEASE_ROOT / "derivation_report.json")
    if _sha256(parent) != PARENT_SHA256:
        raise ValueError("Immutable V11 manifest hash changed; refusing V12 derivation")
    old_freezes = verify_freezes(include_v12=False)
    if not old_freezes["passed"]:
        raise ValueError("At least one V6-V11 freeze is invalid; refusing V12 derivation")
    if any(path.exists() for path in (manifest, canonical, report_path)):
        if not reuse_existing or not all(
            path.is_file() for path in (manifest, canonical, report_path)
        ):
            raise FileExistsError(
                "V12 outputs already exist or are partial; use explicit reuse after inspection"
            )
        report = cast(dict[str, Any], json.loads(report_path.read_text(encoding="utf-8")))
        if report.get("manifest_sha256") != _sha256(manifest) or _sha256(canonical) != _sha256(
            manifest
        ):
            raise ValueError("Existing V12 derivation artifacts do not match their recorded hashes")
        _exclusion_fields, exclusion_rows = _read_csv(parent_exclusions)
        if len(exclusion_rows) != 28:
            raise ValueError(f"V11 quality-exclusion count changed: {len(exclusion_rows)}")
        if exclusions.is_file() and _sha256(exclusions) != _sha256(parent_exclusions):
            raise ValueError("Existing V12 quality-exclusion copy differs from V11")
        if not exclusions.is_file():
            exclusions.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent_exclusions, exclusions)
        report.update(
            {
                "preserved_quality_exclusions": str(exclusions),
                "preserved_quality_exclusions_sha256": _sha256(exclusions),
                "preserved_quality_exclusion_rows": len(exclusion_rows),
            }
        )
        _write_json(report_path, report)
        return report | {"reused": True}

    fields, rows = _read_csv(parent)
    if len(rows) != 31_192:
        raise ValueError(f"V11 row count changed: {len(rows)}")
    components = _components(rows)
    assignments = _assign_components(components)
    component_for_sample = {
        sample: component for component in components for sample in component.samples
    }
    output_rows: list[dict[str, object]] = []
    for row in rows:
        component = component_for_sample[row["sample_id"]]
        split = assignments[component.component_id]
        copied: dict[str, object] = dict(row)
        copied.update(
            {
                "release_id": RELEASE_ID,
                "split": split,
                "parent_release_id": PARENT_RELEASE_ID,
                "parent_manifest_sha256": PARENT_SHA256,
                "split_component_id": component.component_id,
                "experimental_split_policy_id": POLICY_ID,
                "experimental_evaluation_eligibility": (
                    "eligible" if component.eligible else "train_only"
                ),
                "experimental_evaluation_basis": component.eligibility_reason,
                "experimental_evaluation_limitations": (
                    "physical_sign_identity_unmeasured; no independent secondary review; "
                    "not Phase E2; not final replacement evidence"
                ),
            }
        )
        output_rows.append(copied)
    output_fields = list(fields) + [field for field in EXTRA_FIELDS if field not in fields]
    _write_csv(manifest, output_fields, output_rows)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest, canonical)
    _exclusion_fields, exclusion_rows = _read_csv(parent_exclusions)
    if len(exclusion_rows) != 28:
        raise ValueError(f"V11 quality-exclusion count changed: {len(exclusion_rows)}")
    shutil.copyfile(parent_exclusions, exclusions)
    manifest_hash = _sha256(manifest)
    if _sha256(canonical) != manifest_hash:
        raise RuntimeError("V12 canonical archive copy differs from the manifest")

    component_counts = Counter(assignments.values())
    eligible_counts = Counter(
        assignments[component.component_id] for component in components if component.eligible
    )
    image_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter(str(row["split"]) for row in output_rows)
    domain_images: dict[str, Counter[str]] = defaultdict(Counter)
    seen_samples: set[str] = set()
    for row in output_rows:
        sample = str(row["sample_id"])
        if sample in seen_samples:
            continue
        seen_samples.add(sample)
        split = str(row["split"])
        image_counts[split] += 1
        domain_images[str(row["capture_domain"])][split] += 1
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "release_id": RELEASE_ID,
        "parent_release_id": PARENT_RELEASE_ID,
        "parent_manifest": str(parent),
        "parent_manifest_sha256": PARENT_SHA256,
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "canonical_archive_copy": str(canonical),
        "preserved_quality_exclusions": str(exclusions),
        "preserved_quality_exclusions_sha256": _sha256(exclusions),
        "preserved_quality_exclusion_rows": len(exclusion_rows),
        "created_at": _utc_now(),
        "split_seed": SPLIT_SEED,
        "algorithm": "connected_known_group_components_then_stratified_70_15_15_eligible_groups_v1",
        "rows": len(output_rows),
        "images": len(seen_samples),
        "components": len(components),
        "component_counts": dict(component_counts),
        "row_counts": dict(row_counts),
        "image_counts": dict(image_counts),
        "domain_image_counts": {key: dict(value) for key, value in domain_images.items()},
        "eligible_component_counts": dict(eligible_counts),
        "eligible_components": sum(component.eligible for component in components),
        "train_only_components": sum(not component.eligible for component in components),
        "eligibility_reasons": dict(
            Counter(component.eligibility_reason for component in components)
        ),
        "phase_e2_images": 0,
        "limitations": [
            "Unknown physical-sign identities are not asserted to be separated.",
            "Development and internal-test data are available-data experimental evidence only.",
            "Coarse or blanket-only source groups remain train-only.",
            "No collection floor, Phase-E2 gate, safety gate, or replacement gate is claimed passed.",
        ],
    }
    _write_json(report_path, report)
    return report | {"reused": False}


def _known_group_leakage(
    rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    leakage: dict[str, Any] = {}
    unknowns: dict[str, Any] = {}
    for field in GROUP_FIELDS:
        known: dict[str, set[str]] = defaultdict(set)
        placeholder_splits: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = _normal(row.get(field))
            split = _normal(row.get("split"))
            if _is_placeholder(value):
                if value:
                    placeholder_splits[value].add(split)
            else:
                known[value].add(split)
        collisions = [
            {"value": value, "splits": sorted(splits)}
            for value, splits in sorted(known.items())
            if len(splits) > 1
        ]
        if collisions:
            leakage[field] = collisions
        if placeholder_splits:
            unknowns[field] = {
                value: {
                    "splits": sorted(splits),
                    "status": "unmeasured_unknown_identity_not_counted_as_pass",
                }
                for value, splits in sorted(placeholder_splits.items())
            }
    return leakage, unknowns


def _verify_parent_derivation(
    parent_rows: Sequence[Mapping[str, str]], rows: Sequence[Mapping[str, str]]
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    parent_by_instance = {row["instance_id"]: row for row in parent_rows}
    if len(parent_by_instance) != len(parent_rows):
        errors.append({"code": "parent_duplicate_instance_ids"})
        return errors
    if len(rows) != len(parent_rows):
        errors.append({"code": "row_count", "parent": len(parent_rows), "v12": len(rows)})
    allowed_changes = {"release_id", "split", *EXTRA_FIELDS}
    for row_number, row in enumerate(rows, start=2):
        parent = parent_by_instance.get(row.get("instance_id", ""))
        if parent is None:
            errors.append({"row": row_number, "code": "instance_not_in_parent"})
            continue
        changed = [
            field
            for field in parent
            if field not in allowed_changes
            and _normal(parent.get(field)) != _normal(row.get(field))
        ]
        if changed:
            errors.append(
                {"row": row_number, "code": "non_split_parent_mutation", "fields": changed}
            )
    return errors


def _verify_images(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    first_by_hash: dict[str, Mapping[str, str]] = {}
    for row in rows:
        first_by_hash.setdefault(row["image_sha256"], row)
    for index, (expected, row) in enumerate(sorted(first_by_hash.items()), start=1):
        path = Path(row["image_path"])
        if not path.is_file():
            findings.append({"status": "missing", "path": str(path), "sha256": expected})
            continue
        actual = _sha256(path)
        if actual != expected:
            findings.append(
                {
                    "status": "hash_mismatch",
                    "path": str(path),
                    "expected": expected,
                    "actual": actual,
                }
            )
        if index % 2500 == 0:
            print(f"[audit] verified {index}/{len(first_by_hash)} V12 images", flush=True)
    return findings


def audit_v12(*, verify_images: bool = True) -> dict[str, Any]:
    parent_path = project_path(PARENT_MANIFEST)
    manifest_path = project_path(MANIFEST)
    if _sha256(parent_path) != PARENT_SHA256:
        raise ValueError("Immutable V11 manifest hash changed")
    _parent_fields, parent_rows = _read_csv(parent_path)
    fields, rows = _read_csv(manifest_path)
    exclusions_path = project_path(RELEASE_ROOT / "quality_exclusions.csv")
    _exclusion_fields, exclusion_rows = _read_csv(exclusions_path)
    exclusion_instance_ids = {row.get("instance_id", "") for row in exclusion_rows}
    accepted_instance_ids = {row.get("instance_id", "") for row in rows}
    exclusion_intersection = sorted(
        identifier
        for identifier in exclusion_instance_ids & accepted_instance_ids
        if identifier
    )
    missing = sorted(set(PHASE_B_FIELDS + EXTRA_FIELDS) - set(fields))
    derivation_errors = _verify_parent_derivation(parent_rows, rows)
    known_leakage, unknown_identities = _known_group_leakage(rows)
    base = audit_phase_b_manifest(
        manifest_path,
        training_approval_policy="configs/recovery/owner_training_approval_20260905.json",
    )
    waived_double_review_errors: list[dict[str, object]] = []
    hard_validation_errors: list[dict[str, object]] = []
    for error in cast(list[dict[str, object]], base["validation_errors"]):
        raw_row_number = error.get("row", 0)
        row_number = int(raw_row_number) if isinstance(raw_row_number, (int, str)) else 0
        row = rows[row_number - 2] if 2 <= row_number <= len(rows) + 1 else None
        if (
            error.get("code") == "double_review_required"
            and row is not None
            and row.get("split") in {"validation", "internal_test"}
            and row.get("experimental_evaluation_eligibility") == "eligible"
            and row.get("experimental_split_policy_id") == POLICY_ID
        ):
            waived_double_review_errors.append(error)
        else:
            hard_validation_errors.append(error)
    split_policy_errors: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=2):
        split = row.get("split")
        eligible = row.get("experimental_evaluation_eligibility") == "eligible"
        if split not in {"train", "validation", "internal_test"}:
            split_policy_errors.append(
                {"row": row_number, "code": "forbidden_split", "value": split}
            )
        if split in {"validation", "internal_test"} and not eligible:
            split_policy_errors.append({"row": row_number, "code": "ineligible_evaluation_row"})
        if (
            split in {"validation", "internal_test"}
            and row.get("source_id") != "kartaview_malaysia"
        ):
            split_policy_errors.append({"row": row_number, "code": "evaluation_source_not_allowed"})
    eligible_groups: dict[str, str] = {}
    for row in rows:
        if row.get("experimental_evaluation_eligibility") == "eligible":
            eligible_groups[row["split_component_id"]] = row["split"]
    eligible_counts = Counter(eligible_groups.values())
    eligible_total = sum(eligible_counts.values())
    eligible_proportions = {
        split: eligible_counts[split] / eligible_total if eligible_total else 0.0
        for split in ("train", "validation", "internal_test")
    }
    eligible_balance = bool(eligible_total) and (
        0.65 <= eligible_proportions["train"] <= 0.75
        and 0.10 <= eligible_proportions["validation"] <= 0.20
        and 0.10 <= eligible_proportions["internal_test"] <= 0.20
    )
    denylist_hashes: set[str] = set()
    with project_path(SPENT_DENYLIST).open(newline="", encoding="utf-8-sig") as handle:
        denylist_hashes = {row["sha256"] for row in csv.DictReader(handle) if row.get("sha256")}
    manifest_hashes = {row["image_sha256"] for row in rows}
    spent_intersection = sorted(manifest_hashes & denylist_hashes)
    foreign_errors: list[str] = [
        row["instance_id"]
        for row in rows
        if row.get("source_id") in {"gtsdb_official", "tt100k_official"}
        and (
            row.get("expected_kind") not in {"unknown_sign", "no_sign"}
            or (
                row.get("expected_kind") == "unknown_sign"
                and row.get("semantic_sign_id") != "unknown_sign"
            )
        )
    ]
    image_findings = _verify_images(rows) if verify_images else []
    old_freezes = verify_freezes(include_v12=False)
    requirements = {
        "parent_v11_hash_verified": _sha256(parent_path) == PARENT_SHA256,
        "v6_v11_freezes_verified": bool(old_freezes["passed"]),
        "required_columns": not missing,
        "split_only_derivation": not derivation_errors,
        "row_validation_after_explicit_experimental_waiver": not hard_validation_errors,
        "eligible_split_policy": not split_policy_errors,
        "eligible_group_balance": eligible_balance,
        "zero_known_group_leakage": not known_leakage,
        "spent_evaluation_data_absent": not spent_intersection,
        "foreign_semantics_remain_generic": not foreign_errors,
        "v11_quality_exclusions_preserved": (
            len(exclusion_rows) == 28
            and _sha256(exclusions_path)
            == _sha256(
                project_path(
                    Path("_archive/recovery_acquisition_v1/releases")
                    / PARENT_RELEASE_ID
                    / "quality_exclusions.csv"
                )
            )
            and not exclusion_intersection
        ),
        "image_files_and_hashes_verified": verify_images and not image_findings,
        "phase_e2_absent_by_design": not any(row.get("split") == "phase_e2" for row in rows),
    }
    experimental_ready = all(requirements.values())
    waivers = {
        "new_real_development_recordings": {
            "status": "waived_for_experimental_training_only",
            "satisfied": False,
        },
        "real_development_collection_floors": {
            "status": "waived_for_experimental_training_only",
            "satisfied": False,
            "measurements": {
                key: base["counts"].get(key)
                for key in (
                    "road_development_images",
                    "road_development_sign_instances",
                    "road_development_accepted_negative_images",
                    "road_development_negative_fraction",
                    "road_development_small_instances_at_most_1pct",
                    "road_development_very_small_instances_at_most_0_1pct",
                )
            },
            "floors": base["floors"],
        },
        "independent_secondary_review_for_available_data_dev_split": {
            "status": "waived_for_experimental_candidate_selection_only",
            "satisfied": False,
            "affected_rows": len(waived_double_review_errors),
        },
        "unknown_physical_sign_identity": {
            "status": "unmeasured",
            "satisfied": False,
        },
    }
    final_blockers = [
        "separately locked Phase E2 target-device evidence is absent",
        "available-data development/internal-test evidence is not independent Phase E2 evidence",
        "real-camera accuracy and safety gates are unmeasured",
        "paired legacy comparison is unmeasured",
        "target-hardware throughput, latency, reliability, shadow, and canary gates are unmeasured",
        "runtime promotion remains forbidden",
    ]
    report: dict[str, Any] = {
        "schema_version": "3.0",
        "audited_at": _utc_now(),
        "release_id": RELEASE_ID,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "parent_release_id": PARENT_RELEASE_ID,
        "parent_manifest_sha256": PARENT_SHA256,
        "counts": base["counts"],
        "base_gate_b_checks": base["checks"],
        "available_data_experimental_training_readiness": {
            "status": "ready" if experimental_ready else "blocked",
            "ready": experimental_ready,
            "requirements": requirements,
            "eligible_group_counts": dict(eligible_counts),
            "eligible_group_proportions": eligible_proportions,
        },
        "final_replacement_readiness": {
            "status": "blocked",
            "ready": False,
            "phase_e2_status": "blocked_absent",
            "runtime_promotion_allowed": False,
            "blockers": final_blockers,
        },
        "waivers_and_unmeasured_requirements": waivers,
        "known_group_leakage": known_leakage,
        "unknown_identity_evidence": unknown_identities,
        "spent_hash_intersection": spent_intersection,
        "foreign_semantics_errors": foreign_errors,
        "quality_exclusions": {
            "path": str(exclusions_path),
            "sha256": _sha256(exclusions_path),
            "rows": len(exclusion_rows),
            "accepted_instance_intersection": exclusion_intersection,
        },
        "missing_columns": missing,
        "derivation_errors": derivation_errors,
        "hard_validation_errors": hard_validation_errors,
        "waived_experimental_double_review_errors": waived_double_review_errors,
        "split_policy_errors": split_policy_errors,
        "image_integrity_findings": image_findings,
        "old_freeze_verification": old_freezes,
        "legacy_gate_b_passed": False,
    }
    audit_root = project_path(AUDIT_ROOT)
    _write_json(audit_root / "audit.json", report)
    lines = [
        "# Recovery V12 readiness audit",
        "",
        f"- Available-data experimental-training readiness: `{'READY' if experimental_ready else 'BLOCKED'}`",
        "- Final replacement readiness: `BLOCKED`",
        "- Phase E2: `BLOCKED_ABSENT`",
        "- Runtime promotion: `FORBIDDEN`",
        f"- Manifest SHA-256: `{report['manifest_sha256']}`",
        "",
        "## Experimental-training requirements",
        "",
        "| Requirement | Result |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |" for name, value in requirements.items()
    )
    lines.extend(
        [
            "",
            "## Waived or unmeasured",
            "",
            "Real-development collection floors and new real-development recordings are waived only for experimental training. They are not satisfied and do not support real-camera, safety, or replacement claims. Unknown physical-sign identity remains unmeasured. Independent secondary review is waived only for the authorized available-data candidate-selection split.",
            "",
            "## Remaining final blockers",
            "",
        ]
    )
    lines.extend(f"- {blocker}" for blocker in final_blockers)
    lines.append("")
    _write_text(audit_root / "report.md", "\n".join(lines))
    return report


def _ensure_hardlink(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != expected_sha256:
            raise ValueError(f"Existing prepared image is invalid: {destination}")
        return
    os.link(source, destination)


def _bbox(row: Mapping[str, str]) -> tuple[float, float, float, float]:
    return tuple(float(row[field]) for field in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))  # type: ignore[return-value]


def _crop_box(
    row: Mapping[str, str], *, padding: float, jitter_key: str = ""
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _bbox(row)
    width, height = int(row["image_width"]), int(row["image_height"])
    box_width, box_height = x2 - x1, y2 - y1
    dx = dy = 0.0
    if jitter_key:
        dx = (_stable_fraction(jitter_key + ":x") - 0.5) * 0.12 * box_width
        dy = (_stable_fraction(jitter_key + ":y") - 0.5) * 0.12 * box_height
    return (
        max(0, int(x1 - padding * box_width + dx)),
        max(0, int(y1 - padding * box_height + dy)),
        min(width, int(x2 + padding * box_width + dx + 0.999)),
        min(height, int(y2 + padding * box_height + dy + 0.999)),
    )


def _write_dataset_content(path: Path, entries: Sequence[Mapping[str, object]]) -> str:
    payload = {"schema_version": "1.0", "created_at": _utc_now(), "files": list(entries)}
    _write_json(path, payload)
    return _sha256(path)


def _verify_dataset_content(path: Path) -> list[dict[str, object]]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    findings: list[dict[str, object]] = []
    for entry in cast(list[dict[str, object]], payload.get("files", [])):
        file_path = project_path(str(entry["path"]))
        if not file_path.is_file():
            findings.append({"path": str(file_path), "status": "missing"})
            continue
        actual = _sha256(file_path)
        if actual != entry.get("sha256"):
            findings.append(
                {
                    "path": str(file_path),
                    "status": "hash_mismatch",
                    "actual": actual,
                    "expected": entry.get("sha256"),
                }
            )
    return findings


def prepare_datasets(*, reuse_existing: bool = False) -> dict[str, Any]:
    audit_path = project_path(AUDIT_ROOT / "audit.json")
    if not audit_path.is_file():
        raise FileNotFoundError("V12 audit is required before dataset preparation")
    audit = cast(dict[str, Any], json.loads(audit_path.read_text(encoding="utf-8")))
    readiness = cast(dict[str, Any], audit["available_data_experimental_training_readiness"])
    if readiness.get("ready") is not True:
        raise ValueError("V12 available-data experimental-training audit is not ready")
    manifest_path = project_path(MANIFEST)
    _, rows = _read_csv(manifest_path)
    root = project_path(TRAINING_ROOT)
    detector_root = root / "detector"
    classifier_root = root / "classifier"
    detector_content = detector_root / "content_manifest.json"
    classifier_content = classifier_root / "content_manifest.json"
    if detector_content.exists() or classifier_content.exists():
        if not reuse_existing:
            raise FileExistsError(
                "Prepared V12 datasets already exist or are partial; use explicit reuse"
            )
        existing_content = [
            path for path in (detector_content, classifier_content) if path.is_file()
        ]
        findings = [
            finding
            for content_path in existing_content
            for finding in _verify_dataset_content(content_path)
        ]
        if findings:
            raise ValueError(
                f"Existing prepared V12 dataset content failed verification: {findings[:3]}"
            )
        if detector_content.is_file() and classifier_content.is_file():
            return {
                "reused": True,
                "detector_root": str(detector_root),
                "classifier_root": str(classifier_root),
                "detector_content_manifest_sha256": _sha256(detector_content),
                "classifier_content_manifest_sha256": _sha256(classifier_content),
            }

    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)
    detector_entries: list[dict[str, object]] = []
    detector_counts: Counter[str] = Counter()
    detector_boxes: Counter[str] = Counter()
    for index, (_sample_id, sample_rows) in enumerate(sorted(by_sample.items()), start=1):
        first = sample_rows[0]
        split = SPLIT_MAP[first["split"]]
        source = Path(first["image_path"])
        suffix = source.suffix.lower() or ".jpg"
        image_name = f"{first['image_sha256']}{suffix}"
        image_path = detector_root / "images" / split / image_name
        label_path = detector_root / "labels" / split / f"{first['image_sha256']}.txt"
        _ensure_hardlink(source, image_path, first["image_sha256"])
        label_lines: list[str] = []
        for row in sample_rows:
            if row["expected_kind"] == "no_sign":
                continue
            x1, y1, x2, y2 = _bbox(row)
            width, height = float(row["image_width"]), float(row["image_height"])
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            label_lines.append(f"0 {center_x:.9f} {center_y:.9f} {box_width:.9f} {box_height:.9f}")
        label_text = "\n".join(label_lines) + ("\n" if label_lines else "")
        if label_path.exists():
            if label_path.read_text(encoding="utf-8") != label_text:
                raise ValueError(f"Existing detector label differs: {label_path}")
        else:
            _write_text(label_path, label_text)
        detector_entries.extend(
            [
                {
                    "path": _project_relative(image_path),
                    "sha256": first["image_sha256"],
                    "kind": "image",
                },
                {
                    "path": _project_relative(label_path),
                    "sha256": _sha256(label_path),
                    "kind": "label",
                },
            ]
        )
        detector_counts[split] += 1
        detector_boxes[split] += len(label_lines)
        if index % 2500 == 0:
            print(f"[prepare] detector {index}/{len(by_sample)} images", flush=True)
    data_yaml = detector_root / "data.yaml"
    _write_text(
        data_yaml,
        "\n".join(
            [
                f"path: {detector_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/validation",
                "test: images/test",
                "names:",
                "  0: traffic_sign",
                "",
            ]
        ),
    )
    detector_entries.append(
        {
            "path": _project_relative(data_yaml),
            "sha256": _sha256(data_yaml),
            "kind": "configuration",
        }
    )
    detector_content_hash = _write_dataset_content(detector_content, detector_entries)
    detector_metadata: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_id": RELEASE_ID + "_detector",
        "source_release": RELEASE_ID,
        "source_manifest": _project_relative(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "annotation_status": "experimental_available_data_only",
        "release_status": "experimental_training_ready_final_replacement_blocked",
        "coursework_images_included": 0,
        "internal_academic_only": True,
        "publication_prohibited": True,
        "dvc_remote_push_allowed": False,
        "runtime_promotion_allowed": False,
        "phase_e2_present": False,
        "split_counts": dict(detector_counts),
        "box_counts": dict(detector_boxes),
        "content_manifest": _project_relative(detector_content),
        "content_manifest_sha256": detector_content_hash,
        "foreign_semantics": "generic traffic_sign detector supervision only",
    }
    _write_json(detector_root / "dataset_metadata.json", detector_metadata)

    labels = cast(
        list[str],
        json.loads(
            project_path("models/exported/runtime/sign_classifier.labels.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    label_set = set(labels)
    classifier_entries: list[dict[str, object]] = []
    classifier_counts: Counter[str] = Counter()
    crop_manifest_rows: list[dict[str, object]] = []
    sign_rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["expected_kind"] == "sign" and row["semantic_sign_id"] in label_set:
            sign_rows_by_image[row["image_path"]].append(row)
    total_sign_rows = sum(len(values) for values in sign_rows_by_image.values())
    processed = 0
    for image_value, image_rows in sorted(sign_rows_by_image.items()):
        with Image.open(image_value) as source_image:
            image = source_image.convert("RGB")
            for row in image_rows:
                split = SPLIT_MAP[row["split"]]
                variants = (("tight", 0.05),)
                if split == "train":
                    variants += (("detector_context", 0.22),)
                for variant, padding in variants:
                    box = _crop_box(
                        row,
                        padding=padding,
                        jitter_key=row["instance_id"] if variant == "detector_context" else "",
                    )
                    crop = image.crop(box)
                    safe_instance = hashlib.sha256(row["instance_id"].encode()).hexdigest()[:24]
                    name = f"{safe_instance}_{variant}.jpg"
                    destination = classifier_root / split / row["semantic_sign_id"] / name
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_suffix(".jpg.tmp")
                        crop.save(
                            temporary, format="JPEG", quality=92, optimize=False, progressive=False
                        )
                        temporary.replace(destination)
                    crop_hash = _sha256(destination)
                    classifier_entries.append(
                        {
                            "path": _project_relative(destination),
                            "sha256": crop_hash,
                            "kind": "crop",
                        }
                    )
                    classifier_counts[split] += 1
                    crop_manifest_rows.append(
                        {
                            "release_id": RELEASE_ID,
                            "instance_id": row["instance_id"],
                            "sample_id": row["sample_id"],
                            "split": split,
                            "semantic_sign_id": row["semantic_sign_id"],
                            "variant": variant,
                            "crop_path": _project_relative(destination),
                            "crop_sha256": crop_hash,
                            "source_image_sha256": row["image_sha256"],
                            "split_component_id": row["split_component_id"],
                            "source_id": row["source_id"],
                            "capture_domain": row["capture_domain"],
                        }
                    )
                processed += 1
                if processed % 2500 == 0:
                    print(
                        f"[prepare] classifier {processed}/{total_sign_rows} sign rows", flush=True
                    )
    labels_path = classifier_root / "labels.json"
    _write_json(labels_path, labels)
    classifier_entries.append(
        {
            "path": _project_relative(labels_path),
            "sha256": _sha256(labels_path),
            "kind": "configuration",
        }
    )
    crop_manifest = classifier_root / "crop_manifest.csv"
    crop_fields = list(crop_manifest_rows[0]) if crop_manifest_rows else []
    _write_csv(crop_manifest, crop_fields, crop_manifest_rows)
    classifier_entries.append(
        {
            "path": _project_relative(crop_manifest),
            "sha256": _sha256(crop_manifest),
            "kind": "manifest",
        }
    )
    classifier_content_hash = _write_dataset_content(classifier_content, classifier_entries)
    classifier_metadata: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_id": RELEASE_ID + "_classifier",
        "source_release": RELEASE_ID,
        "source_manifest": _project_relative(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "annotation_status": "experimental_available_data_only",
        "release_status": "experimental_training_ready_final_replacement_blocked",
        "coursework_images_included": 0,
        "internal_academic_only": True,
        "publication_prohibited": True,
        "dvc_remote_push_allowed": False,
        "runtime_promotion_allowed": False,
        "phase_e2_present": False,
        "split_counts": dict(classifier_counts),
        "labels": len(labels),
        "crop_policy": "tight_plus_deterministic_detector_context_train; tight_only_evaluation",
        "horizontal_flip": "disabled",
        "deployed_crop_limitation": "detector-context jitter approximates proposals; no trained V12 detector existed at preparation time",
        "content_manifest": _project_relative(classifier_content),
        "content_manifest_sha256": classifier_content_hash,
    }
    _write_json(classifier_root / "dataset_metadata.json", classifier_metadata)
    return {
        "reused": False,
        "detector_root": str(detector_root),
        "classifier_root": str(classifier_root),
        "detector_content_manifest_sha256": detector_content_hash,
        "classifier_content_manifest_sha256": classifier_content_hash,
        "detector_split_counts": dict(detector_counts),
        "classifier_split_counts": dict(classifier_counts),
    }


def freeze_v12() -> dict[str, Any]:
    freeze_path = project_path(FREEZE)
    if freeze_path.exists():
        result = verify_freezes(include_v12=True)
        if not result["passed"]:
            raise ValueError("Existing V12 freeze does not verify")
        return result | {"reused": True}
    required = [
        project_path(MANIFEST),
        project_path(RELEASE_ROOT / "canonical_intake.csv"),
        project_path(RELEASE_ROOT / "quality_exclusions.csv"),
        project_path(RELEASE_ROOT / "derivation_report.json"),
        project_path(AUDIT_ROOT / "audit.json"),
        project_path(AUDIT_ROOT / "report.md"),
        project_path(POLICY),
        project_path("docs/LEGACY_REPLACEMENT_PRODUCT_DATA_CONTRACT_V3.md"),
        project_path("configs/recovery/recovery_v12_training_matrix.yaml"),
        project_path("src/roadsign_assist/datasets/recovery_v12.py"),
        project_path("src/roadsign_assist/recovery_training.py"),
        project_path("scripts/prepare_recovery_v12.py"),
        project_path("scripts/run_recovery_training_v12.ps1"),
        project_path("scripts/show_recovery_training_v12_status.ps1"),
        project_path(TRAINING_ROOT / "detector/data.yaml"),
        project_path(TRAINING_ROOT / "detector/dataset_metadata.json"),
        project_path(TRAINING_ROOT / "detector/content_manifest.json"),
        project_path(TRAINING_ROOT / "classifier/dataset_metadata.json"),
        project_path(TRAINING_ROOT / "classifier/labels.json"),
        project_path(TRAINING_ROOT / "classifier/crop_manifest.csv"),
        project_path(TRAINING_ROOT / "classifier/content_manifest.json"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze incomplete V12 preparation: {missing}")
    payload = {
        "schema_version": "1.0",
        "release_id": RELEASE_ID,
        "parent_release_id": PARENT_RELEASE_ID,
        "parent_manifest_sha256": PARENT_SHA256,
        "frozen_at": _utc_now(),
        "files": {str(path): _sha256(path) for path in required},
        "runtime_promotion_allowed": False,
        "phase_e2_present": False,
    }
    _write_json(freeze_path, payload)
    result = verify_freezes(include_v12=True)
    if not result["passed"]:
        raise RuntimeError("V12 freeze failed immediate verification")
    _write_json(project_path(AUDIT_ROOT / "freeze_verification.json"), result)
    return result | {"reused": False}


def verify_prepared_datasets() -> dict[str, Any]:
    root = project_path(TRAINING_ROOT)
    detector = root / "detector/content_manifest.json"
    classifier = root / "classifier/content_manifest.json"
    findings = _verify_dataset_content(detector) + _verify_dataset_content(classifier)
    return {
        "verified_at": _utc_now(),
        "passed": not findings,
        "findings": findings,
        "detector_content_manifest_sha256": _sha256(detector),
        "classifier_content_manifest_sha256": _sha256(classifier),
    }


__all__ = [
    "AUDIT_ROOT",
    "FREEZE",
    "MANIFEST",
    "PARENT_MANIFEST",
    "PARENT_RELEASE_ID",
    "PARENT_SHA256",
    "POLICY",
    "POLICY_ID",
    "RELEASE_ID",
    "RELEASE_ROOT",
    "TRAINING_ROOT",
    "audit_v12",
    "derive_v12",
    "freeze_v12",
    "prepare_datasets",
    "verify_freezes",
    "verify_prepared_datasets",
]
