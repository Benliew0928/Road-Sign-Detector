from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from roadsign_assist.config import load_yaml
from roadsign_assist.paths import DATA_ROOT, project_path

REQUIRED_COLUMNS = {
    "sample_id",
    "source_id",
    "path",
    "semantic_sign_id",
    "group_id",
    "route_id",
    "session_id",
    "physical_sign_id",
    "has_sign",
    "is_synthetic",
    "annotation_status",
}


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64)


def _effective_group(row: dict[str, str]) -> str:
    fields = [
        row.get("physical_sign_id", "").strip(),
        row.get("session_id", "").strip(),
        row.get("route_id", "").strip(),
        row.get("group_id", "").strip(),
    ]
    value = next((field for field in fields if field), "")
    if not value:
        raise ValueError(f"Sample {row['sample_id']} has no leakage-control group")
    return f"{row['source_id']}::{value}"


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _write_split(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _semantic_labels(row: dict[str, str]) -> set[str]:
    labels = {value for value in row.get("semantic_sign_id", "").split() if value}
    return labels or {"__unlabelled__"}


def _stratified_group_assignments(
    rows: list[dict[str, str]],
    *,
    seed: int,
    fractions: dict[str, float],
) -> dict[str, str]:
    rows_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_group[_effective_group(row)].append(row)

    group_label_counts: dict[str, Counter[str]] = {}
    total_label_counts: Counter[str] = Counter()
    for group, group_rows in rows_by_group.items():
        counts: Counter[str] = Counter()
        for row in group_rows:
            counts.update(_semantic_labels(row))
        group_label_counts[group] = counts
        total_label_counts.update(counts)

    target_rows = {split: len(rows) * fraction for split, fraction in fractions.items()}
    target_labels = {
        split: {label: count * fractions[split] for label, count in total_label_counts.items()}
        for split in fractions
    }
    assigned_rows = dict.fromkeys(fractions, 0)
    assigned_labels = {split: Counter[str]() for split in fractions}

    def group_order(group: str) -> tuple[float, int, int, float]:
        labels = group_label_counts[group]
        rarity = min(total_label_counts[label] for label in labels)
        return (
            float(rarity),
            -len(labels),
            -len(rows_by_group[group]),
            _stable_fraction(group, seed),
        )

    def assignment_score(group: str, candidate: str) -> float:
        score = 0.0
        group_size = len(rows_by_group[group])
        for split in fractions:
            row_count = assigned_rows[split] + (group_size if split == candidate else 0)
            row_target = max(target_rows[split], 1.0)
            score += ((row_count - target_rows[split]) / row_target) ** 2
            for label, total in total_label_counts.items():
                label_count = assigned_labels[split][label]
                if split == candidate:
                    label_count += group_label_counts[group][label]
                label_target = max(target_labels[split][label], 1.0)
                weight = 1.0 / max(1.0, total**0.5)
                score += weight * ((label_count - target_labels[split][label]) / label_target) ** 2
        return score

    assignments: dict[str, str] = {}
    ordered_groups = sorted(rows_by_group, key=group_order)
    for index, group in enumerate(ordered_groups):
        empty_splits = [split for split, count in assigned_rows.items() if count == 0]
        remaining = len(ordered_groups) - index
        candidates = (
            empty_splits if empty_splits and remaining <= len(empty_splits) else list(fractions)
        )
        split = min(
            candidates,
            key=lambda candidate: (
                assignment_score(group, candidate),
                _stable_fraction(f"{group}:{candidate}", seed),
                candidate,
            ),
        )
        assignments[group] = split
        assigned_rows[split] += len(rows_by_group[group])
        assigned_labels[split].update(group_label_counts[group])
    return assignments


def _write_split_diagnostics(
    path: Path,
    split_rows: dict[str, list[dict[str, str]]],
) -> None:
    payload: dict[str, object] = {"schema_version": "1.0", "splits": {}}
    split_payload: dict[str, object] = {}
    for split, rows in split_rows.items():
        labels: Counter[str] = Counter()
        for row in rows:
            labels.update(_semantic_labels(row))
        split_payload[split] = {
            "samples": len(rows),
            "groups": len({row["effective_group"] for row in rows}),
            "label_counts": dict(sorted(labels.items())),
        }
    payload["splits"] = split_payload
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_grouped_splits(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[dict[str, str]]]:
    fieldnames, rows = _read_manifest(project_path(manifest_path))
    missing = REQUIRED_COLUMNS - set(fieldnames)
    if missing:
        raise ValueError(f"Dataset manifest is missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError("Dataset manifest is empty")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id must be unique")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must leave a non-empty test fraction")

    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": 1.0 - train_fraction - validation_fraction,
    }
    assignments = _stratified_group_assignments(
        rows,
        seed=seed,
        fractions=fractions,
    )

    split_rows: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    group_assignments: dict[str, str] = {}
    for row in rows:
        effective_group = _effective_group(row)
        split = assignments[effective_group]
        previous = group_assignments.setdefault(effective_group, split)
        if previous != split:
            raise AssertionError("A leakage-control group was assigned to multiple splits")
        split_rows[split].append(
            {
                **row,
                "effective_group": effective_group,
                "split": split,
            }
        )

    output = project_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    output_fields = [*fieldnames, "effective_group", "split"]
    for split in ("train", "validation", "test"):
        split_rows[split].sort(key=lambda row: row["sample_id"])
        _write_split(output / f"{split}.csv", output_fields, split_rows[split])
    _write_split_diagnostics(output / "diagnostics.json", split_rows)
    return split_rows


def build_default_splits() -> dict[str, list[dict[str, str]]]:
    params = load_yaml("params.yaml")["data"]
    return build_grouped_splits(
        DATA_ROOT / "manifests" / "dataset.csv",
        DATA_ROOT / "splits",
        seed=int(params["split_seed"]),
        train_fraction=float(params["train_fraction"]),
        validation_fraction=float(params["validation_fraction"]),
    )
