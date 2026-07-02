from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 2513
DATASET_ID = "stage_e_current_20260702"

STAGE_D_MANIFESTS = (
    PROJECT_ROOT / "data/manifests/stage_d_zero_gap_qc_manifest.csv",
    PROJECT_ROOT / "data/manifests/stage_d_other_zero_gap_qc_manifest.csv",
    PROJECT_ROOT / "data/manifests/stage_d_manual_pending_qc_manifest.csv",
    PROJECT_ROOT / "data/manifests/stage_d_steep_descent_qc_manifest.csv",
)

FINAL_DATASET_MANIFEST = PROJECT_ROOT / "data/manifests/final_dataset.csv"
FINAL_TRAIN_MANIFEST = PROJECT_ROOT / "data/manifests/final_train.csv"
FINAL_VALIDATION_MANIFEST = PROJECT_ROOT / "data/manifests/final_validation.csv"
FINAL_TEST_MANIFEST = PROJECT_ROOT / "data/manifests/final_test.csv"
ASSIGNMENT_EXTERNAL_TEST_MANIFEST = PROJECT_ROOT / "data/manifests/assignment_external_test.csv"
FINAL_SPLIT_AUDIT = PROJECT_ROOT / "outputs/audit/final_split_audit.json"
CLASSIFIER_ROOT = PROJECT_ROOT / "data/processed/stage_e_classifier_current"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
COURSEWORK_MANIFEST = PROJECT_ROOT / "data/manifests/coursework_manifest.csv"
OFFICIAL_IMAGES_MANIFEST = PROJECT_ROOT / "data/manifests/official_images.csv"

IMAGE_SUFFIX_FALLBACK = ".jpg"


@dataclass(frozen=True)
class FrozenSample:
    source_row: dict[str, str]
    manifest_path: Path
    source_crop_path: Path
    sample_id: str
    leakage_group: str
    base_variant_group: str
    is_controlled_variant: bool


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def project_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def project_path(value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_reset_directory(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to reset outside project root: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def stable_fraction(value: str, seed: int = SEED) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64)


def slug(value: str) -> str:
    value = value.encode("ascii", errors="ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "sample"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_variant_group(row: dict[str, str], crop_path: Path) -> str:
    text = f"{crop_path.name} {row.get('source_label', '')} {row.get('source_file_url', '')}"
    match = re.search(r"controlled_from_(MAN01-[A-Z]+-\d{4})", text)
    if match:
        return f"{row['semantic_sign_id']}::{match.group(1)}"
    match = re.search(r"(MAN01-[A-Z]+-\d{4})", crop_path.name)
    if match:
        return f"{row['semantic_sign_id']}::{match.group(1)}"
    if row.get("source_url", "").strip():
        return f"{row['semantic_sign_id']}::{row['source_url'].strip()}"
    return f"{row['semantic_sign_id']}::{row.get('dedupe_key', '').strip() or crop_path.stem}"


def leakage_group(row: dict[str, str], crop_path: Path) -> str:
    dedupe = row.get("dedupe_key", "").strip()
    if not dedupe:
        dedupe = row.get("crop_sha256", "").strip() or crop_path.stem
    return f"{row.get('source_group', 'unknown')}::{row['semantic_sign_id']}::{dedupe}"


def load_stage_d_samples() -> tuple[list[FrozenSample], list[dict[str, str]]]:
    samples: list[FrozenSample] = []
    skipped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    per_label_counter: Counter[str] = Counter()
    for manifest_path in STAGE_D_MANIFESTS:
        for row in read_csv(manifest_path):
            if row.get("review_decision", "").strip() != "accept":
                skipped.append(
                    {
                        "reason": "review_decision_not_accept",
                        "manifest": project_rel(manifest_path),
                        "crop_path": row.get("crop_path", ""),
                    }
                )
                continue
            label = row.get("semantic_sign_id", "").strip()
            crop_text = row.get("crop_path", "").strip()
            if not label or not crop_text:
                skipped.append(
                    {
                        "reason": "missing_label_or_crop_path",
                        "manifest": project_rel(manifest_path),
                        "crop_path": crop_text,
                    }
                )
                continue
            crop_path = project_path(crop_text)
            if not crop_path.exists():
                skipped.append(
                    {
                        "reason": "crop_path_missing",
                        "manifest": project_rel(manifest_path),
                        "crop_path": crop_text,
                    }
                )
                continue
            dedupe_key = row.get("crop_sha256", "").strip() or sha256_file(crop_path)
            identity = (label, dedupe_key)
            if identity in seen:
                skipped.append(
                    {
                        "reason": "duplicate_label_crop_sha256",
                        "manifest": project_rel(manifest_path),
                        "crop_path": crop_text,
                    }
                )
                continue
            seen.add(identity)
            per_label_counter[label] += 1
            sample_id = f"stagee_{slug(label)}_{per_label_counter[label]:05d}_{dedupe_key[:10]}"
            modality_text = " ".join(
                [
                    row.get("source_dataset", ""),
                    row.get("source_label", ""),
                    row.get("source_group", ""),
                    crop_path.name,
                ]
            ).casefold()
            is_controlled = "controlled" in modality_text or "variant" in modality_text
            samples.append(
                FrozenSample(
                    source_row=row,
                    manifest_path=manifest_path,
                    source_crop_path=crop_path,
                    sample_id=sample_id,
                    leakage_group=leakage_group(row, crop_path),
                    base_variant_group=base_variant_group(row, crop_path),
                    is_controlled_variant=is_controlled,
                )
            )
    samples.sort(key=lambda sample: (sample.source_row["semantic_sign_id"], sample.sample_id))
    return samples, skipped


def target_counts(total: int) -> dict[str, int]:
    if total <= 1:
        return {"train": total, "validation": 0, "test": 0}
    if total == 2:
        return {"train": 1, "validation": 1, "test": 0}
    validation = max(1, round(total * 0.15))
    test = max(1, round(total * 0.15))
    train = max(1, total - validation - test)
    while train + validation + test > total:
        train = max(1, train - 1)
    while train + validation + test < total:
        train += 1
    return {"train": train, "validation": validation, "test": test}


def split_label_samples(samples: list[FrozenSample]) -> dict[str, str]:
    label = samples[0].source_row["semantic_sign_id"]
    grouped: dict[str, list[FrozenSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.leakage_group].append(sample)

    groups = sorted(
        grouped,
        key=lambda group: (
            stable_fraction(f"{label}:{group}"),
            group,
        ),
    )
    counts = target_counts(len(samples))
    assignments: dict[str, str] = {}
    split_sizes = {"train": 0, "validation": 0, "test": 0}
    mandatory = [split for split, count in counts.items() if count > 0]
    for split, group in zip(mandatory, groups, strict=False):
        assignments[group] = split
        split_sizes[split] += len(grouped[group])

    for group in groups[len(assignments) :]:
        group_size = len(grouped[group])

        def score(
            split: str,
            *,
            current_group: str = group,
            current_size: int = group_size,
        ) -> tuple[float, float, float, str]:
            target = max(1, counts[split])
            current = split_sizes[split]
            projected = current + current_size
            is_over_target = 1.0 if current >= target else 0.0
            return (
                is_over_target,
                projected / target,
                stable_fraction(f"{label}:{current_group}:{split}"),
                split,
            )

        allowed = [split for split, count in counts.items() if count > 0]
        chosen = min(allowed, key=score)
        assignments[group] = chosen
        split_sizes[chosen] += group_size
    return assignments


def split_samples(samples: list[FrozenSample]) -> dict[str, list[FrozenSample]]:
    by_label: dict[str, list[FrozenSample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.source_row["semantic_sign_id"]].append(sample)
    split_rows: dict[str, list[FrozenSample]] = {"train": [], "validation": [], "test": []}
    for label, label_samples in sorted(by_label.items()):
        label_samples.sort(key=lambda sample: stable_fraction(f"{label}:{sample.sample_id}"))
        assignments = split_label_samples(label_samples)
        for sample in label_samples:
            split_rows[assignments[sample.leakage_group]].append(sample)
    for rows in split_rows.values():
        rows.sort(key=lambda sample: (sample.source_row["semantic_sign_id"], sample.sample_id))
    return split_rows


def copy_sample_to_dataset(sample: FrozenSample, split: str) -> Path:
    label = sample.source_row["semantic_sign_id"]
    suffix = sample.source_crop_path.suffix.lower() or IMAGE_SUFFIX_FALLBACK
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        suffix = IMAGE_SUFFIX_FALLBACK
    destination = CLASSIFIER_ROOT / split / label / f"{sample.sample_id}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.source_crop_path, destination)
    return destination


def manifest_row(sample: FrozenSample, split: str, dataset_image_path: Path) -> dict[str, str]:
    row = sample.source_row
    return {
        "stage_e_dataset_id": DATASET_ID,
        "sample_id": sample.sample_id,
        "split": split,
        "semantic_sign_id": row.get("semantic_sign_id", ""),
        "name_en": row.get("name_en", ""),
        "priority": row.get("priority", ""),
        "required_for": row.get("required_for", ""),
        "source_crop_path": project_rel(sample.source_crop_path),
        "dataset_image_path": project_rel(dataset_image_path),
        "crop_sha256": row.get("crop_sha256", ""),
        "crop_width": row.get("crop_width", ""),
        "crop_height": row.get("crop_height", ""),
        "stage_d_manifest": project_rel(sample.manifest_path),
        "stage_d_batch": row.get("stage_d_batch", ""),
        "stage_d_status": row.get("stage_d_status", ""),
        "selected_index": row.get("selected_index", ""),
        "review_decision": row.get("review_decision", ""),
        "source_group": row.get("source_group", ""),
        "source_dataset": row.get("source_dataset", ""),
        "source_split": row.get("source_split", ""),
        "source_label": row.get("source_label", ""),
        "source_url": row.get("source_url", ""),
        "license_notes": row.get("license_notes", ""),
        "mapping_evidence": row.get("mapping_evidence", ""),
        "prior_review_status": row.get("prior_review_status", ""),
        "dedupe_key": row.get("dedupe_key", ""),
        "leakage_group": sample.leakage_group,
        "base_variant_group": sample.base_variant_group,
        "is_controlled_variant": "true" if sample.is_controlled_variant else "false",
        "is_synthetic": "false",
        "annotation_status": "stage_d_minimum_qc_accepted",
        "review_notes": "Frozen into Stage E current classifier dataset.",
    }


STAGE_E_FIELDNAMES = [
    "stage_e_dataset_id",
    "sample_id",
    "split",
    "semantic_sign_id",
    "name_en",
    "priority",
    "required_for",
    "source_crop_path",
    "dataset_image_path",
    "crop_sha256",
    "crop_width",
    "crop_height",
    "stage_d_manifest",
    "stage_d_batch",
    "stage_d_status",
    "selected_index",
    "review_decision",
    "source_group",
    "source_dataset",
    "source_split",
    "source_label",
    "source_url",
    "license_notes",
    "mapping_evidence",
    "prior_review_status",
    "dedupe_key",
    "leakage_group",
    "base_variant_group",
    "is_controlled_variant",
    "is_synthetic",
    "annotation_status",
    "review_notes",
]


def write_assignment_external_test() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    official_by_id = {row["image_id"]: row for row in read_csv(OFFICIAL_IMAGES_MANIFEST)}
    rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for row in read_csv(COURSEWORK_MANIFEST):
        official = official_by_id.get(row["image_id"], {})
        image_path = PROJECT_ROOT / "data/official/assignment_images" / row["relative_path"]
        if not image_path.exists():
            missing.append({"image_id": row["image_id"], "path": project_rel(image_path)})
        rows.append(
            {
                "stage_e_dataset_id": DATASET_ID,
                "external_test_id": row["image_id"],
                "path": project_rel(image_path),
                "semantic_sign_id": row.get("semantic_sign_id", ""),
                "verified_coursework_id": row.get("verified_coursework_id", ""),
                "coursework_id_candidate": row.get("coursework_id_candidate", ""),
                "parameter_value": row.get("parameter_value", ""),
                "review_status": row.get("review_status", ""),
                "confidence": row.get("confidence", ""),
                "width": official.get("width", ""),
                "height": official.get("height", ""),
                "sha256": official.get("sha256", ""),
                "role": "external_assignment_acceptance_test_only",
                "included_in_training": "false",
            }
        )
    write_csv(
        ASSIGNMENT_EXTERNAL_TEST_MANIFEST,
        rows,
        [
            "stage_e_dataset_id",
            "external_test_id",
            "path",
            "semantic_sign_id",
            "verified_coursework_id",
            "coursework_id_candidate",
            "parameter_value",
            "review_status",
            "confidence",
            "width",
            "height",
            "sha256",
            "role",
            "included_in_training",
        ],
    )
    return rows, missing


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["semantic_sign_id"] for row in rows)
    return dict(sorted(counts.items()))


def nested_label_counts(split_rows: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, int]]:
    return {split: label_counts(rows) for split, rows in split_rows.items()}


def leakage_crossings(rows: list[dict[str, str]], key: str) -> dict[str, list[str]]:
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_by_group[row[key]].add(row["split"])
    return {
        group: sorted(splits)
        for group, splits in sorted(split_by_group.items())
        if len(splits) > 1
    }


def write_classifier_metadata(
    labels: list[str],
    all_rows: list[dict[str, str]],
    split_rows: dict[str, list[dict[str, str]]],
    audit: dict[str, Any],
) -> None:
    metadata = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_stage": "Stage E current classifier freeze",
        "annotation_status": "approved",
        "training_scope": (
            "current_stage_e_classifier_training; Stage D accepted crops only; "
            "coursework assignment images excluded"
        ),
        "coursework_images_included": 0,
        "assignment_external_test_manifest": project_rel(ASSIGNMENT_EXTERNAL_TEST_MANIFEST),
        "split_strategy": (
            "per-class deterministic stratified split using Stage D crop-level dedupe groups; "
            "controlled variant base crossings are audited separately"
        ),
        "seed": SEED,
        "labels": len(labels),
        "samples": len(all_rows),
        "split_samples": {split: len(rows) for split, rows in split_rows.items()},
        "split_label_counts": nested_label_counts(split_rows),
        "controlled_variant_samples": sum(
            1 for row in all_rows if row["is_controlled_variant"] == "true"
        ),
        "stage_d_manifests": [project_rel(path) for path in STAGE_D_MANIFESTS],
        "audit_path": project_rel(FINAL_SPLIT_AUDIT),
        "limitations": audit["limitations"],
    }
    (CLASSIFIER_ROOT / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (CLASSIFIER_ROOT / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n",
        encoding="utf-8",
    )


def target_class_gaps(labels: set[str]) -> dict[str, Any]:
    tracker_rows = read_csv(TRACKER_PATH)
    priority_counts: dict[str, dict[str, int]] = {}
    missing_with_no_samples: list[dict[str, str]] = []
    for row in tracker_rows:
        priority = row.get("priority", "")
        priority_counts.setdefault(priority, {"target_classes": 0, "included_labels": 0})
        priority_counts[priority]["target_classes"] += 1
        if row["semantic_sign_id"] in labels:
            priority_counts[priority]["included_labels"] += 1
        else:
            missing_with_no_samples.append(
                {
                    "semantic_sign_id": row["semantic_sign_id"],
                    "priority": priority,
                    "gap_to_minimum": row.get("gap_to_minimum", ""),
                    "collection_status": row.get("collection_status", ""),
                }
            )
    return {
        "priority_counts": priority_counts,
        "target_classes_without_stage_e_samples": missing_with_no_samples,
    }


def write_audit(
    all_rows: list[dict[str, str]],
    split_rows: dict[str, list[dict[str, str]]],
    skipped: list[dict[str, str]],
    assignment_rows: list[dict[str, str]],
    assignment_missing: list[dict[str, str]],
) -> dict[str, Any]:
    labels = sorted({row["semantic_sign_id"] for row in all_rows})
    split_label_counts = nested_label_counts(split_rows)
    missing_by_split = {
        split: sorted(set(labels) - set(counts))
        for split, counts in split_label_counts.items()
    }
    strict_leakage_crossings = leakage_crossings(all_rows, "leakage_group")
    base_variant_crossings = leakage_crossings(all_rows, "base_variant_group")
    controlled_base_crossings = {
        group: splits
        for group, splits in base_variant_crossings.items()
        if any(
            row["base_variant_group"] == group and row["is_controlled_variant"] == "true"
            for row in all_rows
        )
    }
    limitations: list[str] = []
    if controlled_base_crossings:
        limitations.append(
            "Controlled visual variants from the same base sign may cross splits in this "
            "current training freeze; exact Stage D crop-level dedupe groups do not cross."
        )
    below = target_class_gaps(set(labels))
    if below["target_classes_without_stage_e_samples"]:
        limitations.append(
            "Some target should/optional classes have no Stage E samples and are ignored "
            "for this current training dataset."
        )
    audit: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "source_stage_d_manifests": [project_rel(path) for path in STAGE_D_MANIFESTS],
        "outputs": {
            "final_dataset": project_rel(FINAL_DATASET_MANIFEST),
            "final_train": project_rel(FINAL_TRAIN_MANIFEST),
            "final_validation": project_rel(FINAL_VALIDATION_MANIFEST),
            "final_test": project_rel(FINAL_TEST_MANIFEST),
            "assignment_external_test": project_rel(ASSIGNMENT_EXTERNAL_TEST_MANIFEST),
            "classifier_folder_dataset": project_rel(CLASSIFIER_ROOT),
        },
        "samples": len(all_rows),
        "labels": len(labels),
        "split_samples": {split: len(rows) for split, rows in split_rows.items()},
        "split_label_counts": split_label_counts,
        "missing_labels_by_split": missing_by_split,
        "class_counts": label_counts(all_rows),
        "priority_counts": below["priority_counts"],
        "target_classes_without_stage_e_samples": below["target_classes_without_stage_e_samples"],
        "controlled_variant_samples": sum(
            1 for row in all_rows if row["is_controlled_variant"] == "true"
        ),
        "strict_leakage_group_cross_split_count": len(strict_leakage_crossings),
        "strict_leakage_group_crossings": strict_leakage_crossings,
        "controlled_base_group_cross_split_count": len(controlled_base_crossings),
        "controlled_base_group_crossing_examples": dict(
            list(controlled_base_crossings.items())[:40]
        ),
        "coursework_images_in_training": 0,
        "assignment_external_test_samples": len(assignment_rows),
        "assignment_external_test_missing_files": assignment_missing,
        "skipped_stage_d_rows": skipped,
        "limitations": limitations,
        "completion_checks": {
            "no_assignment_training_leakage": True,
            "no_strict_dedupe_group_cross_split": len(strict_leakage_crossings) == 0,
            "all_included_labels_in_train": not missing_by_split["train"],
            "all_included_labels_in_validation": not missing_by_split["validation"],
            "all_must_labels_in_train_and_validation": all(
                row["semantic_sign_id"] not in missing_by_split["train"]
                and row["semantic_sign_id"] not in missing_by_split["validation"]
                for row in read_csv(TRACKER_PATH)
                if row.get("priority") == "must"
                and row["semantic_sign_id"] in set(labels)
            ),
        },
    }
    FINAL_SPLIT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_SPLIT_AUDIT.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def freeze_stage_e() -> dict[str, Any]:
    safe_reset_directory(CLASSIFIER_ROOT)
    samples, skipped = load_stage_d_samples()
    if not samples:
        raise RuntimeError("No Stage D accepted samples were found for Stage E")
    split_samples_by_obj = split_samples(samples)

    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    for split, split_samples_for_name in split_samples_by_obj.items():
        for sample in split_samples_for_name:
            dataset_image_path = copy_sample_to_dataset(sample, split)
            split_rows[split].append(manifest_row(sample, split, dataset_image_path))

    all_rows = [row for split in ("train", "validation", "test") for row in split_rows[split]]
    all_rows.sort(key=lambda row: (row["semantic_sign_id"], row["sample_id"]))
    for rows in split_rows.values():
        rows.sort(key=lambda row: (row["semantic_sign_id"], row["sample_id"]))

    write_csv(FINAL_DATASET_MANIFEST, all_rows, STAGE_E_FIELDNAMES)
    write_csv(FINAL_TRAIN_MANIFEST, split_rows["train"], STAGE_E_FIELDNAMES)
    write_csv(FINAL_VALIDATION_MANIFEST, split_rows["validation"], STAGE_E_FIELDNAMES)
    write_csv(FINAL_TEST_MANIFEST, split_rows["test"], STAGE_E_FIELDNAMES)
    assignment_rows, assignment_missing = write_assignment_external_test()
    audit = write_audit(all_rows, split_rows, skipped, assignment_rows, assignment_missing)
    labels = sorted({row["semantic_sign_id"] for row in all_rows})
    write_classifier_metadata(labels, all_rows, split_rows, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the current Stage D accepted crops into Stage E classifier splits."
    )
    parser.parse_args()
    audit = freeze_stage_e()
    print(f"Stage E dataset: {audit['dataset_id']}")
    print(f"Samples: {audit['samples']} across {audit['labels']} labels")
    print(f"Splits: {audit['split_samples']}")
    print(f"Final train: {project_rel(FINAL_TRAIN_MANIFEST)}")
    print(f"Final validation: {project_rel(FINAL_VALIDATION_MANIFEST)}")
    print(f"Final test: {project_rel(FINAL_TEST_MANIFEST)}")
    print(f"Classifier folder: {project_rel(CLASSIFIER_ROOT)}")
    print(f"Audit: {project_rel(FINAL_SPLIT_AUDIT)}")


if __name__ == "__main__":
    main()
