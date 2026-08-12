from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_ROOT / "_archive/2026-08-12-pre-dvc-cleanup"
RELEASE_ID = "classifier_no_controlled_variants_20260812"
RELEASE_ROOT = PROJECT_ROOT / f"data/processed/{RELEASE_ID}"
RELEASE_MANIFEST = PROJECT_ROOT / "data/manifests/classifier_release.csv"
RELEASE_AUDIT = PROJECT_ROOT / f"outputs/audit/{RELEASE_ID}.json"
MANUAL_MANIFEST = PROJECT_ROOT / "data/manifests/stage_c_manual_01_candidates.csv"
OLD_RELEASE_MANIFEST = PROJECT_ROOT / "data/manifests/final_dataset.csv"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
REVIEW_OUTPUT = PROJECT_ROOT / "data/annotations/classifier_review_decisions.csv"
REJECTION_OUTPUT = PROJECT_ROOT / "data/annotations/rejected_controlled_variants.csv"
EXPECTED_CONTROLLED = 202
EXPECTED_RELEASE_SAMPLES = 2_976

STAGE_D_MANIFESTS = (
    PROJECT_ROOT / "data/manifests/stage_d_zero_gap_qc_manifest.csv",
    PROJECT_ROOT / "data/manifests/stage_d_other_zero_gap_qc_manifest.csv",
    PROJECT_ROOT / "data/manifests/stage_d_manual_pending_qc_manifest.csv",
    PROJECT_ROOT / "data/manifests/stage_d_steep_descent_qc_manifest.csv",
)

CONTROLLED_CONTACT_SHEETS = (
    "outputs/review/00_CURRENT_REVIEW/stage_c_manual_no_left_or_right_turn_contact_sheet.jpg",
    "outputs/review/00_CURRENT_REVIEW/stage_c_manual_no_straight_or_left_contact_sheet.jpg",
    "outputs/review/00_CURRENT_REVIEW/stage_c_manual_sound_horn_contact_sheet.jpg",
    "outputs/review/00_CURRENT_REVIEW/stage_c_manual_steep_descent_contact_sheet.jpg",
    "outputs/review/00_CURRENT_REVIEW/stage_c_manual_turn_left_or_right_contact_sheet.jpg",
    "outputs/review/00_CURRENT_REVIEW/stage_d_manual_pending_qc_contact_sheet.jpg",
)

RETAINED_MODELS = {
    "models/exported/experimental/emtd_segmenter_s30.onnx",
    "models/exported/experimental/stage_e_current_efficientnet_v2_s.labels.json",
    "models/exported/experimental/stage_e_current_efficientnet_v2_s_embedding_q97.calibration.json",
    "models/exported/experimental/stage_e_current_efficientnet_v2_s_embedding_q97.onnx",
}

RETAINED_MANIFESTS = {
    ".gitignore",
    "README.md",
    "assignment_external_test.csv",
    "classifier_release.csv",
    "CURRENT_DATA_PROGRESS.csv",
    "dataset_sources.json",
}

HISTORICAL_DOCS = {
    "ASSIGNMENT_MAPPING_AUDIT_REPORT.md",
    "GAP_FILL_ONLINE_MINING_HANDOFF.md",
    "STAGE_C_CHINA_REFERENCE_SOURCES_01_REPORT.md",
    "STAGE_C_GAP_FILL_01_TT100K_REPORT.md",
    "STAGE_C_GAP_FILL_02_PUBLIC_REAL_SOURCES_REPORT.md",
    "STAGE_C_GAP_FILL_03_TO_06_RARE_CLASS_MINING_REPORT.md",
    "STAGE_C_ONLINE_REFERENCE_SOURCES_01_REPORT.md",
    "STAGE_C_ROBOFLOW_IMPORT_REPORT.md",
    "STAGE_C_SPRINT_01_COMMONS_TOPUP_REPORT.md",
}

RELEASE_UNDER_MINIMUM_COUNTS = {
    "no_left_or_right_turn": 7,
    "no_straight_or_left": 9,
    "residential_area_ahead": 49,
    "side_road_right": 49,
    "sound_horn": 11,
    "steep_descent": 10,
    "turn_left_or_right": 11,
}


@dataclass(frozen=True)
class ControlledSelection:
    rows: list[dict[str, str]]
    crop_hashes: set[str]
    candidate_by_path: dict[Path, str]


@dataclass(frozen=True)
class ArchiveGroup:
    group_id: str
    source_root: Path
    archive_root: Path
    files: tuple[Path, ...]
    reason: str
    source_id: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def project_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(filesystem_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filesystem_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return f"\\\\?\\{resolved}"
    return resolved


def filesystem_is_file(path: Path) -> bool:
    return os.path.isfile(filesystem_path(path))


def filesystem_mkdir(path: Path) -> None:
    os.makedirs(filesystem_path(path), exist_ok=True)


def filesystem_move(source: Path, destination: Path) -> None:
    filesystem_mkdir(destination.parent)
    os.replace(filesystem_path(source), filesystem_path(destination))


def files_under(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    if path.is_file():
        return (path,)
    return tuple(sorted((item for item in path.rglob("*") if item.is_file()), key=str))


def validate_controlled_selection() -> ControlledSelection:
    manual_rows = read_csv(MANUAL_MANIFEST)
    controlled = [
        row
        for row in manual_rows
        if row.get("source_modality", "").startswith(
            "controlled_visual_augmentation_"
        )
    ]
    old_release_rows = read_csv(OLD_RELEASE_MANIFEST)
    frozen = [row for row in old_release_rows if row.get("is_controlled_variant") == "true"]
    manual_hashes = {row["crop_sha256"] for row in controlled}
    frozen_hashes = {row["crop_sha256"] for row in frozen}
    if len(controlled) != EXPECTED_CONTROLLED:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTROLLED} controlled source rows, found {len(controlled)}"
        )
    if len(frozen) != EXPECTED_CONTROLLED:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTROLLED} frozen controlled rows, found {len(frozen)}"
        )
    if len(manual_hashes) != EXPECTED_CONTROLLED or manual_hashes != frozen_hashes:
        raise RuntimeError("Controlled Stage-C and frozen release SHA-256 sets do not match")

    candidate_by_path: dict[Path, str] = {}
    for row in controlled:
        for field in ("local_original_path", "local_crop_path"):
            path = PROJECT_ROOT / row[field]
            if not path.is_file():
                raise FileNotFoundError(path)
            candidate_by_path[path.resolve()] = row["candidate_id"]
    if len(candidate_by_path) != EXPECTED_CONTROLLED * 2:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTROLLED * 2} paired files, found {len(candidate_by_path)}"
        )
    return ControlledSelection(controlled, manual_hashes, candidate_by_path)


def tree_digest(file_rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(file_rows, key=lambda value: str(value["original_path"])):
        line = (
            f"{row['original_path']}\0{row['size_bytes']}\0{row['sha256_before']}\n"
        )
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def load_existing_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def inventory_and_move(
    groups: Iterable[ArchiveGroup],
    *,
    candidate_by_path: dict[Path, str] | None = None,
) -> None:
    groups = tuple(groups)
    if not groups:
        return
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = ARCHIVE_ROOT / "archive_index.csv"
    file_manifest_path = ARCHIVE_ROOT / "archive_files.sha256.csv"
    index_rows: list[dict[str, object]] = load_existing_rows(index_path)
    file_rows: list[dict[str, object]] = load_existing_rows(file_manifest_path)
    existing_groups = {row["group_id"] for row in index_rows}
    requested_group_ids = {group.group_id for group in groups}
    new_groups = tuple(
        group for group in groups if group.group_id not in existing_groups and group.files
    )

    new_file_rows: list[dict[str, object]] = []
    new_index_rows: list[dict[str, object]] = []
    for group in new_groups:
        group_rows: list[dict[str, object]] = []
        for source in group.files:
            resolved = source.resolve()
            if group.source_root.is_file():
                relative = Path(group.source_root.name)
            else:
                relative = resolved.relative_to(group.source_root.resolve())
            destination = group.archive_root / relative
            if filesystem_is_file(destination):
                raise FileExistsError(destination)
            size = source.stat().st_size
            item = {
                "group_id": group.group_id,
                "original_path": project_rel(source),
                "archive_path": project_rel(destination),
                "reason": group.reason,
                "source_id": (
                    candidate_by_path.get(resolved, group.source_id)
                    if candidate_by_path
                    else group.source_id
                ),
                "size_bytes": size,
                "sha256_before": sha256_file(source),
                "sha256_after": "",
                "verified": "false",
            }
            group_rows.append(item)
            new_file_rows.append(item)
        new_index_rows.append(
            {
                "group_id": group.group_id,
                "original_path": project_rel(group.source_root),
                "archive_path": project_rel(group.archive_root),
                "reason": group.reason,
                "source_id": group.source_id,
                "file_count": len(group_rows),
                "size_bytes": sum(int(row["size_bytes"]) for row in group_rows),
                "tree_sha256": tree_digest(group_rows),
                "inventoried_at_utc": utc_now(),
                "verified_at_utc": "",
            }
        )

    index_fields = [
        "group_id",
        "original_path",
        "archive_path",
        "reason",
        "source_id",
        "file_count",
        "size_bytes",
        "tree_sha256",
        "inventoried_at_utc",
        "verified_at_utc",
    ]
    file_fields = [
        "group_id",
        "original_path",
        "archive_path",
        "reason",
        "source_id",
        "size_bytes",
        "sha256_before",
        "sha256_after",
        "verified",
    ]
    # Persist the complete pre-move inventory before changing any source path.
    combined_index_rows = index_rows + new_index_rows
    combined_file_rows = file_rows + new_file_rows
    if new_groups:
        write_csv(index_path, combined_index_rows, index_fields)
        write_csv(file_manifest_path, combined_file_rows, file_fields)

    rows_to_process = [
        row for row in combined_file_rows if row["group_id"] in requested_group_ids
    ]
    if not rows_to_process:
        return
    for row in rows_to_process:
        source = PROJECT_ROOT / str(row["original_path"])
        destination = PROJECT_ROOT / str(row["archive_path"])
        if filesystem_is_file(destination):
            continue
        if not filesystem_is_file(source):
            raise FileNotFoundError(
                f"Neither source nor archive file exists: {row['original_path']}"
            )
        filesystem_move(source, destination)

    verified_at = utc_now()
    for row in rows_to_process:
        destination = PROJECT_ROOT / str(row["archive_path"])
        after = sha256_file(destination)
        row["sha256_after"] = after
        row["verified"] = str(after == row["sha256_before"]).lower()
        if row["verified"] != "true":
            raise RuntimeError(f"Checksum mismatch after archive move: {destination}")
    for row in combined_index_rows:
        if row["group_id"] in requested_group_ids:
            row["verified_at_utc"] = verified_at
    write_csv(index_path, combined_index_rows, index_fields)
    write_csv(file_manifest_path, combined_file_rows, file_fields)


def update_review_decisions(selection: ControlledSelection) -> None:
    all_rows: list[dict[str, object]] = []
    fieldnames: list[str] = []
    rejected = 0
    expected_by_manifest = {
        "stage_d_manual_pending_qc_manifest.csv": 162,
        "stage_d_steep_descent_qc_manifest.csv": 40,
        "stage_d_other_zero_gap_qc_manifest.csv": 0,
        "stage_d_zero_gap_qc_manifest.csv": 0,
    }
    modality_by_hash = {row["crop_sha256"]: row["source_modality"] for row in selection.rows}
    for manifest in STAGE_D_MANIFESTS:
        rows = read_csv(manifest)
        if not rows:
            raise RuntimeError(f"Empty Stage-D manifest: {manifest}")
        fields = list(rows[0])
        for field in ("source_manifest", "source_modality"):
            if field not in fieldnames:
                fieldnames.append(field)
        for field in fields:
            if field not in fieldnames:
                fieldnames.append(field)
        matched = 0
        for row in rows:
            if row.get("crop_sha256") in selection.crop_hashes:
                decision = row.get("review_decision")
                if decision not in {"accept", "reject_synthetic_controlled_variant"}:
                    raise RuntimeError(
                        "Controlled row has an unexpected review decision before cleanup: "
                        f"{row.get('crop_path')} ({decision})"
                    )
                row["review_decision"] = "reject_synthetic_controlled_variant"
                row["review_basis"] = "pre_dvc_cleanup_real_reference_only_policy"
                row["stage_d_status"] = "rejected_synthetic_controlled_variant"
                note = row.get("review_notes", "").strip()
                reason = (
                    "Rejected 2026-08-12: controlled sign pasted onto generated/noisy "
                    "background; excluded from clean classifier release."
                )
                if reason not in note:
                    row["review_notes"] = f"{note} {reason}".strip()
                matched += 1
                rejected += 1
            row["source_manifest"] = project_rel(manifest)
            row["source_modality"] = modality_by_hash.get(
                row.get("crop_sha256", ""), row.get("source_dataset", "")
            )
            all_rows.append(row)
        write_csv(manifest, rows, list(rows[0]))
        expected = expected_by_manifest[manifest.name]
        if matched != expected:
            raise RuntimeError(
                f"Expected {expected} controlled rows in {manifest.name}, found {matched}"
            )
    if rejected != EXPECTED_CONTROLLED:
        raise RuntimeError(f"Expected {EXPECTED_CONTROLLED} rejected reviews, found {rejected}")
    write_csv(REVIEW_OUTPUT, all_rows, fieldnames)

    rejection_fields = [
        "candidate_id",
        "semantic_sign_id",
        "source_modality",
        "crop_sha256",
        "source_sha256",
        "archived_original_path",
        "archived_crop_path",
        "review_decision",
        "rejection_reason",
        "source_url",
        "license_notes",
    ]
    rejection_rows: list[dict[str, object]] = []
    for row in selection.rows:
        rejection_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "semantic_sign_id": row["semantic_sign_id"],
                "source_modality": row["source_modality"],
                "crop_sha256": row["crop_sha256"],
                "source_sha256": row["source_sha256"],
                "archived_original_path": (
                    f"_archive/2026-08-12-pre-dvc-cleanup/rejected_controlled_variants/"
                    f"{row['local_original_path']}"
                ),
                "archived_crop_path": (
                    f"_archive/2026-08-12-pre-dvc-cleanup/rejected_controlled_variants/"
                    f"{row['local_crop_path']}"
                ),
                "review_decision": "reject_synthetic_controlled_variant",
                "rejection_reason": (
                    "Controlled sign pasted onto generated/noisy background; excluded "
                    "from clean classifier release."
                ),
                "source_url": row.get("commons_page_url") or row.get("download_url", ""),
                "license_notes": row.get("license_short_name", ""),
            }
        )
    write_csv(REJECTION_OUTPUT, rejection_rows, rejection_fields)


def update_tracker() -> None:
    rows = read_csv(TRACKER_PATH)
    fields = list(rows[0])
    seen: set[str] = set()
    for row in rows:
        label = row["semantic_sign_id"]
        if label not in RELEASE_UNDER_MINIMUM_COUNTS:
            continue
        actual = RELEASE_UNDER_MINIMUM_COUNTS[label]
        minimum = int(row["minimum_clean_crops"])
        gap = minimum - actual
        row["realistic_candidate_total"] = str(actual)
        row["gap_to_minimum"] = str(gap)
        row["collection_status"] = "below_minimum_real_replacements_required"
        row["cleaning_status"] = "controlled_variants_rejected_20260812"
        row["next_action"] = (
            f"Collect {gap} real, class-correct replacements; controlled synthetic "
            f"background variants are archived and excluded from {RELEASE_ID}."
        )
        seen.add(label)
    if seen != set(RELEASE_UNDER_MINIMUM_COUNTS):
        raise RuntimeError(
            "Tracker labels missing: "
            f"{sorted(set(RELEASE_UNDER_MINIMUM_COUNTS) - seen)}"
        )
    write_csv(TRACKER_PATH, rows, fields)


def reject_controlled_variants() -> None:
    selection = validate_controlled_selection()
    update_review_decisions(selection)
    update_tracker()
    variant_files = tuple(sorted(selection.candidate_by_path, key=str))
    contact_files = tuple(
        PROJECT_ROOT / path
        for path in CONTROLLED_CONTACT_SHEETS
        if (PROJECT_ROOT / path).is_file()
    )
    groups = [
        ArchiveGroup(
            group_id="rejected_controlled_variants",
            source_root=PROJECT_ROOT,
            archive_root=ARCHIVE_ROOT / "rejected_controlled_variants",
            files=variant_files,
            reason=(
                "Controlled visual augmentation used generated/noisy backgrounds and is "
                "excluded from the clean classifier release."
            ),
            source_id="stage_c_manual_01",
        ),
        ArchiveGroup(
            group_id="controlled_variant_review_sheets",
            source_root=PROJECT_ROOT,
            archive_root=ARCHIVE_ROOT / "controlled_variant_review_sheets",
            files=contact_files,
            reason="Historical review sheets include the rejected controlled variants.",
            source_id="stage_c_manual_01",
        ),
    ]
    inventory_and_move(groups, candidate_by_path=selection.candidate_by_path)


def verify_release() -> None:
    if not RELEASE_MANIFEST.is_file() or not RELEASE_AUDIT.is_file():
        raise FileNotFoundError("Versioned classifier release or audit is missing")
    rows = read_csv(RELEASE_MANIFEST)
    controlled = sum(row.get("is_controlled_variant") == "true" for row in rows)
    if len(rows) != EXPECTED_RELEASE_SAMPLES or controlled:
        raise RuntimeError(
            f"Release verification failed: samples={len(rows)}, controlled={controlled}"
        )
    counts = Counter(row["semantic_sign_id"] for row in rows)
    if any(
        counts[label] != count
        for label, count in RELEASE_UNDER_MINIMUM_COUNTS.items()
    ):
        raise RuntimeError("Release under-minimum counts do not match the cleanup audit")
    for row in rows:
        image = PROJECT_ROOT / row["dataset_image_path"]
        if not image.is_file():
            raise FileNotFoundError(image)


def group_for_directory(
    group_id: str,
    source: str,
    archive_segment: str,
    reason: str,
    source_id: str,
) -> ArchiveGroup:
    source_path = PROJECT_ROOT / source
    return ArchiveGroup(
        group_id=group_id,
        source_root=source_path,
        archive_root=ARCHIVE_ROOT / archive_segment / source,
        files=files_under(source_path),
        reason=reason,
        source_id=source_id,
    )


def legacy_archive_groups() -> list[ArchiveGroup]:
    groups = [
        group_for_directory(
            "raw_research_sources",
            "data/raw",
            "raw_research_sources",
            "Original downloads and research sources are not required to retrain from prepared datasets.",
            "mixed_source_registry",
        ),
        group_for_directory(
            "staging_candidates",
            "data/staging",
            "staging_candidates",
            "Intermediate imports, probes, candidates, and rejected staging data.",
            "mixed_stage_c_candidates",
        ),
        group_for_directory(
            "legacy_classifier_release",
            "data/processed/stage_e_classifier_current",
            "legacy_datasets",
            "Old 3,178-sample synthetic-assisted classifier release.",
            "stage_e_current_20260702",
        ),
        group_for_directory(
            "obsolete_ocr_smoke_dataset",
            "data/processed/ocr_smoke",
            "legacy_datasets",
            "Generated OCR smoke data is reproducible and is not an active training dataset.",
            "ocr_smoke",
        ),
        group_for_directory(
            "obsolete_stage_f_negative_eval",
            "data/processed/stage_f_negative_eval",
            "legacy_datasets",
            "Historical negative-evaluation dataset; selected metrics are documented separately.",
            "stage_f_negative_eval",
        ),
        group_for_directory(
            "obsolete_generic_splits",
            "data/splits",
            "legacy_datasets",
            "Old generic split outputs; active prepared datasets contain their own splits.",
            "legacy_generic_splits",
        ),
    ]

    manifest_root = PROJECT_ROOT / "data/manifests"
    manifest_files = tuple(
        path
        for path in files_under(manifest_root)
        if path.name not in RETAINED_MANIFESTS
    )
    groups.append(
        ArchiveGroup(
            group_id="historical_manifests",
            source_root=manifest_root,
            archive_root=ARCHIVE_ROOT / "historical_manifests/data/manifests",
            files=manifest_files,
            reason="Historical, pending, per-split, and superseded manifests were consolidated.",
            source_id="mixed_historical_manifests",
        )
    )

    models_root = PROJECT_ROOT / "models"
    model_files = tuple(
        path
        for path in files_under(models_root)
        if project_rel(path) not in RETAINED_MODELS
        and not project_rel(path).startswith("models/ocr/")
    )
    groups.append(
        ArchiveGroup(
            group_id="unselected_models",
            source_root=models_root,
            archive_root=ARCHIVE_ROOT / "unselected_models/models",
            files=model_files,
            reason="Unselected checkpoints, baselines, exports, and preprocessing weights.",
            source_id="legacy_model_artifacts",
        )
    )

    outputs_root = PROJECT_ROOT / "outputs"
    output_files = tuple(
        path for path in files_under(outputs_root) if path.resolve() != RELEASE_AUDIT.resolve()
    )
    groups.append(
        ArchiveGroup(
            group_id="historical_outputs",
            source_root=outputs_root,
            archive_root=ARCHIVE_ROOT / "historical_outputs/outputs",
            files=output_files,
            reason="Bulky training, review, baseline, and obsolete evaluation outputs.",
            source_id="generated_outputs",
        )
    )

    docs_root = PROJECT_ROOT / "docs"
    doc_files = tuple(docs_root / name for name in sorted(HISTORICAL_DOCS) if (docs_root / name).is_file())
    groups.append(
        ArchiveGroup(
            group_id="historical_collection_reports",
            source_root=docs_root,
            archive_root=ARCHIVE_ROOT / "historical_docs/docs",
            files=doc_files,
            reason="Completed Stage-C collection/mining reports now summarized by the active ledger.",
            source_id="historical_collection_reports",
        )
    )
    return groups


def remove_empty_directories(paths: Iterable[Path]) -> None:
    for root in paths:
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(
            (item for item in root.rglob("*") if item.is_dir()), reverse=True
        ):
            with contextlib.suppress(OSError):
                path.rmdir()
        with contextlib.suppress(OSError):
            root.rmdir()


def archive_legacy_material() -> None:
    verify_release()
    groups = legacy_archive_groups()
    inventory_and_move(groups)
    remove_empty_directories(
        [
            PROJECT_ROOT / "data/raw",
            PROJECT_ROOT / "data/staging",
            PROJECT_ROOT / "data/splits",
            PROJECT_ROOT / "data/processed/stage_e_classifier_current",
            PROJECT_ROOT / "data/processed/ocr_smoke",
            PROJECT_ROOT / "data/processed/stage_f_negative_eval",
            PROJECT_ROOT / "outputs",
        ]
    )


def archive_obsolete_dvc_metadata() -> None:
    names = (
        "data/raw.dvc",
        "data/processed.dvc",
        "data/annotations.dvc",
        "data/official.dvc",
        "models.dvc",
        "dvc.yaml",
        "dvc.lock",
    )
    files = tuple(PROJECT_ROOT / name for name in names if (PROJECT_ROOT / name).is_file())
    inventory_and_move(
        [
            ArchiveGroup(
                group_id="obsolete_broad_dvc_metadata",
                source_root=PROJECT_ROOT,
                archive_root=ARCHIVE_ROOT / "obsolete_dvc_metadata",
                files=files,
                reason=(
                    "Broad raw/processed/models pointers and the obsolete preprocessing "
                    "pipeline were replaced by targeted active-artifact pointers."
                ),
                source_id="legacy_dvc_configuration",
            )
        ]
    )


def preflight() -> None:
    selection = validate_controlled_selection()
    summary = {
        "archive_root": project_rel(ARCHIVE_ROOT),
        "controlled_rows": len(selection.rows),
        "paired_files": len(selection.candidate_by_path),
        "retained_release_samples": EXPECTED_RELEASE_SAMPLES,
        "under_minimum_class_counts": RELEASE_UNDER_MINIMUM_COUNTS,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the recoverable 2026-08-12 pre-DVC cleanup."
    )
    parser.add_argument(
        "--phase",
        choices=(
            "preflight",
            "reject-controlled",
            "refresh-tracker",
            "archive-legacy",
            "archive-obsolete-dvc",
        ),
        default="preflight",
    )
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight()
    elif args.phase == "reject-controlled":
        reject_controlled_variants()
    elif args.phase == "refresh-tracker":
        update_tracker()
    elif args.phase == "archive-obsolete-dvc":
        archive_obsolete_dvc_metadata()
    else:
        archive_legacy_material()


if __name__ == "__main__":
    main()
