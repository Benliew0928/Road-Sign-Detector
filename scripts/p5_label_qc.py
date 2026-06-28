from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from roadsign_assist.datasets.contact_sheet import ReviewTile, render_contact_sheet
from roadsign_assist.paths import project_path
from roadsign_assist.catalogue.repository import catalogue_by_id


CLASSIFICATION_ROOT = project_path("data/processed/emtd_classification")
ANNOTATIONS_PATH = project_path("data/annotations/emtd_boxes.csv")
CONTACT_SHEET_ROOT = project_path("outputs/review/p5_class_contact_sheets")
MANIFEST_PATH = project_path("data/manifests/p5_label_qc_manifest.csv")
CLASS_REVIEW_PATH = project_path("data/manifests/p5_class_review.csv")
CORRECTIONS_PATH = project_path("data/manifests/p5_label_corrections.csv")
REPORT_PATH = project_path("outputs/audit/p5_label_qc_report.json")
RARE_THRESHOLD = 5
TINY_EDGE_THRESHOLD = 32


def _read_annotations() -> dict[str, dict[str, str]]:
    if not ANNOTATIONS_PATH.exists():
        return {}
    with ANNOTATIONS_PATH.open(encoding="utf-8", newline="") as handle:
        return {row["instance_id"]: row for row in csv.DictReader(handle)}


def _crop_rows(root: Path = CLASSIFICATION_ROOT) -> list[dict[str, object]]:
    annotations = _read_annotations()
    rows: list[dict[str, object]] = []
    for split_root in sorted(path for path in root.iterdir() if path.is_dir()):
        split = split_root.name
        for label_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
            label = label_root.name
            for path in sorted(label_root.glob("*.jpg")):
                with Image.open(path) as image:
                    width, height = image.size
                annotation = annotations.get(path.stem, {})
                rows.append(
                    {
                        "split": split,
                        "current_label": label,
                        "file": path.relative_to(project_path(".")).as_posix(),
                        "instance_id": path.stem,
                        "source_class_id": annotation.get("class_id", ""),
                        "filename": annotation.get("filename", ""),
                        "width": width,
                        "height": height,
                        "area": width * height,
                        "tiny_crop": str(width < TINY_EDGE_THRESHOLD or height < TINY_EDGE_THRESHOLD).lower(),
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            raise ValueError(f"Cannot infer fieldnames for empty CSV: {path}")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_contact_sheets(rows: list[dict[str, object]]) -> dict[str, str]:
    if CONTACT_SHEET_ROOT.exists():
        shutil.rmtree(CONTACT_SHEET_ROOT)
    CONTACT_SHEET_ROOT.mkdir(parents=True, exist_ok=True)
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["current_label"])].append(row)

    outputs: dict[str, str] = {}
    summary_tiles: list[ReviewTile] = []
    for label, label_rows in sorted(by_label.items()):
        tiles = [
            ReviewTile(
                label=f"{row['split']} {row['instance_id']} c{row['source_class_id']}",
                image_path=project_path(str(row["file"])),
            )
            for row in label_rows
        ]
        output = CONTACT_SHEET_ROOT / f"{label}.jpg"
        render_contact_sheet(tiles, output, columns=6, tile_width=190, tile_height=180)
        outputs[label] = output.relative_to(project_path(".")).as_posix()
        summary_tiles.append(
            ReviewTile(
                label=f"{label} ({len(label_rows)})",
                image_path=project_path(str(label_rows[0]["file"])),
            )
        )

    if summary_tiles:
        summary_output = CONTACT_SHEET_ROOT / "_class_representatives.jpg"
        render_contact_sheet(summary_tiles, summary_output, columns=6, tile_width=210, tile_height=190)
        outputs["_class_representatives"] = summary_output.relative_to(project_path(".")).as_posix()
    return outputs


def _seed_corrections(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    corrections: list[dict[str, object]] = []
    for row in rows:
        if row["current_label"] == "animal_crossing" and str(row["source_class_id"]) == "35":
            corrections.append(
                {
                    "action": "relabel",
                    "status": "approved",
                    "split": row["split"],
                    "current_label": row["current_label"],
                    "new_label": "camera_enforcement",
                    "file": row["file"],
                    "instance_id": row["instance_id"],
                    "source_class_id": row["source_class_id"],
                    "reason": "Owner-identified P5 issue: EMTD animal_crossing crops are camera enforcement signs.",
                    "reviewer": "owner",
                }
            )
    return corrections


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_report(
    rows: list[dict[str, object]],
    *,
    contact_sheets: dict[str, str],
    corrections_count: int,
    status: str,
    class_review_path: Path = CLASS_REVIEW_PATH,
) -> dict[str, object]:
    labels = sorted({str(row["current_label"]) for row in rows})
    total_counts = Counter(str(row["current_label"]) for row in rows)
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_class_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    labels_by_source_class: dict[str, Counter[str]] = defaultdict(Counter)
    tiny_by_label: dict[str, int] = defaultdict(int)
    duplicate_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    known_catalogue = set(catalogue_by_id())
    for row in rows:
        label = str(row["current_label"])
        split = str(row["split"])
        source_class = str(row["source_class_id"])
        split_counts[split][label] += 1
        source_class_by_label[label][source_class] += 1
        labels_by_source_class[source_class][label] += 1
        if row["tiny_crop"] == "true":
            tiny_by_label[label] += 1
        digest = _sha256(project_path(str(row["file"])))
        duplicate_hashes[digest].append(
            {
                "split": split,
                "label": label,
                "file": str(row["file"]),
            }
        )
    cross_split_duplicates = {
        digest: values
        for digest, values in duplicate_hashes.items()
        if len({value["split"] for value in values}) > 1
    }
    return {
        "schema_version": "1.0",
        "classification_root": CLASSIFICATION_ROOT.relative_to(project_path(".")).as_posix(),
        "total_crops": len(rows),
        "labels": len(labels),
        "label_counts": dict(sorted(total_counts.items())),
        "rare_labels": {
            label: count for label, count in sorted(total_counts.items()) if count < RARE_THRESHOLD
        },
        "missing_labels_by_split": {
            split: sorted(set(labels) - set(counts))
            for split, counts in sorted(split_counts.items())
        },
        "tiny_crop_counts": dict(sorted(tiny_by_label.items())),
        "mixed_source_classes_by_label": {
            label: dict(counts)
            for label, counts in sorted(source_class_by_label.items())
            if len(counts) > 1
        },
        "source_classes_with_multiple_labels": {
            source_class: dict(counts)
            for source_class, counts in sorted(labels_by_source_class.items())
            if len(counts) > 1
        },
        "labels_not_in_p2_catalogue": sorted(set(labels) - known_catalogue),
        "exact_duplicate_hashes_cross_split": cross_split_duplicates,
        "seeded_corrections": corrections_count,
        "contact_sheets": contact_sheets,
        "review_manifest": MANIFEST_PATH.relative_to(project_path(".")).as_posix(),
        "class_review_manifest": class_review_path.relative_to(project_path(".")).as_posix(),
        "correction_manifest": CORRECTIONS_PATH.relative_to(project_path(".")).as_posix(),
        "status": status,
        "label_qc_status": status,
    }


def _existing_class_reviews() -> dict[str, dict[str, str]]:
    if not CLASS_REVIEW_PATH.exists():
        return {}
    with CLASS_REVIEW_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["label"]: {
                "reviewer_decision": row.get("reviewer_decision", ""),
                "reviewer_notes": row.get("reviewer_notes", ""),
            }
            for row in csv.DictReader(handle)
        }


def _write_class_review(rows: list[dict[str, object]], contact_sheets: dict[str, str]) -> Path:
    previous = _existing_class_reviews()
    total_counts = Counter(str(row["current_label"]) for row in rows)
    tiny_counts = Counter(str(row["current_label"]) for row in rows if row["tiny_crop"] == "true")
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_class_by_label: dict[str, set[str]] = defaultdict(set)
    labels = sorted(total_counts)
    for row in rows:
        label = str(row["current_label"])
        split_counts[str(row["split"])][label] += 1
        source_class_by_label[label].add(str(row["source_class_id"]))

    review_rows: list[dict[str, object]] = []
    for label in labels:
        preserved = previous.get(label, {})
        review_rows.append(
            {
                "label": label,
                "total_crops": total_counts[label],
                "train_crops": split_counts["train"][label],
                "validation_crops": split_counts["validation"][label],
                "test_crops": split_counts["test"][label],
                "rare_label": str(total_counts[label] < RARE_THRESHOLD).lower(),
                "missing_validation": str(split_counts["validation"][label] == 0).lower(),
                "missing_test": str(split_counts["test"][label] == 0).lower(),
                "tiny_crops": tiny_counts[label],
                "source_class_ids": " ".join(sorted(source_class_by_label[label])),
                "contact_sheet": contact_sheets.get(label, ""),
                "reviewer_decision": preserved.get("reviewer_decision", ""),
                "reviewer_notes": preserved.get("reviewer_notes", ""),
            }
        )
    try:
        _write_csv(CLASS_REVIEW_PATH, review_rows)
        return CLASS_REVIEW_PATH
    except PermissionError:
        fallback = CLASS_REVIEW_PATH.with_name("p5_class_review_next.csv")
        _write_csv(fallback, review_rows)
        print(f"{CLASS_REVIEW_PATH} is locked; wrote {fallback} instead.")
        return fallback


def generate_qc() -> None:
    rows = _crop_rows()
    if not rows:
        raise SystemExit(f"No classification crops found under {CLASSIFICATION_ROOT}")
    contact_sheets = _write_contact_sheets(rows)
    class_review_path = _write_class_review(rows, contact_sheets)
    _write_csv(MANIFEST_PATH, rows)
    corrections = _seed_corrections(rows)
    _write_csv(
        CORRECTIONS_PATH,
        corrections,
        fieldnames=[
            "action",
            "status",
            "split",
            "current_label",
            "new_label",
            "file",
            "instance_id",
            "source_class_id",
            "reason",
            "reviewer",
        ],
    )
    report = _build_report(
        rows,
        contact_sheets=contact_sheets,
        corrections_count=len(corrections),
        status="generated_pending_review" if corrections else "generated_no_seeded_corrections",
        class_review_path=class_review_path,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {CLASS_REVIEW_PATH}")
    print(f"Wrote {CORRECTIONS_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {len(contact_sheets)} contact sheets to {CONTACT_SHEET_ROOT}")


def _draw_empty_sheet(path: Path, message: str) -> None:
    image = Image.new("RGB", (900, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 48), message, fill=(0, 80, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def apply_corrections(corrections_path: Path = CORRECTIONS_PATH) -> None:
    if not corrections_path.exists():
        raise SystemExit(f"Correction manifest does not exist: {corrections_path}")
    with corrections_path.open(encoding="utf-8-sig", newline="") as handle:
        corrections = list(csv.DictReader(handle))
    approved = [row for row in corrections if row["status"] == "approved"]
    if not approved:
        print("No approved corrections to apply.")
        return

    applied: list[dict[str, str]] = []
    for row in approved:
        action = row["action"]
        source = project_path(row["file"])
        if not source.exists():
            raise FileNotFoundError(source)
        if action == "relabel":
            target = source.parents[1] / row["new_label"] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            applied.append({**row, "new_file": target.relative_to(project_path(".")).as_posix()})
        elif action == "remove":
            source.unlink()
            applied.append({**row, "new_file": ""})
        else:
            raise ValueError(f"Unsupported correction action: {action}")

    # Remove empty label folders left by relabels.
    for split_root in CLASSIFICATION_ROOT.iterdir():
        if not split_root.is_dir():
            continue
        for label_root in sorted((path for path in split_root.iterdir() if path.is_dir()), reverse=True):
            if not any(label_root.iterdir()):
                label_root.rmdir()

    _write_csv(
        project_path("data/manifests/p5_label_corrections_applied.csv"),
        applied,
        fieldnames=list(applied[0]),
    )
    _refresh_classification_metadata()
    print(f"Applied {len(applied)} approved corrections.")


def _refresh_classification_metadata() -> None:
    rows = _crop_rows()
    labels = sorted({str(row["current_label"]) for row in rows})
    split_counts: Counter[str] = Counter(str(row["split"]) for row in rows)
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        split_label_counts[str(row["split"])][str(row["current_label"])] += 1

    metadata_path = CLASSIFICATION_ROOT / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["annotation_status"] = "p5_label_qc_partial"
    metadata["training_scope"] = "experimental_qc_partial"
    metadata["label_qc_status"] = "partial_owner_corrections_applied"
    metadata["classification_crops"] = len(rows)
    metadata["classification_split_crops"] = dict(sorted(split_counts.items()))
    metadata["classification_split_label_counts"] = {
        split: dict(sorted(counts.items())) for split, counts in sorted(split_label_counts.items())
    }
    metadata["missing_train_labels"] = sorted(set(labels) - set(split_label_counts["train"]))
    metadata["missing_validation_labels"] = sorted(set(labels) - set(split_label_counts["validation"]))
    metadata["missing_test_labels"] = sorted(set(labels) - set(split_label_counts["test"]))
    (CLASSIFICATION_ROOT / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    rows_after = _crop_rows()
    _write_csv(MANIFEST_PATH, rows_after)
    contact_sheets = _write_contact_sheets(rows_after) if rows_after else {}
    class_review_path = _write_class_review(rows_after, contact_sheets)
    if "animal_crossing" not in {str(row["current_label"]) for row in rows_after}:
        _draw_empty_sheet(CONTACT_SHEET_ROOT / "animal_crossing.jpg", "animal_crossing has no remaining crops after P5 relabel corrections.")
        contact_sheets["animal_crossing"] = (CONTACT_SHEET_ROOT / "animal_crossing.jpg").relative_to(project_path(".")).as_posix()
    report = _build_report(
        rows_after,
        contact_sheets=contact_sheets,
        corrections_count=0,
        status="partial_owner_corrections_applied",
        class_review_path=class_review_path,
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate and apply P5 label-QC artifacts.")
    parser.add_argument("command", choices=["generate", "apply"])
    parser.add_argument(
        "--corrections",
        type=Path,
        default=CORRECTIONS_PATH,
        help="Correction manifest to apply. Defaults to data/manifests/p5_label_corrections.csv.",
    )
    args = parser.parse_args()
    if args.command == "generate":
        generate_qc()
    elif args.command == "apply":
        apply_corrections(args.corrections)


if __name__ == "__main__":
    main()
