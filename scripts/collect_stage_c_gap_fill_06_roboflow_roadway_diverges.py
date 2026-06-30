from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_gap_fill_06_roboflow_roadway_diverges"
SOURCE_DATASET = "Malaysia Road Sign Dataset v1 via Roboflow"
SOURCE_REPO = "test-22a9b/test-fwdci/1"
SOURCE_URL = "https://universe.roboflow.com/test-22a9b/test-fwdci/dataset/1"
SOURCE_ROOT = PROJECT_ROOT / "data/raw/roboflow/malaysia_road_sign_dataset_v1/extracted_full"
CROP_ROOT = PROJECT_ROOT / "data/raw/online_sources/stage_c_gap_fill_06_roboflow_roadway_diverges/crops"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_gap_fill_06_roboflow_roadway_diverges_candidates.csv"
AUDIT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_gap_fill_06_roboflow_roadway_diverges_candidates.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_gap_fill_06_roboflow_roadway_diverges"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
TRACKER_BACKUP_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.pending_stage_c_gap_fill_06.csv"
SEMANTIC_ID = "roadway_diverges"
DISPLAY_NAME = "Traffic diverges"
SOURCE_CLASS_INDEX = 44
SOURCE_CLASS_LABEL = "Roadway diverges"
MIN_BBOX_SIDE = 120.0


FIELDNAMES = [
    "stage_id",
    "candidate_id",
    "semantic_sign_id",
    "display_name",
    "source_dataset",
    "source_repo",
    "source_url",
    "original_dataset_url",
    "legend_url",
    "license_recorded",
    "license_notes",
    "source_split",
    "source_shard",
    "row_index_in_shard",
    "object_index",
    "source_image_path",
    "source_image_sha256",
    "source_class_label",
    "source_class_index",
    "mapping_evidence",
    "image_width",
    "image_height",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "bbox_width",
    "bbox_height",
    "crop_width",
    "crop_height",
    "crop_sha256",
    "local_crop_path",
    "source_modality",
    "quality_gate",
    "review_status",
    "counts_for_candidate_coverage",
    "notes",
]


@dataclass(frozen=True)
class CandidateRef:
    split: str
    image_path: Path
    label_path: Path
    object_index: int
    bbox_xywh_norm: tuple[float, float, float, float]
    image_width: int
    image_height: int
    base_id: str


def project_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_reset_path(path: Path) -> None:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(f"Refusing to reset outside project root: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def current_gap() -> int:
    with TRACKER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["semantic_sign_id"] == SEMANTIC_ID:
                return max(0, int(row["gap_to_minimum"]))
    return 0


def image_for_label(split: str, label_path: Path) -> Path | None:
    image_dir = SOURCE_ROOT / split / "images"
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = image_dir / f"{label_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def base_id_for(path: Path) -> str:
    return re.sub(r"\.rf\..*$", "", path.stem)


def collect_refs() -> list[CandidateRef]:
    refs: list[CandidateRef] = []
    for split in ("train", "valid", "test"):
        label_dir = SOURCE_ROOT / split / "labels"
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_for_label(split, label_path)
            if image_path is None:
                continue
            try:
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
            except OSError:
                continue
            for object_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    class_index = int(float(parts[0]))
                    cx, cy, width, height = [float(value) for value in parts[1:]]
                except ValueError:
                    continue
                if class_index != SOURCE_CLASS_INDEX:
                    continue
                if width * image_width < MIN_BBOX_SIDE or height * image_height < MIN_BBOX_SIDE:
                    continue
                refs.append(
                    CandidateRef(
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                        object_index=object_index,
                        bbox_xywh_norm=(cx, cy, width, height),
                        image_width=image_width,
                        image_height=image_height,
                        base_id=base_id_for(image_path),
                    )
                )
    return refs


def spread_select(refs: list[CandidateRef], needed: int) -> list[CandidateRef]:
    by_base: dict[str, CandidateRef] = {}
    for ref in sorted(refs, key=lambda item: (item.split, item.base_id, item.image_path.name, item.object_index)):
        by_base.setdefault(ref.base_id, ref)
    unique_refs = list(by_base.values())
    pool = unique_refs if len(unique_refs) >= needed else refs
    if len(pool) <= needed:
        return list(pool)
    step = len(pool) / needed
    return [pool[min(len(pool) - 1, int(index * step))] for index in range(needed)]


def bbox_pixels(ref: CandidateRef) -> tuple[float, float, float, float]:
    cx, cy, width, height = ref.bbox_xywh_norm
    x1 = (cx - width / 2) * ref.image_width
    y1 = (cy - height / 2) * ref.image_height
    x2 = (cx + width / 2) * ref.image_width
    y2 = (cy + height / 2) * ref.image_height
    return x1, y1, x2, y2


def crop_with_context(image: Image.Image, bbox_xyxy: tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = bbox_xyxy
    width = x2 - x1
    height = y2 - y1
    pad = max(width, height) * 0.2
    return image.crop(
        (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(image.width, x2 + pad),
            min(image.height, y2 + pad),
        )
    )


def materialize(refs: list[CandidateRef]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, ref in enumerate(refs, start=1):
        image_bytes = ref.image_path.read_bytes()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        x1, y1, x2, y2 = bbox_pixels(ref)
        crop = crop_with_context(image, (x1, y1, x2, y2))
        crop_bytes_io = BytesIO()
        crop.save(crop_bytes_io, format="JPEG", quality=94)
        crop_bytes = crop_bytes_io.getvalue()
        candidate_id = f"RF06-roadway-diverges-{index:04d}"
        crop_dir = CROP_ROOT / SEMANTIC_ID
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"{candidate_id}_{ref.split}_{ref.image_path.stem}_o{ref.object_index:02d}.jpg"
        crop_path.write_bytes(crop_bytes)
        rows.append(
            {
                "stage_id": STAGE_ID,
                "candidate_id": candidate_id,
                "semantic_sign_id": SEMANTIC_ID,
                "display_name": DISPLAY_NAME,
                "source_dataset": SOURCE_DATASET,
                "source_repo": SOURCE_REPO,
                "source_url": SOURCE_URL,
                "original_dataset_url": SOURCE_URL,
                "legend_url": project_rel(SOURCE_ROOT / "data.yaml"),
                "license_recorded": "Roboflow dataset metadata: CC BY 4.0.",
                "license_notes": "Use with attribution; review Roboflow project terms before redistribution.",
                "source_split": ref.split,
                "source_shard": project_rel(ref.label_path),
                "row_index_in_shard": "0",
                "object_index": str(ref.object_index),
                "source_image_path": project_rel(ref.image_path),
                "source_image_sha256": sha256_bytes(image_bytes),
                "source_class_label": SOURCE_CLASS_LABEL,
                "source_class_index": str(SOURCE_CLASS_INDEX),
                "mapping_evidence": "Roboflow class name is exactly Roadway diverges and probe sheet visually matches assignment sign_032 split/traffic-diverges warning.",
                "image_width": str(image.width),
                "image_height": str(image.height),
                "bbox_xmin": f"{x1:.3f}",
                "bbox_ymin": f"{y1:.3f}",
                "bbox_xmax": f"{x2:.3f}",
                "bbox_ymax": f"{y2:.3f}",
                "bbox_width": f"{x2 - x1:.3f}",
                "bbox_height": f"{y2 - y1:.3f}",
                "crop_width": str(crop.width),
                "crop_height": str(crop.height),
                "crop_sha256": sha256_bytes(crop_bytes),
                "local_crop_path": project_rel(crop_path),
                "source_modality": "real_road_photo_crop_from_detection_dataset",
                "quality_gate": "accepted_auto_exact_label_with_probe_visual_check",
                "review_status": "auto_exact_label_pending_stage_d_visual_qc",
                "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
                "notes": "Selected with minimum 120px bbox side and one-per-base-image spread sampling before using augment variants.",
            }
        )
    return rows


def make_contact_sheet(rows: list[dict[str, str]], output_path: Path, max_items: int = 100) -> None:
    selected = rows[:max_items]
    cols = 5
    tile_w, tile_h = 190, 190
    sheet_rows = max(1, (len(selected) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tile_w, sheet_rows * tile_h), (236, 236, 236))
    for index, row in enumerate(selected):
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        try:
            image = Image.open(PROJECT_ROOT / row["local_crop_path"]).convert("RGB")
            image = ImageOps.contain(image, (150, 145), Image.Resampling.LANCZOS)
            tile.paste(image, ((tile_w - image.width) // 2, 6))
        except OSError:
            pass
        draw = ImageDraw.Draw(tile)
        draw.text((6, 155), f"{row['source_split']} {row['source_class_label']}"[:28], fill="black")
        draw.text((6, 172), row["candidate_id"][:28], fill="black")
        sheet.paste(tile, ((index % cols) * tile_w, (index // cols) * tile_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_outputs(rows: list[dict[str, str]], audit: dict[str, Any]) -> None:
    rows.sort(key=lambda row: row["candidate_id"])
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(rows, SHEET_ROOT / f"{SEMANTIC_ID}.jpg")
    make_contact_sheet(rows, SHEET_ROOT / "_qa_roadway_diverges_50.jpg", max_items=50)
    audit["generated_at"] = datetime.now(timezone.utc).isoformat()
    audit["manifest_path"] = project_rel(MANIFEST_PATH)
    audit["crop_root"] = project_rel(CROP_ROOT)
    audit["review_root"] = project_rel(SHEET_ROOT)
    audit["counts_by_class"] = dict(sorted(Counter(row["semantic_sign_id"] for row in rows).items()))
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def update_tracker(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["semantic_sign_id"] for row in rows)
    with TRACKER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        tracker_rows = list(reader)
    for tracker_row in tracker_rows:
        if tracker_row["semantic_sign_id"] != SEMANTIC_ID:
            continue
        current = int(tracker_row["realistic_candidate_total"])
        minimum = int(tracker_row["minimum_clean_crops"])
        new_total = current + counts.get(SEMANTIC_ID, 0)
        tracker_row["realistic_candidate_total"] = str(new_total)
        tracker_row["gap_to_minimum"] = str(max(0, minimum - new_total))
        tracker_row["cleaning_status"] = "stage_d_qc_needed"
        tracker_row["collection_status"] = "meets_minimum_pending_qc" if new_total >= minimum else "still_below_minimum"
        tracker_row["next_action"] = "Review Stage C gap-fill 06 Roboflow roadway-diverges sheet, then include accepted crops in Stage E split freeze."
    for path in (TRACKER_PATH, TRACKER_BACKUP_PATH):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tracker_rows)
    return dict(counts)


def write_gap_snapshot() -> None:
    csv_path = PROJECT_ROOT / "outputs/audit/post_stage_c_gap_fill_06_gap_report.csv"
    json_path = PROJECT_ROOT / "outputs/audit/post_stage_c_gap_fill_06_gap_report.json"
    shutil.copy2(TRACKER_PATH, csv_path)
    with TRACKER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    must = [row for row in rows if row["priority"] == "must"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_tracker": project_rel(TRACKER_PATH),
        "must_have_classes": len(must),
        "must_have_meeting_minimum_candidate_count": sum(int(row["gap_to_minimum"]) == 0 for row in must),
        "must_have_below_minimum_candidate_count": sum(int(row["gap_to_minimum"]) > 0 for row in must),
        "note": "Snapshot after Stage C Gap Fill 06. Candidate coverage still requires Stage D QC before Stage E split freeze.",
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--target", type=int, default=50)
    args = parser.parse_args()
    if args.reset:
        safe_reset_path(CROP_ROOT)
        safe_reset_path(SHEET_ROOT)
        MANIFEST_PATH.unlink(missing_ok=True)
        AUDIT_PATH.unlink(missing_ok=True)

    gap = current_gap()
    if gap <= 0:
        print("No remaining roadway_diverges gap.")
        return
    refs = collect_refs()
    selected = spread_select(refs, min(args.target, gap))
    rows = materialize(selected)
    audit = {
        "stage_id": STAGE_ID,
        "source_url": SOURCE_URL,
        "available_refs_after_size_filter": len(refs),
        "unique_base_images_after_size_filter": len({ref.base_id for ref in refs}),
        "selected_count": len(selected),
        "minimum_bbox_side_px": MIN_BBOX_SIDE,
        "note": "No AI-generated images. Exact Roboflow Roadway diverges class only; visual probe accepted before import.",
    }
    write_outputs(rows, audit)
    counts = update_tracker(rows)
    write_gap_snapshot()
    print(f"Wrote {len(rows)} candidates")
    print(json.dumps(dict(sorted(counts.items())), indent=2))


if __name__ == "__main__":
    main()
