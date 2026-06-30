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

import polars as pl
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_gap_fill_04_mapillary_exact"
HF_REPO_ID = "ThankGod/mapillary_traffic_sign_dataset"
HF_REPO_TYPE = "dataset"
SOURCE_DATASET = "Mapillary Traffic Sign Dataset via Hugging Face mirror"
SOURCE_URL = "https://huggingface.co/datasets/ThankGod/mapillary_traffic_sign_dataset"
ORIGINAL_DATASET_URL = "https://www.mapillary.com/dataset/trafficsign"
ID2LABELS_PATH = PROJECT_ROOT / "data/raw/online_sources/hf_metadata_probe/mapillary_traffic_sign_dataset/id2labels.txt"
STAGE_ROOT = PROJECT_ROOT / "data/raw/online_sources/stage_c_gap_fill_04_mapillary"
SHARD_ROOT = STAGE_ROOT / "_hf_parquet_shards"
CROP_ROOT = STAGE_ROOT / "crops"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_gap_fill_04_mapillary_exact_candidates.csv"
AUDIT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_gap_fill_04_mapillary_exact_candidates.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_gap_fill_04_mapillary_exact"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
TRACKER_BACKUP_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.pending_stage_c_gap_fill_04.csv"
MIN_BBOX_SIDE = 26.0


@dataclass(frozen=True)
class TargetDef:
    semantic_sign_id: str
    display_name: str
    mapillary_labels: tuple[str, ...]
    mapping_evidence: str
    label_notes: str = ""


@dataclass(frozen=True)
class CandidateRef:
    target: TargetDef
    source_label: str
    source_label_id: int
    source_split: str
    source_shard: str
    row_index: int
    object_index: int
    source_image_id: str
    bbox: tuple[float, float, float, float]
    bbox_width: float
    bbox_height: float


TARGETS: tuple[TargetDef, ...] = (
    TargetDef(
        "width_restriction",
        "Width restriction",
        ("regulatory--width-limit--g1",),
        "Mapillary label explicitly names regulatory width-limit signs, matching the project width_restriction class.",
    ),
)


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


FALLBACK_SHARDS: tuple[tuple[str, int], ...] = (
    ("val_mtsd_new/train-00000-of-00007.parquet", 447_664_792),
    ("val_mtsd_new/train-00001-of-00007.parquet", 453_872_134),
    ("val_mtsd_new/train-00002-of-00007.parquet", 466_206_372),
    ("val_mtsd_new/train-00003-of-00007.parquet", 451_127_497),
    ("val_mtsd_new/train-00004-of-00007.parquet", 473_474_192),
    ("val_mtsd_new/train-00005-of-00007.parquet", 463_264_381),
    ("val_mtsd_new/train-00006-of-00007.parquet", 459_611_722),
)


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


def parse_mapillary_labels() -> dict[int, str]:
    labels: dict[int, str] = {}
    text = ID2LABELS_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = re.search(r"(\d+)\s*:\s*'([^']+)'", line)
        if match:
            labels[int(match.group(1))] = match.group(2)
    return labels


def label_to_id_map() -> dict[str, int]:
    return {label: idx for idx, label in parse_mapillary_labels().items()}


def local_shard_path(shard_path: str) -> Path:
    return SHARD_ROOT / shard_path


def is_complete_shard(shard_path: str, expected_size: int | None) -> bool:
    path = local_shard_path(shard_path)
    if not path.exists():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return path.stat().st_size > 0


def discover_shards() -> list[tuple[str, int | None]]:
    try:
        api = HfApi()
        info = api.dataset_info(HF_REPO_ID, files_metadata=True)
        shards: list[tuple[str, int | None]] = []
        for sibling in info.siblings:
            filename = sibling.rfilename
            if filename.startswith("val_mtsd_new/") and filename.endswith(".parquet"):
                shards.append((filename, getattr(sibling, "size", None)))
        if shards:
            return sorted(shards)
    except Exception as exc:
        print(f"Warning: using fallback shard list after HF metadata error: {exc}", flush=True)
    return list(FALLBACK_SHARDS)


def download_shard(shard_path: str) -> Path:
    local = local_shard_path(shard_path)
    if local.exists() and local.stat().st_size > 0:
        return local
    print(f"Downloading Mapillary shard {shard_path}...", flush=True)
    SHARD_ROOT.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=shard_path,
        local_dir=SHARD_ROOT,
    )
    return Path(path)


def current_gaps() -> dict[str, int]:
    target_ids = {target.semantic_sign_id for target in TARGETS}
    gaps = {target_id: 0 for target_id in target_ids}
    with TRACKER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            semantic_id = row["semantic_sign_id"]
            if semantic_id in gaps:
                gaps[semantic_id] = max(0, int(row["gap_to_minimum"]))
    return gaps


def load_existing_stage_rows() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def existing_markers() -> set[str]:
    markers: set[str] = set()
    if not MANIFEST_PATH.exists():
        return markers
    for row in load_existing_stage_rows():
        markers.add(
            "|".join(
                [
                    row["source_shard"],
                    row["row_index_in_shard"],
                    row["object_index"],
                    row["source_class_label"],
                ]
            )
        )
    return markers


def marker_for(shard_path: str, row_index: int, object_index: int, source_label: str) -> str:
    return "|".join([shard_path, str(row_index), str(object_index), source_label])


def scan_shards(
    shards: list[tuple[str, int | None]],
    label_to_idx: dict[str, int],
    gaps: dict[str, int],
    used_markers: set[str],
) -> dict[str, list[CandidateRef]]:
    idx_to_target: dict[int, tuple[TargetDef, str]] = {}
    for target in TARGETS:
        if gaps.get(target.semantic_sign_id, 0) <= 0:
            continue
        for source_label in target.mapillary_labels:
            if source_label in label_to_idx:
                idx_to_target[label_to_idx[source_label]] = (target, source_label)

    refs_by_class: dict[str, list[CandidateRef]] = defaultdict(list)
    for shard_path, _expected_size in shards:
        local = local_shard_path(shard_path)
        if not local.exists():
            continue
        print(f"Scanning {shard_path}...", flush=True)
        df = pl.read_parquet(local, columns=["image_id", "objects"])
        for row_index, row in enumerate(df.to_dicts()):
            objects = row.get("objects") or {}
            categories = objects.get("category") or []
            bboxes = objects.get("bbox") or []
            source_image_id = str(row.get("image_id", ""))
            for object_index, category in enumerate(categories):
                if category not in idx_to_target or object_index >= len(bboxes):
                    continue
                target, source_label = idx_to_target[category]
                if marker_for(shard_path, row_index, object_index, source_label) in used_markers:
                    continue
                x1, y1, x2, y2 = [float(value) for value in bboxes[object_index]]
                width = x2 - x1
                height = y2 - y1
                if width < MIN_BBOX_SIDE or height < MIN_BBOX_SIDE:
                    continue
                refs_by_class[target.semantic_sign_id].append(
                    CandidateRef(
                        target=target,
                        source_label=source_label,
                        source_label_id=int(category),
                        source_split="val_mtsd_new",
                        source_shard=shard_path,
                        row_index=row_index,
                        object_index=object_index,
                        source_image_id=source_image_id,
                        bbox=(x1, y1, x2, y2),
                        bbox_width=width,
                        bbox_height=height,
                    )
                )
    return refs_by_class


def spread_select(refs: list[CandidateRef], needed: int) -> list[CandidateRef]:
    if len(refs) <= needed:
        return list(refs)
    refs = sorted(refs, key=lambda ref: (ref.source_shard, ref.row_index, ref.object_index))
    step = len(refs) / needed
    return [refs[min(len(refs) - 1, int(index * step))] for index in range(needed)]


def select_refs(refs_by_class: dict[str, list[CandidateRef]], gaps: dict[str, int]) -> list[CandidateRef]:
    selected: list[CandidateRef] = []
    for semantic_id, refs in sorted(refs_by_class.items()):
        needed = gaps.get(semantic_id, 0)
        if needed <= 0:
            continue
        selected.extend(spread_select(refs, needed))
    return selected


def crop_with_context(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    pad = max(width, height) * 0.35
    return image.crop(
        (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(image.width, x2 + pad),
            min(image.height, y2 + pad),
        )
    )


def materialize_refs(refs: list[CandidateRef]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    by_shard: dict[str, list[CandidateRef]] = defaultdict(list)
    for ref in refs:
        by_shard[ref.source_shard].append(ref)

    counters: Counter[str] = Counter()
    for shard_path, shard_refs in sorted(by_shard.items()):
        local = local_shard_path(shard_path)
        df = pl.read_parquet(local)
        row_dicts = df.to_dicts()
        for ref in sorted(shard_refs, key=lambda item: (item.row_index, item.object_index)):
            row = row_dicts[ref.row_index]
            image_struct = row.get("image") or {}
            image_bytes = image_struct.get("bytes")
            if not image_bytes:
                continue
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            crop = crop_with_context(image, ref.bbox)
            crop_bytes_io = BytesIO()
            crop.save(crop_bytes_io, format="JPEG", quality=94)
            crop_bytes = crop_bytes_io.getvalue()
            crop_sha = sha256_bytes(crop_bytes)

            counters[ref.target.semantic_sign_id] += 1
            safe_label = re.sub(r"[^a-z0-9]+", "-", ref.source_label.lower()).strip("-")
            candidate_id = f"MTSD04-{safe_label}-{counters[ref.target.semantic_sign_id]:04d}"
            crop_dir = CROP_ROOT / ref.target.semantic_sign_id
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f"{candidate_id}_{Path(shard_path).stem}_r{ref.row_index:04d}_o{ref.object_index:02d}.jpg"
            crop_path.write_bytes(crop_bytes)

            x1, y1, x2, y2 = ref.bbox
            notes = "Cropped from Mapillary object bounding box with context margin; source class retained."
            if ref.target.label_notes:
                notes += " " + ref.target.label_notes
            source_image_path = image_struct.get("path") or f"mapillary_image_id_{ref.source_image_id}"
            rows.append(
                {
                    "stage_id": STAGE_ID,
                    "candidate_id": candidate_id,
                    "semantic_sign_id": ref.target.semantic_sign_id,
                    "display_name": ref.target.display_name,
                    "source_dataset": SOURCE_DATASET,
                    "source_repo": HF_REPO_ID,
                    "source_url": SOURCE_URL,
                    "original_dataset_url": ORIGINAL_DATASET_URL,
                    "legend_url": project_rel(ID2LABELS_PATH),
                    "license_recorded": "Hugging Face mirror has no explicit dataset card license in the cached README.",
                    "license_notes": "Treat as academic candidate data; verify Mapillary dataset terms before redistribution.",
                    "source_split": ref.source_split,
                    "source_shard": shard_path,
                    "row_index_in_shard": str(ref.row_index),
                    "object_index": str(ref.object_index),
                    "source_image_path": str(source_image_path),
                    "source_image_sha256": sha256_bytes(image_bytes),
                    "source_class_label": ref.source_label,
                    "source_class_index": str(ref.source_label_id),
                    "mapping_evidence": ref.target.mapping_evidence,
                    "image_width": str(image.width),
                    "image_height": str(image.height),
                    "bbox_xmin": f"{x1:.3f}",
                    "bbox_ymin": f"{y1:.3f}",
                    "bbox_xmax": f"{x2:.3f}",
                    "bbox_ymax": f"{y2:.3f}",
                    "bbox_width": f"{ref.bbox_width:.3f}",
                    "bbox_height": f"{ref.bbox_height:.3f}",
                    "crop_width": str(crop.width),
                    "crop_height": str(crop.height),
                    "crop_sha256": crop_sha,
                    "local_crop_path": project_rel(crop_path),
                    "source_modality": "real_road_photo_crop_from_detection_dataset",
                    "quality_gate": "accepted_auto_exact_label",
                    "review_status": "auto_exact_label_pending_stage_d_visual_qc",
                    "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
                    "notes": notes,
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
        draw.text((6, 155), row["source_class_label"][:28], fill="black")
        draw.text((6, 172), row["candidate_id"][:28], fill="black")
        sheet.paste(tile, ((index % cols) * tile_w, (index // cols) * tile_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_qa_sheet(rows: list[dict[str, str]], output_path: Path) -> None:
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if len(by_class[row["semantic_sign_id"]]) < 8:
            by_class[row["semantic_sign_id"]].append(row)

    classes = sorted(by_class)
    label_w = 230
    tile_w = 150
    row_h = 172
    sheet = Image.new("RGB", (label_w + 8 * tile_w, max(1, len(classes)) * row_h), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for row_index, semantic_id in enumerate(classes):
        y = row_index * row_h
        draw.rectangle(
            (0, y, label_w + 8 * tile_w, y + row_h - 1),
            fill=(246, 246, 246) if row_index % 2 == 0 else (232, 232, 232),
        )
        draw.text((12, y + 12), semantic_id, fill="black")
        draw.text((12, y + 34), f"{len([r for r in rows if r['semantic_sign_id'] == semantic_id])} candidates", fill=(70, 70, 70))
        for col_index, sample in enumerate(by_class[semantic_id]):
            x = label_w + col_index * tile_w
            tile = Image.new("RGB", (tile_w, 150), "white")
            try:
                image = Image.open(PROJECT_ROOT / sample["local_crop_path"]).convert("RGB")
                image = ImageOps.contain(image, (132, 108), Image.Resampling.LANCZOS)
                tile.paste(image, ((tile_w - image.width) // 2, 8))
            except OSError:
                pass
            td = ImageDraw.Draw(tile)
            td.text((5, 122), sample["candidate_id"][:21], fill="black")
            td.text((5, 136), sample["source_class_label"][:21], fill=(55, 55, 55))
            sheet.paste(tile, (x, y + 10))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_outputs(rows: list[dict[str, str]], audit: dict[str, Any]) -> None:
    rows.sort(key=lambda row: (row["semantic_sign_id"], row["candidate_id"]))
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["semantic_sign_id"]].append(row)
    for semantic_id, class_rows in by_class.items():
        make_contact_sheet(class_rows, SHEET_ROOT / f"{semantic_id}.jpg")
    make_qa_sheet(rows, SHEET_ROOT / "_qa_exact_8each.jpg")

    audit["generated_at"] = datetime.now(timezone.utc).isoformat()
    audit["manifest_path"] = project_rel(MANIFEST_PATH)
    audit["crop_root"] = project_rel(CROP_ROOT)
    audit["review_root"] = project_rel(SHEET_ROOT)
    audit["counts_by_class"] = dict(sorted(Counter(row["semantic_sign_id"] for row in rows).items()))
    audit["note"] = (
        "No AI-generated images. Mapillary exact named categories only. "
        "Rejected mappings: pass-left-or-right is not roadway_diverges, and no-straight-through is not the red no-straight-ahead prohibition."
    )
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def update_tracker(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["semantic_sign_id"] for row in rows)
    with TRACKER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        tracker_rows = list(reader)

    for tracker_row in tracker_rows:
        semantic_id = tracker_row["semantic_sign_id"]
        if semantic_id not in counts:
            continue
        current = int(tracker_row["realistic_candidate_total"])
        minimum = int(tracker_row["minimum_clean_crops"])
        new_total = current + counts[semantic_id]
        tracker_row["realistic_candidate_total"] = str(new_total)
        tracker_row["gap_to_minimum"] = str(max(0, minimum - new_total))
        tracker_row["cleaning_status"] = "stage_d_qc_needed"
        tracker_row["collection_status"] = (
            "meets_minimum_pending_qc" if new_total >= minimum else "still_below_minimum"
        )
        tracker_row["next_action"] = (
            "Review Stage C gap-fill 04 Mapillary contact sheets, then include accepted crops in Stage E split freeze."
        )

    for path in (TRACKER_PATH, TRACKER_BACKUP_PATH):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tracker_rows)
    return dict(counts)


def write_gap_snapshot() -> None:
    csv_path = PROJECT_ROOT / "outputs/audit/post_stage_c_gap_fill_04_gap_report.csv"
    json_path = PROJECT_ROOT / "outputs/audit/post_stage_c_gap_fill_04_gap_report.json"
    shutil.copy2(TRACKER_PATH, csv_path)
    with TRACKER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    must = [row for row in rows if row["priority"] == "must"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_tracker": project_rel(TRACKER_PATH),
        "target_classes": len(rows),
        "must_have_classes": len(must),
        "must_have_meeting_minimum_candidate_count": sum(int(row["gap_to_minimum"]) == 0 for row in must),
        "must_have_below_minimum_candidate_count": sum(int(row["gap_to_minimum"]) > 0 for row in must),
        "note": "Snapshot after Stage C Gap Fill 04. Candidate coverage still requires Stage D QC before Stage E split freeze.",
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--max-downloads", type=int, default=1)
    parser.add_argument("--download-all", action="store_true")
    args = parser.parse_args()

    if args.reset:
        safe_reset_path(CROP_ROOT)
        safe_reset_path(SHEET_ROOT)
        MANIFEST_PATH.unlink(missing_ok=True)
        AUDIT_PATH.unlink(missing_ok=True)

    label_to_idx = label_to_id_map()
    missing = sorted({label for target in TARGETS for label in target.mapillary_labels if label not in label_to_idx})
    if missing:
        raise RuntimeError(f"Missing Mapillary labels in id2labels file: {missing}")

    gaps = current_gaps()
    if not any(gaps.values()):
        print("No remaining Stage 04 gaps in current tracker.")
        return

    all_shards = discover_shards()
    downloads = 0
    while True:
        complete_shards = [item for item in all_shards if is_complete_shard(*item)]
        refs_by_class = scan_shards(complete_shards, label_to_idx, gaps, existing_markers())
        available = {semantic_id: len(refs_by_class.get(semantic_id, [])) for semantic_id in gaps}
        enough = all(available.get(semantic_id, 0) >= gap for semantic_id, gap in gaps.items() if gap > 0)
        print("Available exact Mapillary refs:", json.dumps(available, indent=2), flush=True)
        if enough:
            break
        if not args.download_all and downloads >= args.max_downloads:
            break
        next_shard = next((item for item in all_shards if not is_complete_shard(*item)), None)
        if next_shard is None:
            break
        download_shard(next_shard[0])
        downloads += 1

    refs_by_class = scan_shards([item for item in all_shards if is_complete_shard(*item)], label_to_idx, gaps, existing_markers())
    selected = select_refs(refs_by_class, gaps)
    rows = materialize_refs(selected)
    existing_rows = load_existing_stage_rows()
    combined_rows = existing_rows + rows
    audit: dict[str, Any] = {
        "stage_id": STAGE_ID,
        "source_urls": {
            "mapillary_hf_mirror": SOURCE_URL,
            "mapillary_original": ORIGINAL_DATASET_URL,
            "local_label_map": project_rel(ID2LABELS_PATH),
        },
        "gaps_at_start": gaps,
        "available_refs_by_class": {semantic_id: len(refs_by_class.get(semantic_id, [])) for semantic_id in gaps},
        "selected_refs_by_class": dict(sorted(Counter(ref.target.semantic_sign_id for ref in selected).items())),
        "new_counts_by_class": dict(sorted(Counter(row["semantic_sign_id"] for row in rows).items())),
        "existing_stage04_rows_preserved": len(existing_rows),
        "downloaded_shards_this_run": downloads,
        "complete_shards_scanned": len([item for item in all_shards if is_complete_shard(*item)]),
        "rejected_non_exact_mapping": {
            "roadway_diverges": "Mapillary warning--pass-left-or-right probe showed arrow plates/control signs, not the assignment split/fork warning.",
            "no_straight_ahead": "Mapillary regulatory--no-straight-through samples showed yellow no-through-road/T-junction signs, not the assignment red no-straight-ahead prohibition.",
        },
    }
    write_outputs(combined_rows, audit)
    counts = update_tracker(rows)
    write_gap_snapshot()
    print(f"Wrote {len(rows)} candidates")
    print(json.dumps(dict(sorted(counts.items())), indent=2))
    remaining = {
        semantic_id: max(0, gaps[semantic_id] - counts.get(semantic_id, 0))
        for semantic_id in gaps
        if gaps[semantic_id] > counts.get(semantic_id, 0)
    }
    print("Remaining gaps after this source:", json.dumps(remaining, indent=2))


if __name__ == "__main__":
    main()
