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
from huggingface_hub import HfApi, hf_hub_url
from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_gap_fill_05_prashant_tt100k_remote"
HF_REPO_ID = "PrashantDixit0/TT-100K"
HF_REPO_TYPE = "dataset"
SOURCE_DATASET = "Tsinghua-Tencent 100K / TT100K alternate Hugging Face mirror"
SOURCE_URL = "https://huggingface.co/datasets/PrashantDixit0/TT-100K"
ORIGINAL_DATASET_URL = "https://cg.cs.tsinghua.edu.cn/traffic-sign/"
DATA_YAML_PATH = PROJECT_ROOT / "data/raw/online_sources/hf_metadata_probe/prashant_tt100k/data.yaml"
STAGE_ROOT = PROJECT_ROOT / "data/raw/online_sources/stage_c_gap_fill_05_prashant_tt100k"
CROP_ROOT = STAGE_ROOT / "crops"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_gap_fill_05_prashant_tt100k_remote_candidates.csv"
AUDIT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_gap_fill_05_prashant_tt100k_remote_candidates.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_gap_fill_05_prashant_tt100k_remote"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
TRACKER_BACKUP_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.pending_stage_c_gap_fill_05.csv"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"
MIN_BBOX_SIDE = 24.0


@dataclass(frozen=True)
class TargetDef:
    semantic_sign_id: str
    display_name: str
    tt100k_labels: tuple[str, ...]
    mapping_evidence: str
    label_notes: str = ""


@dataclass(frozen=True)
class CandidateRef:
    target: TargetDef
    source_label: str
    source_split: str
    source_shard: str
    split_global_row_index: int
    row_index_in_shard: int
    object_index: int
    source_image_id: str
    bbox_xywh: tuple[float, float, float, float]


TARGETS: tuple[TargetDef, ...] = (
    TargetDef(
        "no_left_or_right_turn",
        "No left or right turn",
        ("p20",),
        "TT100K p20 icon is the compound no-left/no-right-turn prohibition.",
    ),
    TargetDef(
        "no_straight_ahead",
        "No straight ahead",
        ("p14",),
        "TT100K p14 icon is the straight-ahead prohibition matching assignment sign_010.",
    ),
    TargetDef(
        "no_straight_or_left",
        "No straight or left turn",
        ("p28",),
        "TT100K p28 icon is the compound no-straight/no-left prohibition matching assignment sign_008.",
    ),
    TargetDef(
        "residential_area_ahead",
        "Residential area ahead",
        ("w3",),
        "TT100K w3 icon is the village/residential-area-ahead warning matching assignment sign_045.",
    ),
    TargetDef(
        "roadway_diverges",
        "Traffic diverges",
        ("w53",),
        "TT100K w53 icon is the split-traffic/traffic-diverges warning matching assignment sign_032.",
    ),
    TargetDef(
        "sound_horn",
        "Sound horn",
        ("i9",),
        "TT100K i9 icon is the blue mandatory sound-horn sign matching assignment sign_029.",
    ),
    TargetDef(
        "stop_for_checking",
        "Stop for checking",
        ("ps",),
        "TT100K ps crops show red stop-for-checking signs with Chinese inspection text matching assignment sign_057.",
    ),
    TargetDef(
        "tractors_ahead",
        "Tractors ahead",
        ("w34",),
        "TT100K w34 icon is the watch-out-for-tractors warning matching assignment sign_051.",
    ),
    TargetDef(
        "turn_left_or_right",
        "Turn left or right",
        ("i11",),
        "TT100K i11 icon is the blue mandatory turn-left-or-right sign matching assignment sign_023.",
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


def parse_data_yaml_labels() -> set[str]:
    labels: set[str] = set()
    for line in DATA_YAML_PATH.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*\d+\s*:\s*(\S+)\s*$", line)
        if match:
            labels.add(match.group(1))
    return labels


def current_gaps() -> dict[str, int]:
    target_ids = {target.semantic_sign_id for target in TARGETS}
    gaps = {target_id: 0 for target_id in target_ids}
    with TRACKER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            semantic_id = row["semantic_sign_id"]
            if semantic_id in gaps:
                gaps[semantic_id] = max(0, int(row["gap_to_minimum"]))
    return gaps


def discover_parquet_files() -> list[tuple[str, str]]:
    api = HfApi()
    info = api.dataset_info(HF_REPO_ID, files_metadata=True)
    files: list[tuple[str, str]] = []
    for sibling in info.siblings:
        filename = sibling.rfilename
        match = re.match(r"data/(train|test|val)-\d+-of-\d+\.parquet$", filename)
        if match:
            files.append((match.group(1), filename))
    split_order = {"train": 0, "test": 1, "val": 2}
    return sorted(files, key=lambda item: (split_order[item[0]], item[1]))


def remote_parquet_url(path: str) -> str:
    return hf_hub_url(HF_REPO_ID, path, repo_type=HF_REPO_TYPE)


def load_existing_stage_rows() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def marker_from_parts(image_id: str, label: str, bbox_xywh: tuple[float, float, float, float]) -> str:
    rounded = ",".join(str(int(round(value))) for value in bbox_xywh)
    return f"{image_id}|{label}|{rounded}"


def existing_tt100k_markers() -> set[str]:
    markers: set[str] = set()
    for manifest in (PROJECT_ROOT / "data/manifests").glob("stage_c_gap_fill_*_candidates.csv"):
        try:
            with manifest.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    source_repo = row.get("source_repo", "")
                    source_dataset = row.get("source_dataset", "")
                    if "tt100k" not in (source_repo + " " + source_dataset).lower():
                        continue
                    source_label = row.get("source_class_label", "")
                    image_id = Path(row.get("source_image_path", "")).stem
                    if not image_id:
                        continue
                    try:
                        x1 = float(row.get("bbox_xmin", "0"))
                        y1 = float(row.get("bbox_ymin", "0"))
                        width = float(row.get("bbox_width", "0"))
                        height = float(row.get("bbox_height", "0"))
                    except ValueError:
                        continue
                    markers.add(marker_from_parts(image_id, source_label, (x1, y1, width, height)))
        except OSError:
            continue
    return markers


def scan_remote_files(max_files: int | None, gaps: dict[str, int], existing_markers: set[str]) -> tuple[dict[str, list[CandidateRef]], dict[str, Any]]:
    labels_present = parse_data_yaml_labels()
    missing_labels = sorted({label for target in TARGETS for label in target.tt100k_labels if label not in labels_present})
    target_by_label: dict[str, TargetDef] = {}
    for target in TARGETS:
        if gaps.get(target.semantic_sign_id, 0) <= 0:
            continue
        for label in target.tt100k_labels:
            if label in labels_present:
                target_by_label[label] = target

    refs_by_class: dict[str, list[CandidateRef]] = defaultdict(list)
    row_offsets: dict[str, int] = defaultdict(int)
    files_scanned = 0
    rows_scanned = 0
    errors: list[str] = []
    for split, filename in discover_parquet_files():
        if max_files is not None and files_scanned >= max_files:
            break
        url = remote_parquet_url(filename)
        try:
            print(f"Remote metadata scan {filename}", flush=True)
            df = pl.read_parquet(url, columns=["image_id", "objects"])
        except Exception as exc:
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")
            continue
        split_start_offset = row_offsets[split]
        row_offsets[split] += df.height
        files_scanned += 1
        rows_scanned += df.height
        for row_index, row in enumerate(df.to_dicts()):
            objects = row.get("objects") or []
            image_id = str(row.get("image_id", ""))
            for object_index, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    continue
                label = obj.get("category")
                if label not in target_by_label:
                    continue
                bbox_values = obj.get("bbox") or []
                if len(bbox_values) != 4:
                    continue
                x, y, width, height = [float(value) for value in bbox_values]
                if width < MIN_BBOX_SIDE or height < MIN_BBOX_SIDE:
                    continue
                if marker_from_parts(image_id, label, (x, y, width, height)) in existing_markers:
                    continue
                target = target_by_label[label]
                refs_by_class[target.semantic_sign_id].append(
                    CandidateRef(
                        target=target,
                        source_label=label,
                        source_split=split,
                        source_shard=filename,
                        split_global_row_index=split_start_offset + row_index,
                        row_index_in_shard=row_index,
                        object_index=object_index,
                        source_image_id=image_id,
                        bbox_xywh=(x, y, width, height),
                    )
                )
    audit = {
        "files_scanned": files_scanned,
        "rows_scanned": rows_scanned,
        "missing_labels_in_data_yaml": missing_labels,
        "scan_errors": errors[:20],
        "scan_error_count": len(errors),
    }
    return refs_by_class, audit


def spread_select(refs: list[CandidateRef], needed: int) -> list[CandidateRef]:
    if len(refs) <= needed:
        return list(refs)
    refs = sorted(refs, key=lambda ref: (ref.source_split, ref.source_shard, ref.row_index_in_shard, ref.object_index))
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


def decode_image_bytes(image_struct: dict[str, Any]) -> bytes:
    raw = image_struct.get("bytes")
    if isinstance(raw, bytes):
        return raw
    raise ValueError("Missing image bytes from dataset row")


def crop_with_context(image: Image.Image, bbox_xywh: tuple[float, float, float, float]) -> tuple[Image.Image, tuple[float, float, float, float]]:
    x, y, width, height = bbox_xywh
    x2 = x + width
    y2 = y + height
    pad = max(width, height) * 0.35
    crop = image.crop(
        (
            max(0, x - pad),
            max(0, y - pad),
            min(image.width, x2 + pad),
            min(image.height, y2 + pad),
        )
    )
    return crop, (x, y, x2, y2)


def materialize_refs(refs: list[CandidateRef]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    by_shard: dict[str, list[CandidateRef]] = defaultdict(list)
    for ref in refs:
        by_shard[ref.source_shard].append(ref)
    for shard_path, shard_refs in sorted(by_shard.items()):
        print(f"Materializing selected rows from {shard_path}", flush=True)
        shard_df = pl.read_parquet(remote_parquet_url(shard_path))
        row_dicts = shard_df.to_dicts()
        for ref in sorted(shard_refs, key=lambda item: (item.row_index_in_shard, item.object_index)):
            row = row_dicts[ref.row_index_in_shard]
            image_bytes = decode_image_bytes(row.get("image") or {})
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            crop, bbox_xyxy = crop_with_context(image, ref.bbox_xywh)
            crop_bytes_io = BytesIO()
            crop.save(crop_bytes_io, format="JPEG", quality=94)
            crop_bytes = crop_bytes_io.getvalue()
            crop_sha = sha256_bytes(crop_bytes)

            counters[ref.target.semantic_sign_id] += 1
            candidate_id = f"PTT05-{ref.source_label}-{counters[ref.target.semantic_sign_id]:04d}"
            crop_dir = CROP_ROOT / ref.target.semantic_sign_id
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f"{candidate_id}_{ref.source_split}_r{ref.split_global_row_index:05d}_o{ref.object_index:02d}.jpg"
            crop_path.write_bytes(crop_bytes)

            x1, y1, x2, y2 = bbox_xyxy
            width = x2 - x1
            height = y2 - y1
            notes = "Remote metadata-scanned TT100K shard; full shard read only when at least one exact-label candidate was selected."
            if ref.target.label_notes:
                notes += " " + ref.target.label_notes
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
                    "legend_url": project_rel(DATA_YAML_PATH),
                    "license_recorded": "HF card: cc-by-2.0; data.yaml notes original TT100K CC BY-NC 2.0 for non-commercial use.",
                    "license_notes": "Use for academic coursework candidate data; verify terms before redistribution.",
                    "source_split": ref.source_split,
                    "source_shard": ref.source_shard,
                    "row_index_in_shard": str(ref.row_index_in_shard),
                    "object_index": str(ref.object_index),
                    "source_image_path": f"{ref.source_image_id}.png",
                    "source_image_sha256": sha256_bytes(image_bytes),
                    "source_class_label": ref.source_label,
                    "source_class_index": ref.source_label,
                    "mapping_evidence": ref.target.mapping_evidence,
                    "image_width": str(image.width),
                    "image_height": str(image.height),
                    "bbox_xmin": f"{x1:.3f}",
                    "bbox_ymin": f"{y1:.3f}",
                    "bbox_xmax": f"{x2:.3f}",
                    "bbox_ymax": f"{y2:.3f}",
                    "bbox_width": f"{width:.3f}",
                    "bbox_height": f"{height:.3f}",
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
        draw.text((6, 155), f"{row['source_class_label']} {row['source_split']}"[:28], fill="black")
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
    make_qa_sheet(rows, SHEET_ROOT / "_qa_prashant_tt100k_8each.jpg")

    audit["generated_at"] = datetime.now(timezone.utc).isoformat()
    audit["manifest_path"] = project_rel(MANIFEST_PATH)
    audit["crop_root"] = project_rel(CROP_ROOT)
    audit["review_root"] = project_rel(SHEET_ROOT)
    audit["counts_by_class"] = dict(sorted(Counter(row["semantic_sign_id"] for row in rows).items()))
    audit["note"] = "No AI-generated images. Remote Parquet metadata scan plus exact TT100K labels only; selected remote shards are read only when a candidate is materialized."
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
            "Review Stage C gap-fill 05 TT100K remote contact sheets, then include accepted crops in Stage E split freeze."
        )

    for path in (TRACKER_PATH, TRACKER_BACKUP_PATH):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tracker_rows)
    return dict(counts)


def write_gap_snapshot() -> None:
    csv_path = PROJECT_ROOT / "outputs/audit/post_stage_c_gap_fill_05_gap_report.csv"
    json_path = PROJECT_ROOT / "outputs/audit/post_stage_c_gap_fill_05_gap_report.json"
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
        "note": "Snapshot after Stage C Gap Fill 05. Candidate coverage still requires Stage D QC before Stage E split freeze.",
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    if args.reset:
        safe_reset_path(CROP_ROOT)
        safe_reset_path(SHEET_ROOT)
        MANIFEST_PATH.unlink(missing_ok=True)
        AUDIT_PATH.unlink(missing_ok=True)

    gaps = current_gaps()
    if not any(gaps.values()):
        print("No remaining Stage 05 gaps in current tracker.")
        return

    existing_rows = load_existing_stage_rows()
    refs_by_class, scan_audit = scan_remote_files(args.max_files, gaps, existing_tt100k_markers())
    selected = select_refs(refs_by_class, gaps)
    rows = materialize_refs(selected)
    combined_rows = existing_rows + rows
    audit: dict[str, Any] = {
        "stage_id": STAGE_ID,
        "source_urls": {
            "hf_mirror": SOURCE_URL,
            "original_tt100k": ORIGINAL_DATASET_URL,
            "local_data_yaml": project_rel(DATA_YAML_PATH),
        },
        "gaps_at_start": gaps,
        "available_refs_by_class": {semantic_id: len(refs_by_class.get(semantic_id, [])) for semantic_id in gaps},
        "selected_refs_by_class": dict(sorted(Counter(ref.target.semantic_sign_id for ref in selected).items())),
        "new_counts_by_class": dict(sorted(Counter(row["semantic_sign_id"] for row in rows).items())),
        "existing_stage05_rows_preserved": len(existing_rows),
        "rejected_non_exact_mapping": {
            "width_restriction": "Prashant TT100K pb/pb5 samples were visually rejected because crops showed blank red circles/plates without a clear width-limit value/sign.",
        },
        **scan_audit,
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
