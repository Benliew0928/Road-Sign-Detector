from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
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
STAGE_ID = "stage_c_gap_fill_03_remaining_tt100k"
HF_REPO_ID = "Genius-Society/tt100k"
HF_REPO_TYPE = "dataset"
SOURCE_URL = "https://huggingface.co/datasets/Genius-Society/tt100k"
ORIGINAL_DATASET_URL = "https://cg.cs.tsinghua.edu.cn/traffic-sign/"
LEGEND_URL = "https://lijiancheng0614.github.io/2019/04/16/2019_04_16_TT100K/"
README_PATH = PROJECT_ROOT / "data/raw/online_sources/hf_probe/tt100k/README.md"
SHARD_ROOT = PROJECT_ROOT / "data/raw/online_sources/stage_c_gap_fill_01_tt100k/_hf_arrow_shards"
STAGE_ROOT = PROJECT_ROOT / "data/raw/online_sources/stage_c_gap_fill_03_remaining_tt100k"
CROP_ROOT = STAGE_ROOT / "crops"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_gap_fill_03_remaining_tt100k_candidates.csv"
AUDIT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_gap_fill_03_remaining_tt100k_candidates.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_gap_fill_03_remaining_tt100k"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
TRACKER_BACKUP_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.pending_stage_c_gap_fill_03.csv"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"
MIN_BBOX_SIDE = 26.0


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
    source_index: int
    source_split: str
    source_shard: str
    row_index: int
    object_index: int
    source_image_path: str
    bbox: dict[str, float]
    bbox_width: float
    bbox_height: float


TARGETS: tuple[TargetDef, ...] = (
    TargetDef(
        "no_left_or_right_turn",
        "No left or right turn",
        ("p20",),
        "TT100K p20 icon is the compound no-left/no-right-turn prohibition.",
    ),
    TargetDef(
        "no_motor_vehicles",
        "No motor vehicles",
        ("p10",),
        "TT100K p10 icon is the red-ring motor-vehicle prohibition matching the coursework visual.",
    ),
    TargetDef(
        "no_straight_ahead",
        "No straight ahead",
        ("p14",),
        "TT100K p14 icon is the straight-ahead prohibition matching the coursework visual.",
    ),
    TargetDef(
        "no_straight_or_left",
        "No straight or left turn",
        ("p28",),
        "TT100K p28 icon is the compound no-straight/no-left prohibition matching sign_008.",
    ),
    TargetDef(
        "residential_area_ahead",
        "Residential area ahead",
        ("w3",),
        "TT100K w3 icon is the village/residential-area-ahead warning matching sign_045.",
    ),
    TargetDef(
        "roadway_diverges",
        "Traffic diverges",
        ("w53",),
        "TT100K w53 icon is the split-traffic/traffic-diverges warning matching sign_032.",
    ),
    TargetDef(
        "school_zone",
        "School zone",
        ("w55",),
        "TT100K w55 icon is the children/school-area warning; used for the project school-zone hazard class.",
    ),
    TargetDef(
        "side_road_right",
        "Side road on right",
        ("w16",),
        "TT100K w16 icon is the side-road-on-right warning matching sign_043.",
    ),
    TargetDef(
        "slow_text",
        "Slow",
        ("w30",),
        "TT100K w30 icon contains the Chinese slow warning text matching sign_042.",
    ),
    TargetDef(
        "sound_horn",
        "Sound horn",
        ("i9",),
        "TT100K i9 icon is the blue mandatory sound-horn sign matching sign_029.",
    ),
    TargetDef(
        "steep_descent",
        "Steep descent",
        ("w25",),
        "TT100K w25 icon is the steep-descent warning matching sign_040.",
    ),
    TargetDef(
        "stop_for_checking",
        "Stop for checking",
        ("ps",),
        "TT100K ps crops show red stop-for-checking signs with Chinese inspection text matching sign_057.",
    ),
    TargetDef(
        "tractors_ahead",
        "Tractors ahead",
        ("w34",),
        "TT100K w34 icon is the watch-out-for-tractors warning matching sign_051.",
    ),
    TargetDef(
        "turn_left_or_right",
        "Turn left or right",
        ("i11",),
        "TT100K i11 icon is the blue mandatory turn-left-or-right sign matching sign_023.",
    ),
    TargetDef(
        "width_restriction",
        "Width restriction",
        ("pw2", "pw2.5", "pw3", "pw3.2", "pw3.5", "pw4", "pw4.2", "pw4.5", "pwx"),
        "TT100K pw* labels are parameterized vehicle-width restriction signs; values are kept in source_class_label.",
        "Parameter-specific width values are intentionally merged into the project width_restriction class.",
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
    ("default/train/data-00000-of-00014.arrow", 495_592_880),
    ("default/train/data-00001-of-00014.arrow", 494_831_328),
    ("default/train/data-00002-of-00014.arrow", 498_664_592),
    ("default/train/data-00003-of-00014.arrow", 496_629_000),
    ("default/train/data-00004-of-00014.arrow", 502_568_128),
    ("default/train/data-00005-of-00014.arrow", 504_214_096),
    ("default/train/data-00006-of-00014.arrow", 488_336_040),
    ("default/train/data-00007-of-00014.arrow", 494_414_128),
    ("default/train/data-00008-of-00014.arrow", 499_445_728),
    ("default/train/data-00009-of-00014.arrow", 492_103_344),
    ("default/train/data-00010-of-00014.arrow", 502_478_832),
    ("default/train/data-00011-of-00014.arrow", 499_794_816),
    ("default/train/data-00012-of-00014.arrow", 497_181_288),
    ("default/train/data-00013-of-00014.arrow", 492_555_928),
    ("default/validation/data-00000-of-00003.arrow", 416_288_800),
    ("default/validation/data-00001-of-00003.arrow", 420_258_648),
    ("default/validation/data-00002-of-00003.arrow", 411_413_464),
    ("default/test/data-00000-of-00008.arrow", 443_261_080),
    ("default/test/data-00001-of-00008.arrow", 439_747_760),
    ("default/test/data-00002-of-00008.arrow", 439_118_600),
    ("default/test/data-00003-of-00008.arrow", 435_329_104),
    ("default/test/data-00004-of-00008.arrow", 441_499_216),
    ("default/test/data-00005-of-00008.arrow", 438_325_816),
    ("default/test/data-00006-of-00008.arrow", 441_492_504),
    ("default/test/data-00007-of-00008.arrow", 437_079_304),
)


def project_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def safe_reset_path(path: Path) -> None:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(f"Refusing to reset outside project root: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def label_maps() -> tuple[dict[int, str], dict[str, int]]:
    text = README_PATH.read_text(encoding="utf-8")
    idx_to_label = {
        int(match.group("idx")): match.group("label")
        for match in re.finditer(r"'(?P<idx>\d+)': (?P<label>[^\n\r ]+)", text)
    }
    label_to_idx = {label: idx for idx, label in idx_to_label.items()}
    return idx_to_label, label_to_idx


def discover_shards() -> list[tuple[str, int | None]]:
    try:
        items = list(
            HfApi().list_repo_tree(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                recursive=True,
                expand=True,
            )
        )
        shards = [
            (item.path, getattr(item, "size", None))
            for item in items
            if item.path.endswith(".arrow")
        ]
    except Exception as exc:  # noqa: BLE001 - static fallback keeps the run deterministic.
        print(f"Shard discovery failed, using fallback list: {exc}", flush=True)
        shards = list(FALLBACK_SHARDS)

    split_order = {"train": 0, "validation": 1, "test": 2}

    def sort_key(item: tuple[str, int | None]) -> tuple[int, str]:
        parts = item[0].split("/")
        split = parts[1] if len(parts) > 1 else ""
        return split_order.get(split, 99), item[0]

    return sorted(shards, key=sort_key)


def is_complete_shard(shard_path: str, expected_size: int | None) -> bool:
    local = SHARD_ROOT / shard_path
    if not local.exists():
        return False
    if expected_size is None:
        return local.stat().st_size > 1_000_000
    return local.stat().st_size == expected_size


def download_shard(shard_path: str, expected_size: int | None) -> Path:
    local = SHARD_ROOT / shard_path
    if is_complete_shard(shard_path, expected_size):
        return local

    local.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local.with_suffix(local.suffix + ".part")
    url = hf_hub_url(HF_REPO_ID, shard_path, repo_type=HF_REPO_TYPE)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for resumable TT100K shard downloads")

    command = [
        curl,
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--continue-at",
        "-",
        "--retry",
        "4",
        "--retry-delay",
        "8",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "50000",
        "--speed-time",
        "90",
        "--max-time",
        "1200",
        "--user-agent",
        USER_AGENT,
        "--output",
        str(tmp_path),
        url,
    ]
    for attempt in range(1, 5):
        print(f"Downloading {shard_path} attempt {attempt}/4", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            if expected_size is not None and tmp_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Downloaded size mismatch for {shard_path}: "
                    f"{tmp_path.stat().st_size} != {expected_size}"
                )
            tmp_path.replace(local)
            return local
        time.sleep(8 * attempt)
    raise RuntimeError(f"Failed to download {shard_path}")


def current_gaps() -> dict[str, int]:
    with TRACKER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        gaps = {}
        for row in csv.DictReader(handle):
            if row["semantic_sign_id"] in {target.semantic_sign_id for target in TARGETS}:
                gaps[row["semantic_sign_id"]] = max(0, int(row["gap_to_minimum"]))
    return gaps


def existing_tt100k_markers() -> set[str]:
    markers: set[str] = set()
    manifest_paths = [
        PROJECT_ROOT / "data/manifests/stage_c_gap_fill_01_tt100k_candidates.csv",
        PROJECT_ROOT / "data/manifests/stage_c_gap_fill_02_public_real_sources_candidates.csv",
        MANIFEST_PATH,
    ]
    member_re = re.compile(r"(_hf_arrow_shards/)?(?P<shard>default/.+?\.arrow):(?P<row>\d+):(?P<object>\d+)")
    for path in manifest_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("source_dataset") and "TT100K" not in row["source_dataset"]:
                    continue
                source_shard = row.get("source_shard", "")
                row_index = row.get("row_index_in_shard", "")
                object_index = row.get("object_index", "")
                if source_shard and row_index != "" and object_index != "":
                    markers.add(f"{source_shard}:{row_index}:{object_index}")
                    continue
                member = row.get("source_member_path", "")
                match = member_re.search(member)
                if match:
                    markers.add(
                        f"{match.group('shard')}:{match.group('row')}:{match.group('object')}"
                    )
    return markers


def scan_shards(
    shards: list[tuple[str, int | None]],
    label_to_idx: dict[str, int],
    gaps: dict[str, int],
    used_markers: set[str],
) -> dict[str, list[CandidateRef]]:
    label_to_target = {
        label: target
        for target in TARGETS
        if gaps.get(target.semantic_sign_id, 0) > 0
        for label in target.tt100k_labels
    }
    index_to_target_label = {
        label_to_idx[label]: (target, label)
        for label, target in label_to_target.items()
        if label in label_to_idx
    }

    refs: dict[str, list[CandidateRef]] = defaultdict(list)
    for shard_path, expected_size in shards:
        if not is_complete_shard(shard_path, expected_size):
            continue
        local = SHARD_ROOT / shard_path
        split = shard_path.split("/")[1]
        print(f"Scanning {shard_path}", flush=True)
        df = pl.read_ipc_stream(local)
        for row_index, row in enumerate(df.iter_rows(named=True)):
            objects = row["objects"] or {}
            categories = objects.get("category") or []
            boxes = objects.get("bbox") or []
            for object_index, (category, bbox) in enumerate(zip(categories, boxes)):
                if category not in index_to_target_label:
                    continue
                marker = f"{shard_path}:{row_index}:{object_index}"
                if marker in used_markers:
                    continue
                target, source_label = index_to_target_label[category]
                bbox_width = float(bbox["xmax"]) - float(bbox["xmin"])
                bbox_height = float(bbox["ymax"]) - float(bbox["ymin"])
                if bbox_width < MIN_BBOX_SIDE or bbox_height < MIN_BBOX_SIDE:
                    continue
                refs[target.semantic_sign_id].append(
                    CandidateRef(
                        target=target,
                        source_label=source_label,
                        source_index=category,
                        source_split=split,
                        source_shard=shard_path,
                        row_index=row_index,
                        object_index=object_index,
                        source_image_path=row["image"]["path"] or "",
                        bbox={
                            "xmin": float(bbox["xmin"]),
                            "ymin": float(bbox["ymin"]),
                            "xmax": float(bbox["xmax"]),
                            "ymax": float(bbox["ymax"]),
                        },
                        bbox_width=bbox_width,
                        bbox_height=bbox_height,
                    )
                )
    return refs


def diverse_refs(refs: list[CandidateRef], wanted: int) -> list[CandidateRef]:
    ordered = sorted(
        refs,
        key=lambda ref: (
            ref.source_split,
            ref.source_shard,
            ref.row_index,
            ref.object_index,
            ref.source_label,
        ),
    )
    if len(ordered) <= wanted:
        return ordered
    indices = sorted({round(index * (len(ordered) - 1) / (wanted - 1)) for index in range(wanted)})
    return [ordered[index] for index in indices]


def select_refs(refs_by_class: dict[str, list[CandidateRef]], gaps: dict[str, int]) -> list[CandidateRef]:
    selected: list[CandidateRef] = []
    for semantic_id, refs in sorted(refs_by_class.items()):
        wanted = gaps.get(semantic_id, 0)
        selected.extend(diverse_refs(refs, wanted))
    return selected


def crop_with_context(image: Image.Image, bbox: dict[str, float]) -> Image.Image:
    width, height = image.size
    xmin = max(0, int(bbox["xmin"]))
    ymin = max(0, int(bbox["ymin"]))
    xmax = min(width, int(bbox["xmax"]))
    ymax = min(height, int(bbox["ymax"]))
    box_width = max(1, xmax - xmin)
    box_height = max(1, ymax - ymin)
    dx = max(2, int(box_width * 0.45))
    dy = max(2, int(box_height * 0.45))
    return image.crop(
        (
            max(0, xmin - dx),
            max(0, ymin - dy),
            min(width, xmax + dx),
            min(height, ymax + dy),
        )
    )


def save_crop(crop: Image.Image, path: Path) -> tuple[str, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    crop.convert("RGB").save(buffer, format="JPEG", quality=94, optimize=True)
    content = buffer.getvalue()
    path.write_bytes(content)
    return sha256_bytes(content), crop.width, crop.height


def materialize_refs(selected: list[CandidateRef]) -> list[dict[str, str]]:
    by_shard: dict[str, list[CandidateRef]] = defaultdict(list)
    for ref in selected:
        by_shard[ref.source_shard].append(ref)

    rows: list[dict[str, str]] = []
    class_counts: Counter[str] = Counter()
    for shard_path, refs in sorted(by_shard.items()):
        local = SHARD_ROOT / shard_path
        df = pl.read_ipc_stream(local)
        refs_by_row: dict[int, list[CandidateRef]] = defaultdict(list)
        for ref in refs:
            refs_by_row[ref.row_index].append(ref)
        for row_index, row_refs in refs_by_row.items():
            row = df.row(row_index, named=True)
            image_bytes = row["image"]["bytes"]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            source_sha = sha256_bytes(image_bytes)
            image_width, image_height = image.size
            for ref in row_refs:
                crop = crop_with_context(image, ref.bbox)
                class_counts[ref.target.semantic_sign_id] += 1
                candidate_number = class_counts[ref.target.semantic_sign_id]
                candidate_id = f"TTG03-{ref.source_label}-{candidate_number:04d}"
                crop_path = (
                    CROP_ROOT
                    / ref.target.semantic_sign_id
                    / (
                        f"{candidate_id}_{ref.source_split}_{Path(ref.source_shard).stem}"
                        f"_r{ref.row_index:04d}_o{ref.object_index:02d}.jpg"
                    )
                )
                crop_sha, crop_width, crop_height = save_crop(crop, crop_path)
                rows.append(
                    {
                        "stage_id": STAGE_ID,
                        "candidate_id": candidate_id,
                        "semantic_sign_id": ref.target.semantic_sign_id,
                        "display_name": ref.target.display_name,
                        "source_dataset": "Tsinghua-Tencent 100K / TT100K",
                        "source_repo": HF_REPO_ID,
                        "source_url": SOURCE_URL,
                        "original_dataset_url": ORIGINAL_DATASET_URL,
                        "legend_url": LEGEND_URL,
                        "license_recorded": "HF dataset card: cc-by-nc-4.0; original/other mirrors may state CC BY-NC 2.0",
                        "license_notes": "Academic/non-commercial use only; verify exact licence before public redistribution.",
                        "source_split": ref.source_split,
                        "source_shard": ref.source_shard,
                        "row_index_in_shard": str(ref.row_index),
                        "object_index": str(ref.object_index),
                        "source_image_path": ref.source_image_path,
                        "source_image_sha256": source_sha,
                        "source_class_label": ref.source_label,
                        "source_class_index": str(ref.source_index),
                        "mapping_evidence": ref.target.mapping_evidence,
                        "image_width": str(image_width),
                        "image_height": str(image_height),
                        "bbox_xmin": f"{ref.bbox['xmin']:.3f}",
                        "bbox_ymin": f"{ref.bbox['ymin']:.3f}",
                        "bbox_xmax": f"{ref.bbox['xmax']:.3f}",
                        "bbox_ymax": f"{ref.bbox['ymax']:.3f}",
                        "bbox_width": f"{ref.bbox_width:.3f}",
                        "bbox_height": f"{ref.bbox_height:.3f}",
                        "crop_width": str(crop_width),
                        "crop_height": str(crop_height),
                        "crop_sha256": crop_sha,
                        "local_crop_path": project_rel(crop_path),
                        "source_modality": "real_road_photo_crop_from_detection_dataset",
                        "quality_gate": "accepted_auto_exact_label",
                        "review_status": "auto_exact_label_pending_stage_d_visual_qc",
                        "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
                        "notes": ref.target.label_notes or "Cropped from TT100K object bounding box with context margin; source label retained.",
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
        if len(by_class[row["semantic_sign_id"]]) < 6:
            by_class[row["semantic_sign_id"]].append(row)

    classes = sorted(by_class)
    label_w = 230
    tile_w = 150
    row_h = 172
    sheet = Image.new("RGB", (label_w + 6 * tile_w, max(1, len(classes)) * row_h), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for row_index, semantic_id in enumerate(classes):
        y = row_index * row_h
        draw.rectangle(
            (0, y, label_w + 6 * tile_w, y + row_h - 1),
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
    make_qa_sheet(rows, SHEET_ROOT / "_qa_remaining_15_6each.jpg")

    audit["generated_at"] = datetime.now(timezone.utc).isoformat()
    audit["manifest_path"] = project_rel(MANIFEST_PATH)
    audit["crop_root"] = project_rel(CROP_ROOT)
    audit["review_root"] = project_rel(SHEET_ROOT)
    audit["counts_by_class"] = dict(sorted(Counter(row["semantic_sign_id"] for row in rows).items()))
    audit["note"] = "No AI-generated images. TT100K exact source labels only; selected with class-spread sampling."
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def load_existing_stage_rows() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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
            "Review Stage C gap-fill 03 contact sheets, then include accepted crops in Stage E split freeze."
        )

    for path in (TRACKER_PATH, TRACKER_BACKUP_PATH):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tracker_rows)
    return dict(counts)


def write_gap_snapshot() -> None:
    csv_path = PROJECT_ROOT / "outputs/audit/post_stage_c_gap_fill_03_gap_report.csv"
    json_path = PROJECT_ROOT / "outputs/audit/post_stage_c_gap_fill_03_gap_report.json"
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
        "note": "Snapshot after Stage C Gap Fill 03. Candidate coverage still requires Stage D QC before Stage E split freeze.",
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--max-downloads", type=int, default=3)
    parser.add_argument("--download-all", action="store_true")
    parser.add_argument("--skip-shard", action="append", default=[])
    args = parser.parse_args()

    if args.reset:
        safe_reset_path(STAGE_ROOT)
        safe_reset_path(SHEET_ROOT)
        MANIFEST_PATH.unlink(missing_ok=True)
        AUDIT_PATH.unlink(missing_ok=True)

    _, label_to_idx = label_maps()
    missing_labels = sorted({label for target in TARGETS for label in target.tt100k_labels if label not in label_to_idx})
    if missing_labels:
        raise RuntimeError(f"Missing TT100K labels in README metadata: {missing_labels}")

    gaps = current_gaps()
    if not any(gaps.values()):
        print("No remaining Stage 03 gaps in current tracker.")
        return

    skip_shards = set(args.skip_shard)
    all_shards = [item for item in discover_shards() if item[0] not in skip_shards]
    downloads = 0
    while True:
        complete_shards = [item for item in all_shards if is_complete_shard(*item)]
        used_markers = existing_tt100k_markers()
        refs_by_class = scan_shards(complete_shards, label_to_idx, gaps, used_markers)
        available = {semantic_id: len(refs_by_class.get(semantic_id, [])) for semantic_id in gaps}
        enough = all(available.get(semantic_id, 0) >= gap for semantic_id, gap in gaps.items() if gap > 0)
        print("Available exact candidate refs:", json.dumps(available, indent=2), flush=True)
        if enough:
            break
        if not args.download_all and downloads >= args.max_downloads:
            break
        next_shard = next((item for item in all_shards if not is_complete_shard(*item)), None)
        if next_shard is None:
            break
        download_shard(*next_shard)
        downloads += 1

    selected = select_refs(refs_by_class, gaps)
    rows = materialize_refs(selected)
    existing_rows = load_existing_stage_rows()
    combined_rows = existing_rows + rows
    audit: dict[str, Any] = {
        "stage_id": STAGE_ID,
        "source_urls": {
            "tt100k_hf": SOURCE_URL,
            "tt100k_original": ORIGINAL_DATASET_URL,
            "tt100k_legend": LEGEND_URL,
        },
        "gaps_at_start": gaps,
        "available_refs_by_class": {semantic_id: len(refs_by_class.get(semantic_id, [])) for semantic_id in gaps},
        "selected_refs_by_class": dict(sorted(Counter(ref.target.semantic_sign_id for ref in selected).items())),
        "new_counts_by_class": dict(sorted(Counter(row["semantic_sign_id"] for row in rows).items())),
        "existing_stage03_rows_preserved": len(existing_rows),
        "downloaded_shards_this_run": downloads,
        "complete_shards_scanned": len([item for item in all_shards if is_complete_shard(*item)]),
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
