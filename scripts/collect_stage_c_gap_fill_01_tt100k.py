from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
import requests
from huggingface_hub import HfApi, hf_hub_url
from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_gap_fill_01_tt100k_exact_real_crops"
HF_REPO_ID = "Genius-Society/tt100k"
HF_REPO_TYPE = "dataset"
SOURCE_URL = "https://huggingface.co/datasets/Genius-Society/tt100k"
ORIGINAL_DATASET_URL = "https://cg.cs.tsinghua.edu.cn/traffic-sign/"
LEGEND_URL = "https://lijiancheng0614.github.io/2019/04/16/2019_04_16_TT100K/"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"

STAGE_ROOT = PROJECT_ROOT / "data/raw/online_sources/stage_c_gap_fill_01_tt100k"
SHARD_ROOT = STAGE_ROOT / "_hf_arrow_shards"
CROP_ROOT = STAGE_ROOT / "crops"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_gap_fill_01_tt100k_candidates.csv"
AUDIT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_gap_fill_01_tt100k_candidates.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_gap_fill_01_tt100k"


@dataclass(frozen=True)
class TargetClass:
    semantic_sign_id: str
    display_name: str
    tt100k_label: str
    tt100k_index: int
    mapping_evidence: str


TARGETS: tuple[TargetClass, ...] = (
    TargetClass(
        "bicycle_crossing",
        "Bicycle crossing",
        "w56",
        161,
        "TT100K legend icon w56 shows the triangular bicycle warning sign.",
    ),
    TargetClass(
        "motor_vehicles_only",
        "Motor vehicles only",
        "i4",
        9,
        "TT100K legend icon i4 shows the blue circular motor-vehicle-only sign.",
    ),
    TargetClass(
        "no_left_or_right_turn",
        "No left or right turn",
        "p20",
        32,
        "TT100K legend icon p20 shows the compound no-left/no-right-turn prohibition.",
    ),
    TargetClass(
        "no_motor_vehicles",
        "No motor vehicles",
        "p10",
        21,
        "TT100K legend icon p10 shows the red-ring motor-vehicle prohibition matching the coursework visual.",
    ),
    TargetClass(
        "no_straight_ahead",
        "No straight ahead",
        "p14",
        25,
        "TT100K legend icon p14 shows the straight-ahead prohibition matching the coursework visual.",
    ),
)

TARGET_BY_INDEX = {target.tt100k_index: target for target in TARGETS}

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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_reset_path(path: Path) -> None:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(f"Refusing to reset outside project root: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def discover_shards() -> list[tuple[str, int | None]]:
    api = HfApi()
    items = None
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            items = list(
                api.list_repo_tree(
                    repo_id=HF_REPO_ID,
                    repo_type=HF_REPO_TYPE,
                    recursive=True,
                    expand=True,
                )
            )
            break
        except Exception as exc:  # noqa: BLE001 - network fallback is expected here.
            last_error = exc
            print(f"Shard discovery failed attempt {attempt}/3: {exc}", flush=True)
            time.sleep(6 * attempt)
    if items is None:
        print(f"Using static TT100K shard fallback after discovery failure: {last_error}", flush=True)
        return list(FALLBACK_SHARDS)

    shards = [
        (item.path, getattr(item, "size", None))
        for item in items
        if item.path.endswith(".arrow")
    ]
    split_order = {"train": 0, "validation": 1, "test": 2}

    def sort_key(item: tuple[str, int | None]) -> tuple[int, str]:
        path = item[0]
        parts = path.split("/")
        split = parts[1] if len(parts) > 1 else ""
        return split_order.get(split, 99), path

    return sorted(shards, key=sort_key)


def download_shard(shard_path: str, expected_size: int | None, *, retries: int = 4) -> Path:
    local_shard = SHARD_ROOT / shard_path
    if local_shard.exists() and local_shard.stat().st_size > 1_000_000:
        if expected_size is None or local_shard.stat().st_size == expected_size:
            return local_shard

    local_shard.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_shard.with_suffix(local_shard.suffix + ".part")
    url = hf_hub_url(HF_REPO_ID, shard_path, repo_type=HF_REPO_TYPE)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        if attempt > 1:
            time.sleep(8 * attempt)
        try:
            print(f"Downloading {shard_path} via HTTPS attempt {attempt}/{retries}", flush=True)
            curl = shutil.which("curl.exe") or shutil.which("curl")
            if curl:
                command = [
                    curl,
                    "--fail",
                    "--location",
                    "--continue-at",
                    "-",
                    "--retry",
                    "3",
                    "--retry-delay",
                    "8",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "900",
                    "--user-agent",
                    USER_AGENT,
                    "--output",
                    str(tmp_path),
                    url,
                ]
                completed = subprocess.run(command, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(f"curl returned exit code {completed.returncode}")
                if expected_size is not None and tmp_path.stat().st_size != expected_size:
                    raise RuntimeError(
                        f"Downloaded size mismatch for {shard_path}: "
                        f"{tmp_path.stat().st_size} != {expected_size}"
                    )
                tmp_path.replace(local_shard)
                return local_shard

            with requests.get(
                url,
                stream=True,
                timeout=(30, 90),
                headers={"User-Agent": USER_AGENT},
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                downloaded = 0
                next_report = 100 * 1024 * 1024
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            print(
                                f"  {shard_path}: {downloaded / (1024 * 1024):.0f} MB",
                                flush=True,
                            )
                            next_report += 100 * 1024 * 1024
            if expected_size is not None and tmp_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Downloaded size mismatch for {shard_path}: "
                    f"{tmp_path.stat().st_size} != {expected_size}"
                )
            tmp_path.replace(local_shard)
            return local_shard
        except Exception as exc:  # noqa: BLE001 - preserve retry context for the audit log.
            last_error = exc
            print(f"Download failed for {shard_path}: {exc}", flush=True)
    raise RuntimeError(f"Failed to download {shard_path}") from last_error


def load_arrow(path: Path) -> pl.DataFrame:
    return pl.read_ipc_stream(path)


def crop_with_context(
    image: Image.Image,
    bbox: dict[str, float],
    margin_ratio: float,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    xmin = max(0, int(bbox["xmin"]))
    ymin = max(0, int(bbox["ymin"]))
    xmax = min(width, int(bbox["xmax"]))
    ymax = min(height, int(bbox["ymax"]))
    box_width = max(1, xmax - xmin)
    box_height = max(1, ymax - ymin)
    dx = max(2, int(box_width * margin_ratio))
    dy = max(2, int(box_height * margin_ratio))
    left = max(0, xmin - dx)
    top = max(0, ymin - dy)
    right = min(width, xmax + dx)
    bottom = min(height, ymax + dy)
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def save_crop(crop: Image.Image, path: Path) -> tuple[str, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    crop.convert("RGB").save(buffer, format="JPEG", quality=94, optimize=True)
    content = buffer.getvalue()
    path.write_bytes(content)
    return sha256_bytes(content), crop.width, crop.height


def make_contact_sheet(rows: list[dict[str, str]], output_path: Path, *, max_items: int = 140) -> None:
    selected = rows[:max_items]
    thumb_w, thumb_h = 150, 150
    tile_w, tile_h = 190, 190
    cols = 5
    sheet_rows = max(1, (len(selected) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tile_w, sheet_rows * tile_h), (236, 236, 236))
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(selected):
        crop_path = PROJECT_ROOT / row["local_crop_path"]
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        try:
            image = Image.open(crop_path).convert("RGB")
            image = ImageOps.contain(image, (thumb_w, thumb_h), Image.Resampling.LANCZOS)
            tile.paste(image, ((tile_w - image.width) // 2, 6))
        except OSError:
            pass
        label = f"{row['source_class_label']} {row['source_split']}"
        draw_tile = ImageDraw.Draw(tile)
        draw_tile.text((6, 160), label[:28], fill="black")
        draw_tile.text((6, 175), row["candidate_id"], fill="black")
        x = (index % cols) * tile_w
        y = (index // cols) * tile_h
        sheet.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def build_combined_sheet(rows: list[dict[str, str]], output_path: Path) -> None:
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["semantic_sign_id"], []).append(row)

    class_tiles: list[Image.Image] = []
    for semantic_id in sorted(by_class):
        sample_rows = by_class[semantic_id][:18]
        tile = Image.new("RGB", (960, 430), (245, 245, 245))
        ImageDraw.Draw(tile).text((14, 12), f"{semantic_id} ({len(by_class[semantic_id])})", fill="black")
        for idx, row in enumerate(sample_rows):
            try:
                image = Image.open(PROJECT_ROOT / row["local_crop_path"]).convert("RGB")
                image = ImageOps.contain(image, (120, 120), Image.Resampling.LANCZOS)
            except OSError:
                image = Image.new("RGB", (120, 120), (220, 220, 220))
            x = 14 + (idx % 6) * 155
            y = 44 + (idx // 6) * 130
            tile.paste(image, (x + (120 - image.width) // 2, y))
            ImageDraw.Draw(tile).text((x, y + 122), row["candidate_id"], fill="black")
        class_tiles.append(tile)

    combined = Image.new("RGB", (960, max(1, len(class_tiles)) * 430), (235, 235, 235))
    for idx, tile in enumerate(class_tiles):
        combined.paste(tile, (0, idx * 430))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path, quality=92)


def write_outputs(
    rows: list[dict[str, str]],
    audit: dict[str, Any],
    counts: Counter[str],
) -> None:
    rows.sort(key=lambda row: (row["semantic_sign_id"], row["candidate_id"]))
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["semantic_sign_id"], []).append(row)

    for semantic_id, class_rows in by_class.items():
        make_contact_sheet(class_rows, SHEET_ROOT / f"{semantic_id}.jpg")
    if rows:
        build_combined_sheet(rows, SHEET_ROOT / "_all_tt100k_gap_fill_candidates.jpg")

    audit["generated_at"] = datetime.now(timezone.utc).isoformat()
    audit["counts_by_class"] = dict(sorted(counts.items()))
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def extract_candidates(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if args.reset:
        safe_reset_path(STAGE_ROOT)
        safe_reset_path(SHEET_ROOT)
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()
        if AUDIT_PATH.exists():
            AUDIT_PATH.unlink()

    CROP_ROOT.mkdir(parents=True, exist_ok=True)
    SHARD_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHEET_ROOT.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    seen_crop_hashes: set[str] = set()
    shards = discover_shards()
    if args.max_shards:
        shards = shards[: args.max_shards]

    print(f"Discovered {len(shards)} TT100K Arrow shards.")
    audit = {
        "stage_id": STAGE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_repo": HF_REPO_ID,
        "source_url": SOURCE_URL,
        "original_dataset_url": ORIGINAL_DATASET_URL,
        "legend_url": LEGEND_URL,
        "target_per_class": args.target_per_class,
        "min_bbox_px": args.min_bbox_px,
        "review_bbox_px": args.review_bbox_px,
        "crop_margin": args.crop_margin,
        "processed_shards": [],
        "counts_by_class": {},
        "skipped": {},
        "manifest_path": project_rel(MANIFEST_PATH),
        "crop_root": project_rel(CROP_ROOT),
        "review_root": project_rel(SHEET_ROOT),
        "mapping": [
            {
                "semantic_sign_id": target.semantic_sign_id,
                "display_name": target.display_name,
                "tt100k_label": target.tt100k_label,
                "tt100k_index": target.tt100k_index,
                "mapping_evidence": target.mapping_evidence,
            }
            for target in TARGETS
        ],
    }

    for shard_index, (shard_path, expected_size) in enumerate(shards, start=1):
        print(f"[{shard_index}/{len(shards)}] Downloading {shard_path}", flush=True)
        local_shard = download_shard(shard_path, expected_size)
        print(f"Processing {local_shard}")
        df = load_arrow(local_shard)
        split = shard_path.split("/")[1]
        audit["processed_shards"].append(shard_path)

        for row_index, row in enumerate(df.iter_rows(named=True)):
            objects = row["objects"] or {}
            categories = objects.get("category") or []
            boxes = objects.get("bbox") or []
            target_positions = [
                (object_index, category, bbox)
                for object_index, (category, bbox) in enumerate(zip(categories, boxes))
                if category in TARGET_BY_INDEX
            ]
            if not target_positions:
                continue

            image_bytes = row["image"]["bytes"]
            image_sha = sha256_bytes(image_bytes)
            source_image_path = row["image"].get("path", "")
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_width, image_height = image.size

            for object_index, category, bbox in target_positions:
                target = TARGET_BY_INDEX[category]
                bbox_width = float(bbox["xmax"]) - float(bbox["xmin"])
                bbox_height = float(bbox["ymax"]) - float(bbox["ymin"])
                if bbox_width < args.min_bbox_px or bbox_height < args.min_bbox_px:
                    skipped[f"{target.semantic_sign_id}:too_tiny_bbox"] += 1
                    continue

                crop, crop_box = crop_with_context(image, bbox, args.crop_margin)
                candidate_number = counts[target.semantic_sign_id] + 1
                candidate_id = f"TTG01-{target.tt100k_label}-{candidate_number:04d}"
                filename = (
                    f"{candidate_id}_{split}_{Path(shard_path).stem}_r{row_index:04d}"
                    f"_o{object_index:02d}.jpg"
                )
                crop_path = CROP_ROOT / target.semantic_sign_id / filename

                temp_buffer = BytesIO()
                crop.convert("RGB").save(temp_buffer, format="JPEG", quality=94, optimize=True)
                crop_sha = sha256_bytes(temp_buffer.getvalue())
                if crop_sha in seen_crop_hashes:
                    skipped[f"{target.semantic_sign_id}:duplicate_crop_sha"] += 1
                    continue
                seen_crop_hashes.add(crop_sha)

                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop_path.write_bytes(temp_buffer.getvalue())
                counts[target.semantic_sign_id] += 1

                quality_gate = "accepted_auto_exact_label"
                if bbox_width < args.review_bbox_px or bbox_height < args.review_bbox_px:
                    quality_gate = "accepted_but_small_bbox_review"

                rows.append(
                    {
                        "stage_id": STAGE_ID,
                        "candidate_id": candidate_id,
                        "semantic_sign_id": target.semantic_sign_id,
                        "display_name": target.display_name,
                        "source_dataset": "Tsinghua-Tencent 100K / TT100K",
                        "source_repo": HF_REPO_ID,
                        "source_url": SOURCE_URL,
                        "original_dataset_url": ORIGINAL_DATASET_URL,
                        "legend_url": LEGEND_URL,
                        "license_recorded": "HF dataset card: cc-by-nc-4.0; original/other mirrors may state CC BY-NC 2.0",
                        "license_notes": "Academic/non-commercial use only; verify exact licence before public redistribution.",
                        "source_split": split,
                        "source_shard": shard_path,
                        "row_index_in_shard": str(row_index),
                        "object_index": str(object_index),
                        "source_image_path": source_image_path,
                        "source_image_sha256": image_sha,
                        "source_class_label": target.tt100k_label,
                        "source_class_index": str(target.tt100k_index),
                        "mapping_evidence": target.mapping_evidence,
                        "image_width": str(image_width),
                        "image_height": str(image_height),
                        "bbox_xmin": f"{float(bbox['xmin']):.3f}",
                        "bbox_ymin": f"{float(bbox['ymin']):.3f}",
                        "bbox_xmax": f"{float(bbox['xmax']):.3f}",
                        "bbox_ymax": f"{float(bbox['ymax']):.3f}",
                        "bbox_width": f"{bbox_width:.3f}",
                        "bbox_height": f"{bbox_height:.3f}",
                        "crop_width": str(crop.width),
                        "crop_height": str(crop.height),
                        "crop_sha256": crop_sha,
                        "local_crop_path": project_rel(crop_path),
                        "source_modality": "real_road_photo_crop_from_detection_dataset",
                        "quality_gate": quality_gate,
                        "review_status": "auto_exact_label_pending_stage_d_visual_qc",
                        "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
                        "notes": "Cropped from TT100K object bounding box with context margin; source label retained.",
                    }
                )

        audit["skipped"] = dict(sorted(skipped.items()))
        write_outputs(rows, audit, counts)
        print("Current counts:", dict(sorted(counts.items())), flush=True)
        if args.delete_shards:
            local_shard.unlink(missing_ok=True)

        if all(counts[target.semantic_sign_id] >= args.target_per_class for target in TARGETS):
            print(f"Reached target_per_class={args.target_per_class}; stopping early.")
            break

    write_outputs(rows, audit, counts)

    return rows, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect exact real-road TT100K crops for five Stage C gap classes."
    )
    parser.add_argument("--target-per-class", type=int, default=100)
    parser.add_argument("--min-bbox-px", type=int, default=26)
    parser.add_argument("--review-bbox-px", type=int, default=40)
    parser.add_argument("--crop-margin", type=float, default=0.35)
    parser.add_argument("--max-shards", type=int, default=0, help="0 means all discovered shards.")
    parser.add_argument("--delete-shards", action="store_true", help="Delete local Arrow shard after processing.")
    parser.add_argument("--reset", action="store_true", help="Reset only this sprint's output folders first.")
    return parser.parse_args()


def main() -> None:
    rows, audit = extract_candidates(parse_args())
    print(f"Wrote {len(rows)} candidates to {MANIFEST_PATH}")
    print("Final counts:", audit["counts_by_class"])
    print(f"Review sheets: {SHEET_ROOT}")


if __name__ == "__main__":
    main()
