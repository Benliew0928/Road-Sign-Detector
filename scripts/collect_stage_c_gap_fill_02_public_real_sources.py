from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
import requests
from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_gap_fill_02_public_real_sources"

GTSRB_URL = (
    "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
    "GTSRB-Training_fixed.zip"
)
GTSRB_OFFICIAL_URL = "https://benchmark.ini.rub.de/"
GTSRB_MD5 = "513f3c79a4c5141765e10e952eaa2478"
TT100K_HF_URL = "https://huggingface.co/datasets/Genius-Society/tt100k"
TT100K_ORIGINAL_URL = "https://cg.cs.tsinghua.edu.cn/traffic-sign/"
TT100K_LEGEND_URL = "https://lijiancheng0614.github.io/2019/04/16/2019_04_16_TT100K/"

RAW_ROOT = PROJECT_ROOT / "data/raw/online_sources/stage_c_gap_fill_02_public_real_sources"
GTSRB_ZIP_PATH = RAW_ROOT / "gtsrb/GTSRB-Training_fixed.zip"
TT100K_SHARD_ROOT = PROJECT_ROOT / "data/raw/online_sources/stage_c_gap_fill_01_tt100k/_hf_arrow_shards"
CROP_ROOT = RAW_ROOT / "crops"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_gap_fill_02_public_real_sources_candidates.csv"
AUDIT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_gap_fill_02_public_real_sources_candidates.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_gap_fill_02_public_real_sources"
TRACKER_IN = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.pending_stage_c_gap_fill_01.csv"
TRACKER_FALLBACK_IN = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
TRACKER_PENDING_OUT = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.pending_stage_c_gap_fill_02.csv"

USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"


@dataclass(frozen=True)
class GtsrbTarget:
    semantic_sign_id: str
    display_name: str
    class_id: int
    source_label: str
    target_new_count: int
    mapping_evidence: str


@dataclass(frozen=True)
class Tt100kTarget:
    semantic_sign_id: str
    display_name: str
    class_index: int
    source_label: str
    target_new_count: int
    mapping_evidence: str


GTSRB_TARGETS: tuple[GtsrbTarget, ...] = (
    GtsrbTarget(
        "bicycle_crossing",
        "Bicycle crossing",
        29,
        "Bicycles crossing",
        50,
        "GTSRB class 29 is the real-road bicycle-crossing warning sign.",
    ),
    GtsrbTarget(
        "no_heavy_vehicle",
        "No heavy vehicles",
        16,
        "Vehicles over 3.5 metric tons prohibited",
        50,
        "GTSRB class 16 is the heavy-vehicle prohibition sign.",
    ),
    GtsrbTarget(
        "uneven_road",
        "Uneven road",
        22,
        "Bumpy road",
        50,
        "GTSRB class 22 is the bumpy/uneven-road warning sign.",
    ),
    GtsrbTarget(
        "turn_left",
        "Turn left",
        34,
        "Turn left ahead",
        50,
        "GTSRB class 34 is a blue mandatory turn-left sign.",
    ),
    GtsrbTarget(
        "turn_right",
        "Turn right",
        33,
        "Turn right ahead",
        50,
        "GTSRB class 33 is a blue mandatory turn-right sign.",
    ),
    GtsrbTarget(
        "straight_ahead",
        "Straight ahead only",
        35,
        "Ahead only",
        50,
        "GTSRB class 35 is a blue mandatory ahead-only sign.",
    ),
    GtsrbTarget(
        "straight_or_right",
        "Straight or right",
        36,
        "Go straight or right",
        50,
        "GTSRB class 36 is a blue mandatory straight-or-right sign.",
    ),
    GtsrbTarget(
        "roundabout_mandatory",
        "Roundabout",
        40,
        "Roundabout mandatory",
        50,
        "GTSRB class 40 is the blue mandatory roundabout sign.",
    ),
    GtsrbTarget(
        "straight_or_left",
        "Straight or left",
        37,
        "Go straight or left",
        50,
        "GTSRB class 37 is a blue mandatory straight-or-left sign.",
    ),
)

TT100K_TARGETS: tuple[Tt100kTarget, ...] = (
    Tt100kTarget(
        "no_bicycle",
        "No bicycles",
        44,
        "p6",
        20,
        "TT100K legend icon p6 is the red-ring no-bicycle prohibition sign.",
    ),
)

FIELDNAMES = [
    "stage_id",
    "candidate_id",
    "semantic_sign_id",
    "display_name",
    "source_dataset",
    "source_url",
    "license_notes",
    "source_split",
    "source_class_id",
    "source_class_label",
    "mapping_evidence",
    "source_member_path",
    "source_sha256",
    "local_crop_path",
    "crop_sha256",
    "crop_width",
    "crop_height",
    "source_modality",
    "review_status",
    "counts_for_candidate_coverage",
    "notes",
]


def project_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - provenance checksum, not security.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_reset(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    resolved = path.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(f"Refusing to reset outside workspace: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def download_with_curl(url: str, output_path: Path, expected_md5: str | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        if expected_md5 is None or file_md5(output_path).lower() == expected_md5.lower():
            return

    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for robust resumable downloads")

    command = [
        curl,
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        "5",
        "--retry-delay",
        "8",
        "--connect-timeout",
        "30",
        "--max-time",
        "1800",
        "--user-agent",
        USER_AGENT,
        "--output",
        str(output_path),
        url,
    ]
    for attempt in range(1, 4):
        print(f"Downloading {url} attempt {attempt}/3", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            if expected_md5 and file_md5(output_path).lower() != expected_md5.lower():
                raise RuntimeError(f"MD5 mismatch for {output_path}")
            return
        time.sleep(8 * attempt)
    raise RuntimeError(f"curl failed repeatedly for {url}")


def save_jpeg(image: Image.Image, path: Path) -> tuple[str, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=94, optimize=True)
    content = buffer.getvalue()
    path.write_bytes(content)
    return sha256_bytes(content), image.width, image.height


def collect_gtsrb() -> list[dict[str, str]]:
    download_with_curl(GTSRB_URL, GTSRB_ZIP_PATH, GTSRB_MD5)
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    targets_by_folder = {f"{target.class_id:05d}": target for target in GTSRB_TARGETS}
    seen_hashes: set[str] = set()

    with zipfile.ZipFile(GTSRB_ZIP_PATH) as archive:
        members_by_folder: dict[str, list[str]] = {folder: [] for folder in targets_by_folder}
        for member in archive.namelist():
            if not member.lower().endswith(".ppm") or len(Path(member).parts) < 3:
                continue
            folder = Path(member).parts[-2]
            if folder in targets_by_folder:
                members_by_folder[folder].append(member)

        for folder, folder_members in members_by_folder.items():
            target = targets_by_folder[folder]
            for member in diverse_member_order(sorted(folder_members), target.target_new_count):
                if counts[target.semantic_sign_id] >= target.target_new_count:
                    break
                source_bytes = archive.read(member)
                source_sha = sha256_bytes(source_bytes)
                if source_sha in seen_hashes:
                    continue
                seen_hashes.add(source_sha)
                image = Image.open(BytesIO(source_bytes)).convert("RGB")
                candidate_number = counts[target.semantic_sign_id] + 1
                candidate_id = f"GTSRB02-{target.class_id:02d}-{candidate_number:04d}"
                crop_path = (
                    CROP_ROOT
                    / target.semantic_sign_id
                    / f"{candidate_id}_{Path(member).stem}.jpg"
                )
                crop_sha, crop_width, crop_height = save_jpeg(image, crop_path)
                counts[target.semantic_sign_id] += 1
                rows.append(
                    {
                        "stage_id": STAGE_ID,
                        "candidate_id": candidate_id,
                        "semantic_sign_id": target.semantic_sign_id,
                        "display_name": target.display_name,
                        "source_dataset": "German Traffic Sign Recognition Benchmark / GTSRB",
                        "source_url": GTSRB_OFFICIAL_URL,
                        "license_notes": "GTSRB official site states free use with required citation; keep for academic/non-commercial project use.",
                        "source_split": "train",
                        "source_class_id": str(target.class_id),
                        "source_class_label": target.source_label,
                        "mapping_evidence": target.mapping_evidence,
                        "source_member_path": member,
                        "source_sha256": source_sha,
                        "local_crop_path": project_rel(crop_path),
                        "crop_sha256": crop_sha,
                        "crop_width": str(crop_width),
                        "crop_height": str(crop_height),
                        "source_modality": "real_road_photo_crop_classification_dataset",
                        "review_status": "auto_exact_label_pending_stage_d_visual_qc",
                        "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
                        "notes": "GTSRB image is already a cropped sign classification sample.",
                    }
                )
    return rows


def diverse_member_order(members: list[str], wanted: int) -> list[str]:
    """Prefer files spread across a source class instead of one near-duplicate run."""
    if len(members) <= wanted or wanted <= 1:
        return members

    preferred_indices = {
        round(index * (len(members) - 1) / (wanted - 1)) for index in range(wanted)
    }
    preferred = [members[index] for index in sorted(preferred_indices)]
    remainder = [member for index, member in enumerate(members) if index not in preferred_indices]
    return preferred + remainder


def tt100k_shards() -> list[Path]:
    return sorted(TT100K_SHARD_ROOT.glob("default/*/data-*.arrow"))


def collect_tt100k_local() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    target_by_index = {target.class_index: target for target in TT100K_TARGETS}
    seen_hashes: set[str] = set()

    for shard_path in tt100k_shards():
        if shard_path.suffix != ".arrow":
            continue
        split = shard_path.parts[-2]
        df = pl.read_ipc_stream(shard_path)
        for row_index, row in enumerate(df.iter_rows(named=True)):
            objects = row["objects"] or {}
            categories = objects.get("category") or []
            boxes = objects.get("bbox") or []
            if not any(category in target_by_index for category in categories):
                continue
            image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
            image_width, image_height = image.size
            for object_index, (category, bbox) in enumerate(zip(categories, boxes)):
                if category not in target_by_index:
                    continue
                target = target_by_index[category]
                if counts[target.semantic_sign_id] >= target.target_new_count:
                    continue
                bbox_width = float(bbox["xmax"]) - float(bbox["xmin"])
                bbox_height = float(bbox["ymax"]) - float(bbox["ymin"])
                if bbox_width < 26 or bbox_height < 26:
                    continue
                left = max(0, int(bbox["xmin"] - bbox_width * 0.35))
                top = max(0, int(bbox["ymin"] - bbox_height * 0.35))
                right = min(image_width, int(bbox["xmax"] + bbox_width * 0.35))
                bottom = min(image_height, int(bbox["ymax"] + bbox_height * 0.35))
                crop = image.crop((left, top, right, bottom))
                source_marker = f"{shard_path.as_posix()}:{row_index}:{object_index}".encode()
                source_sha = sha256_bytes(row["image"]["bytes"] + source_marker)
                if source_sha in seen_hashes:
                    continue
                seen_hashes.add(source_sha)
                candidate_number = counts[target.semantic_sign_id] + 1
                candidate_id = f"TTG02-{target.source_label}-{candidate_number:04d}"
                crop_path = (
                    CROP_ROOT
                    / target.semantic_sign_id
                    / f"{candidate_id}_{split}_{shard_path.stem}_r{row_index:04d}_o{object_index:02d}.jpg"
                )
                crop_sha, crop_width, crop_height = save_jpeg(crop, crop_path)
                counts[target.semantic_sign_id] += 1
                rows.append(
                    {
                        "stage_id": STAGE_ID,
                        "candidate_id": candidate_id,
                        "semantic_sign_id": target.semantic_sign_id,
                        "display_name": target.display_name,
                        "source_dataset": "Tsinghua-Tencent 100K / TT100K",
                        "source_url": TT100K_HF_URL,
                        "license_notes": "HF card records CC BY-NC 4.0; original/other mirrors may state CC BY-NC 2.0. Academic/non-commercial use only.",
                        "source_split": split,
                        "source_class_id": str(target.class_index),
                        "source_class_label": target.source_label,
                        "mapping_evidence": target.mapping_evidence,
                        "source_member_path": f"{project_rel(shard_path)}:{row_index}:{object_index}",
                        "source_sha256": source_sha,
                        "local_crop_path": project_rel(crop_path),
                        "crop_sha256": crop_sha,
                        "crop_width": str(crop_width),
                        "crop_height": str(crop_height),
                        "source_modality": "real_road_photo_crop_detection_dataset",
                        "review_status": "auto_exact_label_pending_stage_d_visual_qc",
                        "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
                        "notes": "Cropped from TT100K object bounding box with context margin.",
                    }
                )
    return rows


def make_contact_sheet(rows: list[dict[str, str]], path: Path, max_items: int = 80) -> None:
    selected = rows[:max_items]
    cols = 5
    tile_w, tile_h = 180, 190
    sheet_rows = max(1, (len(selected) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tile_w, sheet_rows * tile_h), (235, 235, 235))
    for index, row in enumerate(selected):
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        try:
            image = Image.open(PROJECT_ROOT / row["local_crop_path"]).convert("RGB")
            image = ImageOps.contain(image, (140, 140), Image.Resampling.LANCZOS)
            tile.paste(image, ((tile_w - image.width) // 2, 6))
        except OSError:
            pass
        draw = ImageDraw.Draw(tile)
        draw.text((6, 152), row["candidate_id"][:26], fill="black")
        draw.text((6, 168), row["source_class_label"][:26], fill="black")
        sheet.paste(tile, ((index % cols) * tile_w, (index // cols) * tile_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def update_tracker(rows: list[dict[str, str]]) -> bool:
    source_path = TRACKER_IN if TRACKER_IN.exists() else TRACKER_FALLBACK_IN
    counts = Counter(row["semantic_sign_id"] for row in rows)

    with source_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames_list = reader.fieldnames or []
        tracker_rows = list(reader)

    filled = 0
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
        if new_total >= minimum:
            tracker_row["collection_status"] = "meets_minimum_pending_qc"
            filled += 1
        else:
            tracker_row["collection_status"] = "still_below_minimum"
        tracker_row["next_action"] = (
            "Review Stage C gap-fill 02 contact sheet, then include accepted crops in Stage E split freeze."
        )

    TRACKER_PENDING_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TRACKER_PENDING_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_list)
        writer.writeheader()
        writer.writerows(tracker_rows)
    return filled >= 10


def write_outputs(rows: list[dict[str, str]]) -> dict[str, Any]:
    rows.sort(key=lambda row: (row["semantic_sign_id"], row["candidate_id"]))
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["semantic_sign_id"], []).append(row)
    for semantic_id, class_rows in by_class.items():
        make_contact_sheet(class_rows, SHEET_ROOT / f"{semantic_id}.jpg")
    make_contact_sheet(rows, SHEET_ROOT / "_all_gap_fill_02_candidates.jpg", max_items=120)

    counts = dict(sorted(Counter(row["semantic_sign_id"] for row in rows).items()))
    audit = {
        "stage_id": STAGE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts_by_class": counts,
        "source_urls": {
            "gtsrb_official": GTSRB_OFFICIAL_URL,
            "gtsrb_download": GTSRB_URL,
            "tt100k_hf": TT100K_HF_URL,
            "tt100k_original": TT100K_ORIGINAL_URL,
            "tt100k_legend": TT100K_LEGEND_URL,
        },
        "manifest_path": project_rel(MANIFEST_PATH),
        "crop_root": project_rel(CROP_ROOT),
        "review_root": project_rel(SHEET_ROOT),
        "pending_tracker": project_rel(TRACKER_PENDING_OUT),
        "note": "No AI-generated images. GTSRB samples are cropped real classification images; TT100K samples are detection crops from local completed shards.",
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        safe_reset(CROP_ROOT)
        safe_reset(SHEET_ROOT)
        MANIFEST_PATH.unlink(missing_ok=True)
        AUDIT_PATH.unlink(missing_ok=True)

    CROP_ROOT.mkdir(parents=True, exist_ok=True)
    rows = collect_gtsrb() + collect_tt100k_local()
    audit = write_outputs(rows)
    filled_ok = update_tracker(rows)
    print(f"Wrote {len(rows)} candidates")
    print(json.dumps(audit["counts_by_class"], indent=2))
    print(f"Filled at least 10 tracked class gaps: {filled_ok}")


if __name__ == "__main__":
    main()
