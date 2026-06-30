from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_gap_fill_01_commons_official_exact_references"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"

RAW_ROOT = PROJECT_ROOT / "data/raw/online_sources/stage_c_gap_fill_01_commons_official"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_gap_fill_01_commons_official_candidates.csv"
AUDIT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_gap_fill_01_commons_official_candidates.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_gap_fill_01_commons_official"


@dataclass(frozen=True)
class Candidate:
    semantic_sign_id: str
    title: str
    source_modality: str
    mapping_confidence: str
    notes: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "bicycle_crossing",
        "File:CN road sign 警 13.svg",
        "official_style_reference_diagram",
        "high_exact_assignment_style",
        "China/GB-style triangular cyclist warning reference.",
    ),
    Candidate(
        "bicycle_crossing",
        "File:SA road sign - Bicycle crossing.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Official-style bicycle crossing sign.",
    ),
    Candidate(
        "bicycle_crossing",
        "File:Taiwan (ROC) road sign W39.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Taiwan warning sign for bicycle crossing.",
    ),
    Candidate(
        "bicycle_crossing",
        "File:Taiwan (ROC) road sign W39 Other Version.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Taiwan alternate warning sign for bicycle crossing.",
    ),
    Candidate(
        "bicycle_crossing",
        "File:Korean Sign - Bicycle Crossing 1.jpg",
        "real_photo_or_scanned_reference",
        "high_exact_meaning",
        "Korean bicycle crossing sign reference discovered by Commons search.",
    ),
    Candidate(
        "bicycle_crossing",
        "File:Korean Sign - Bicycle Crossing 2.jpg",
        "real_photo_or_scanned_reference",
        "high_exact_meaning",
        "Korean bicycle crossing sign reference discovered by Commons search.",
    ),
    Candidate(
        "bicycle_crossing",
        "File:Korean Sign - Bicycle Crossing 3.jpg",
        "real_photo_or_scanned_reference",
        "high_exact_meaning",
        "Korean bicycle crossing sign reference discovered by Commons search.",
    ),
    Candidate(
        "bicycle_crossing",
        "File:Bicycle crossing (Romania).png",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Official-style bicycle crossing warning reference.",
    ),
    Candidate(
        "bicycle_crossing",
        "File:Portugal road sign A17a.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Portugal official cycle crossing warning sign.",
    ),
    Candidate(
        "bicycle_crossing",
        "File:Finland road sign B7.svg",
        "official_style_reference_diagram",
        "medium_check_visual_style",
        "Finland cycle crossing sign; verify exact visual use before final training.",
    ),
    Candidate(
        "motor_vehicles_only",
        "File:Taiwan (ROC) road sign R23.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Official-style motor vehicles only sign.",
    ),
    Candidate(
        "motor_vehicles_only",
        "File:Taiwan (ROC) road sign R23.3.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Official-style motor vehicles only sign.",
    ),
    Candidate(
        "motor_vehicles_only",
        "File:Taiwan (ROC) road sign R23 Other Version.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Official-style motor vehicles only alternate sign.",
    ),
    Candidate(
        "motor_vehicles_only",
        "File:KR road sign 301.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Korean official motor vehicles only sign.",
    ),
    Candidate(
        "motor_vehicles_only",
        "File:KR road sign 301 (obsolete).svg",
        "official_style_reference_diagram",
        "medium_obsolete_style",
        "Obsolete but visually relevant motor vehicles only sign.",
    ),
    Candidate(
        "motor_vehicles_only",
        "File:Japan road sign 325.svg",
        "official_style_reference_diagram",
        "medium_check_visual_style",
        "Japanese road sign discovered by motor vehicles only search.",
    ),
    Candidate(
        "no_left_or_right_turn",
        "File:China road sign 禁 25.svg",
        "official_style_reference_diagram",
        "high_exact_assignment_style",
        "China/GB-style no left and right turns reference matching coursework sign_012.",
    ),
    Candidate(
        "no_left_or_right_turn",
        "File:Vietnam road sign P137.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Vietnam no left/right turn official-style sign.",
    ),
    Candidate(
        "no_motor_vehicles",
        "File:CN road sign 禁 6.svg",
        "official_style_reference_diagram",
        "high_exact_assignment_style",
        "China/GB-style car/motor-vehicle prohibition matching coursework sign_016.",
    ),
    Candidate(
        "no_motor_vehicles",
        "File:Taiwan (ROC) road sign P2 Other Version.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Taiwan no motor vehicles official-style sign.",
    ),
    Candidate(
        "no_motor_vehicles",
        "File:Taiwan (ROC) road sign P2 Other Version 1.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Taiwan alternate no motor vehicles official-style sign.",
    ),
    Candidate(
        "no_motor_vehicles",
        "File:Old Taiwan (ROC) road sign P8 No Motor Vehicles.svg",
        "official_style_reference_diagram",
        "medium_obsolete_style",
        "Obsolete Taiwan no motor vehicles sign.",
    ),
    Candidate(
        "no_straight_ahead",
        "File:China road sign 禁 24.svg",
        "official_style_reference_diagram",
        "high_exact_assignment_style",
        "China/GB-style no straight ahead reference matching coursework sign_010.",
    ),
    Candidate(
        "no_straight_ahead",
        "File:Vietnam road sign P136.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Vietnam no straight ahead official-style sign.",
    ),
    Candidate(
        "no_straight_ahead",
        "File:IE road sign RUS-011.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Ireland no straight ahead sign.",
    ),
    Candidate(
        "no_straight_ahead",
        "File:Taiwan road sign Art074.7.png",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Taiwan no straight ahead sign.",
    ),
    Candidate(
        "no_straight_ahead",
        "File:Taiwan road sign p21.svg",
        "official_style_reference_diagram",
        "high_exact_meaning",
        "Taiwan no straight ahead sign.",
    ),
)

FIELDNAMES = [
    "stage_id",
    "candidate_id",
    "semantic_sign_id",
    "source_title",
    "commons_page_url",
    "source_file_url",
    "thumbnail_url",
    "license_short_name",
    "license_url",
    "artist",
    "credit",
    "source_modality",
    "mapping_confidence",
    "local_original_path",
    "local_raster_path",
    "raster_sha256",
    "download_status",
    "review_status",
    "counts_for_candidate_coverage",
    "notes",
]


def project_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def slug(value: str) -> str:
    value = value.replace("File:", "")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")[:140]


def request_json(params: dict[str, str]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, 5):
        if attempt > 1:
            time.sleep(4 * attempt)
        try:
            response = requests.get(
                COMMONS_API_URL,
                params=params,
                timeout=45,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code == 429:
                raise RuntimeError(response.text[:200])
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Commons request failed: {params}") from last_error


def commons_info(title: str) -> dict[str, Any]:
    data = request_json(
        {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime",
            "iiurlwidth": "900",
        }
    )
    page = next(iter(data.get("query", {}).get("pages", {}).values()))
    if "missing" in page:
        raise ValueError(f"Commons title not found: {title}")
    return page


def meta_value(meta: dict[str, dict[str, str]], key: str) -> str:
    value = meta.get(key, {}).get("value", "")
    return re.sub(r"<[^>]+>", "", value).strip()


def download_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.content


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, ""
    except (UnidentifiedImageError, OSError) as exc:
        return False, str(exc)


def build_row(candidate: Candidate, index: int) -> dict[str, str]:
    candidate_id = f"COMG01-{index:03d}"
    class_root = RAW_ROOT / candidate.semantic_sign_id
    title_slug = slug(candidate.title)
    original_path = class_root / "original" / title_slug
    raster_path = class_root / "raster" / f"{Path(title_slug).stem}.png"

    try:
        page = commons_info(candidate.title)
        info = page.get("imageinfo", [{}])[0]
        meta = info.get("extmetadata", {})
        source_url = info.get("url", "")
        thumb_url = info.get("thumburl") or source_url
        page_url = f"https://commons.wikimedia.org/wiki/{candidate.title.replace(' ', '_')}"

        original_path.parent.mkdir(parents=True, exist_ok=True)
        raster_path.parent.mkdir(parents=True, exist_ok=True)
        if source_url:
            original_path.write_bytes(download_bytes(source_url))
        if thumb_url:
            raster_path.write_bytes(download_bytes(thumb_url))

        ok, error = validate_image(raster_path)
        status = "downloaded" if ok else f"downloaded_but_invalid_raster:{error}"
        review_status = "auto_source_label_pending_stage_d_visual_qc" if ok else "invalid_raster"
        counts = "yes_official_source_pending_stage_d_qc" if ok else "no"
        raster_sha = sha256_file(raster_path) if ok else ""
    except Exception as exc:  # noqa: BLE001
        info = {}
        meta = {}
        source_url = ""
        thumb_url = ""
        page_url = f"https://commons.wikimedia.org/wiki/{candidate.title.replace(' ', '_')}"
        status = f"failed:{type(exc).__name__}:{exc}"
        review_status = "failed_download"
        counts = "no"
        raster_sha = ""

    return {
        "stage_id": STAGE_ID,
        "candidate_id": candidate_id,
        "semantic_sign_id": candidate.semantic_sign_id,
        "source_title": candidate.title,
        "commons_page_url": page_url,
        "source_file_url": source_url,
        "thumbnail_url": thumb_url,
        "license_short_name": meta_value(meta, "LicenseShortName"),
        "license_url": meta_value(meta, "LicenseUrl"),
        "artist": meta_value(meta, "Artist"),
        "credit": meta_value(meta, "Credit"),
        "source_modality": candidate.source_modality,
        "mapping_confidence": candidate.mapping_confidence,
        "local_original_path": project_rel(original_path) if original_path.exists() else "",
        "local_raster_path": project_rel(raster_path) if raster_path.exists() else "",
        "raster_sha256": raster_sha,
        "download_status": status,
        "review_status": review_status,
        "counts_for_candidate_coverage": counts,
        "notes": candidate.notes,
    }


def make_contact_sheet(rows: list[dict[str, str]], path: Path) -> None:
    selected = [row for row in rows if row["local_raster_path"]]
    cols = 5
    tile_w, tile_h = 190, 205
    sheet_rows = max(1, (len(selected) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tile_w, sheet_rows * tile_h), (235, 235, 235))
    for idx, row in enumerate(selected):
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        try:
            image = Image.open(PROJECT_ROOT / row["local_raster_path"]).convert("RGB")
            image = ImageOps.contain(image, (150, 150), Image.Resampling.LANCZOS)
            tile.paste(image, ((tile_w - image.width) // 2, 6))
        except OSError:
            pass
        draw = ImageDraw.Draw(tile)
        draw.text((6, 162), row["candidate_id"], fill="black")
        draw.text((6, 178), row["mapping_confidence"][:28], fill="black")
        sheet.paste(tile, ((idx % cols) * tile_w, (idx // cols) * tile_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        for path in (RAW_ROOT, SHEET_ROOT):
            if path.exists():
                import shutil

                shutil.rmtree(path)
        MANIFEST_PATH.unlink(missing_ok=True)
        AUDIT_PATH.unlink(missing_ok=True)

    rows = [build_row(candidate, index) for index, candidate in enumerate(CANDIDATES, start=1)]
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["counts_for_candidate_coverage"].startswith("yes"):
            counts[row["semantic_sign_id"]] = counts.get(row["semantic_sign_id"], 0) + 1
            by_class.setdefault(row["semantic_sign_id"], []).append(row)

    for semantic_id, class_rows in by_class.items():
        make_contact_sheet(class_rows, SHEET_ROOT / f"{semantic_id}.jpg")
    make_contact_sheet(rows, SHEET_ROOT / "_all_commons_official_candidates.jpg")

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(
        json.dumps(
            {
                "stage_id": STAGE_ID,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "Wikimedia Commons per-file pages and API metadata",
                "counts_by_class": counts,
                "manifest_path": project_rel(MANIFEST_PATH),
                "raw_root": project_rel(RAW_ROOT),
                "review_root": project_rel(SHEET_ROOT),
                "note": "Official/reference candidates are useful but remain separate from real-road TT100K crops.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} Commons candidates to {MANIFEST_PATH}")
    print(counts)


if __name__ == "__main__":
    main()
