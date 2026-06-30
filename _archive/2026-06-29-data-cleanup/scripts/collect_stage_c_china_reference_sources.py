from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_china_reference_sources_01"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"

RAW_ROOT = PROJECT_ROOT / "data/raw/online_sources/stage_c_china_reference_01/wikimedia_commons"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_china_reference_sources_01.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_china_reference_sources_01.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_china_reference_sources_01"


@dataclass(frozen=True)
class Candidate:
    semantic_sign_id: str
    title: str
    source_family: str
    source_modality: str
    mapping_confidence: str
    counts_for_realistic_photo_coverage: str
    evidence_label: str
    notes: str


CANDIDATES = [
    Candidate(
        "bicycle_crossing",
        "File:CN road sign \u8b66 13.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Cyclists",
        "Exact China warning-sign reference for cyclists/bicycle crossing style.",
    ),
    Candidate(
        "motor_vehicles_only",
        "File:China road sign \u793a 30.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "medium_possible_style_difference",
        "no_reference_only",
        "Road signs in China gallery: Lane for automobiles",
        "China indicative automobiles-only lane reference. Useful anchor, but it is rectangular lane-style artwork, so keep separate from a circular motor-vehicles-only interpretation.",
    ),
    Candidate(
        "no_left_or_right_turn",
        "File:China road sign \u7981 25.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: No left and right turns",
        "Exact China prohibitory reference for no left/right turns.",
    ),
    Candidate(
        "no_motor_vehicles",
        "File:CN road sign \u7981 6.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: No motor vehicles",
        "Exact China prohibitory reference for no motor vehicles.",
    ),
    Candidate(
        "no_straight_or_left",
        "File:China road sign \u7981 26.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Do not proceed straight and no left turns",
        "Exact China prohibitory reference for no straight/left.",
    ),
    Candidate(
        "no_straight_or_right",
        "File:China road sign \u7981 27.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Do not proceed straight and no right turns",
        "Exact China prohibitory reference for no straight/right.",
    ),
    Candidate(
        "roundabout_mandatory",
        "File:CN road sign \u793a 9.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Roundabout",
        "Exact China blue mandatory/indicative roundabout reference.",
    ),
    Candidate(
        "slow_text",
        "File:China road sign \u8b66 35.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Slow",
        "Exact China warning reference with Chinese slow text.",
    ),
    Candidate(
        "sound_horn",
        "File:CN road sign \u793a 12.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Honking allowed",
        "Exact China blue honking/sound-horn reference; this is the opposite of no-horn.",
    ),
    Candidate(
        "steep_descent",
        "File:CN road sign \u8b66 5-2.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Steep descent",
        "Exact China warning reference for steep descent.",
    ),
    Candidate(
        "stop_for_checking",
        "File:CN road sign \u7981 41.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Security check",
        "Exact China security-check/stop-for-checking prohibitory reference.",
    ),
    Candidate(
        "straight_ahead",
        "File:China road sign \u793a 1.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Proceed straight",
        "Exact China blue mandatory proceed-straight reference.",
    ),
    Candidate(
        "straight_or_right",
        "File:China road sign \u793a 5.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Proceed straight and/or turn right",
        "Exact China blue mandatory straight/right reference.",
    ),
    Candidate(
        "turn_left",
        "File:CN road sign \u793a 2.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Turn left",
        "Exact China blue mandatory turn-left reference.",
    ),
    Candidate(
        "turn_left_or_right",
        "File:China road sign \u793a 6.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Turn left and/or right",
        "Exact China blue mandatory left/right reference.",
    ),
    Candidate(
        "turn_right",
        "File:CN road sign \u793a 3.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Turn right",
        "Exact China blue mandatory turn-right reference.",
    ),
    Candidate(
        "uneven_road",
        "File:CN road sign \u8b66 26.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Bumpy road",
        "Exact China warning reference for bumpy/uneven road.",
    ),
    Candidate(
        "width_restriction",
        "File:China road sign \u7981 34.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Road signs in China gallery: Maximum width",
        "Exact China prohibitory maximum-width reference.",
    ),
    Candidate(
        "residential_area_ahead",
        "File:CN road sign \u8b66 23.svg",
        "wikimedia_commons_china_gb_road_sign_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Wikimedia Commons / China GB-style reference: Residential area",
        "Exact China warning reference for village/residential area ahead.",
    ),
]

UNRESOLVED_TARGETS = [
    {
        "semantic_sign_id": "no_lane_changing",
        "reason": (
            "Reliable China/GB-style searches point mostly to lane markings/solid-line rules, "
            "not a clearly matching standalone road-sign board in the Road signs in China gallery."
        ),
        "next_source": "Coursework original sign sheet, lecturer confirmation, or a public source with an exact no-lane-changing sign board.",
    }
]

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
    "source_family",
    "source_modality",
    "mapping_confidence",
    "counts_for_realistic_photo_coverage",
    "evidence_label",
    "local_original_path",
    "local_raster_path",
    "download_status",
    "review_status",
    "notes",
]


def slug(value: str) -> str:
    value = value.replace("File:", "")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:140]


def request_with_retries(url: str, *, params: dict[str, str] | None = None) -> requests.Response:
    last_error: Exception | None = None
    for attempt, delay in enumerate((0, 8, 24, 60), start=1):
        if delay:
            time.sleep(delay)
        try:
            response = requests.get(
                url,
                params=params,
                timeout=50,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code == 429 and attempt < 4:
                last_error = RuntimeError("HTTP 429 Too Many Requests")
                continue
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001 - manifest needs exact failure.
            last_error = exc
            if attempt == 4:
                raise
    raise RuntimeError("unreachable") from last_error


def commons_info(title: str) -> dict[str, Any]:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "iiurlwidth": "720",
    }
    response = request_with_retries(COMMONS_API_URL, params=params)
    pages = response.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    if not page or "missing" in page:
        raise ValueError(f"No Commons file found for {title}")
    image_info = (page.get("imageinfo") or [{}])[0]
    return {
        "title": page.get("title", title),
        "imageinfo": image_info,
    }


def meta_value(extmetadata: dict[str, dict[str, str]], key: str) -> str:
    value = extmetadata.get(key, {}).get("value", "")
    return re.sub(r"<[^>]+>", "", value).strip()


def download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    response = request_with_retries(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


def validate_raster(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, ""
    except UnidentifiedImageError as exc:
        return False, str(exc)
    except OSError as exc:
        return False, str(exc)


def commons_page_url(title: str) -> str:
    return "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"), safe="/:_()-.")


def build_row(candidate: Candidate, index: int) -> dict[str, str]:
    candidate_id = f"CHREF01-{index:03d}"
    class_root = RAW_ROOT / candidate.semantic_sign_id
    base_slug = slug(candidate.title)
    original_path = class_root / "original" / base_slug
    raster_path = class_root / "raster" / f"{Path(base_slug).stem}.png"

    original_download_note = ""
    try:
        info = commons_info(candidate.title)
        image_info = info["imageinfo"]
        extmetadata = image_info.get("extmetadata", {})
        source_file_url = image_info.get("url", "")
        thumbnail_url = image_info.get("thumburl") or source_file_url
        if not source_file_url or not thumbnail_url:
            raise ValueError("Commons imageinfo did not include a downloadable URL")

        original_ext = Path(source_file_url.split("?")[0]).suffix or Path(base_slug).suffix or ".bin"
        original_path = original_path.with_suffix(original_ext)
        try:
            download_file(source_file_url, original_path)
            time.sleep(1.5)
        except Exception as exc:  # noqa: BLE001 - thumbnail fallback is acceptable for review.
            original_download_note = (
                "Original file download failed; using API thumbnail raster for local review. "
                f"Original error: {type(exc).__name__}: {exc}"
            )
        download_file(thumbnail_url, raster_path)
        is_valid, error = validate_raster(raster_path)
        if not is_valid:
            status = f"invalid_raster: {error}"
            review_status = "downloaded_but_invalid_raster"
        else:
            status = "downloaded_reference_candidate"
            review_status = "pending_stage_d_visual_qc"
    except Exception as exc:  # noqa: BLE001 - manifest should capture source failures.
        image_info = {}
        extmetadata = {}
        source_file_url = ""
        thumbnail_url = ""
        status = f"download_failed: {type(exc).__name__}: {exc}"
        review_status = "download_failed"

    return {
        "stage_id": STAGE_ID,
        "candidate_id": candidate_id,
        "semantic_sign_id": candidate.semantic_sign_id,
        "source_title": candidate.title,
        "commons_page_url": commons_page_url(candidate.title),
        "source_file_url": source_file_url,
        "thumbnail_url": thumbnail_url,
        "license_short_name": meta_value(extmetadata, "LicenseShortName"),
        "license_url": meta_value(extmetadata, "LicenseUrl"),
        "artist": meta_value(extmetadata, "Artist"),
        "credit": meta_value(extmetadata, "Credit"),
        "source_family": candidate.source_family,
        "source_modality": candidate.source_modality,
        "mapping_confidence": candidate.mapping_confidence,
        "counts_for_realistic_photo_coverage": candidate.counts_for_realistic_photo_coverage,
        "evidence_label": candidate.evidence_label,
        "local_original_path": original_path.relative_to(PROJECT_ROOT).as_posix()
        if original_path.exists()
        else "",
        "local_raster_path": raster_path.relative_to(PROJECT_ROOT).as_posix()
        if raster_path.exists()
        else "",
        "download_status": status,
        "review_status": review_status,
        "notes": f"{candidate.notes} {original_download_note}".strip(),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows() -> dict[str, dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle)}


def ordered_rows(rows_by_id: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(1, len(CANDIDATES) + 1):
        candidate_id = f"CHREF01-{index:03d}"
        if candidate_id in rows_by_id:
            rows.append(rows_by_id[candidate_id])
    return rows


def render_sheet(rows: list[dict[str, str]], output_path: Path) -> str:
    columns = 4
    tile_width = 280
    tile_height = 250
    label_height = 68
    sheet = Image.new(
        "RGB",
        (columns * tile_width, max(1, math.ceil(len(rows) / columns)) * tile_height),
        color=(20, 25, 24),
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        image_path = PROJECT_ROOT / row["local_raster_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            thumbnail = ImageOps.contain(
                background.convert("RGB"),
                (tile_width - 18, tile_height - label_height - 18),
            )
        sheet.paste(
            thumbnail,
            (
                x + (tile_width - thumbnail.width) // 2,
                y + 8 + (tile_height - label_height - 18 - thumbnail.height) // 2,
            ),
        )
        draw.rectangle(
            (x, y + tile_height - label_height, x + tile_width, y + tile_height),
            fill=(31, 42, 38),
        )
        draw.text((x + 7, y + tile_height - label_height + 6), row["candidate_id"], fill=(232, 239, 236))
        draw.text((x + 7, y + tile_height - label_height + 24), row["semantic_sign_id"][:35], fill=(232, 239, 236))
        draw.text((x + 7, y + tile_height - label_height + 43), row["mapping_confidence"][:35], fill=(180, 198, 190))
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline=(63, 78, 72), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def render_sheets(rows: list[dict[str, str]]) -> dict[str, str]:
    SHEET_ROOT.mkdir(parents=True, exist_ok=True)
    downloaded = [row for row in rows if row["download_status"] == "downloaded_reference_candidate"]
    outputs: dict[str, str] = {}
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in downloaded:
        by_class.setdefault(row["semantic_sign_id"], []).append(row)
    for semantic_id, class_rows in sorted(by_class.items()):
        outputs[semantic_id] = render_sheet(class_rows, SHEET_ROOT / f"{semantic_id}.jpg")
    if downloaded:
        outputs["_all_china_reference_candidates"] = render_sheet(
            downloaded,
            SHEET_ROOT / "_all_china_reference_candidates.jpg",
        )
    return outputs


def build_report(rows: list[dict[str, str]], contact_sheets: dict[str, str]) -> dict[str, Any]:
    downloaded_rows = [row for row in rows if row["download_status"] == "downloaded_reference_candidate"]
    high_rows = [row for row in downloaded_rows if row["mapping_confidence"] == "high"]
    counts_by_class: dict[str, int] = {}
    for row in downloaded_rows:
        counts_by_class[row["semantic_sign_id"]] = counts_by_class.get(row["semantic_sign_id"], 0) + 1
    failed_rows = [row for row in rows if row["download_status"] != "downloaded_reference_candidate"]
    return {
        "schema_version": "1.0",
        "stage_id": STAGE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_page": "https://en.wikipedia.org/wiki/Road_signs_in_China",
        "source_page_note": (
            "The Road signs in China gallery names the sign concepts used as evidence labels. "
            "Individual file provenance/licences are recorded per Commons file in the manifest."
        ),
        "manifest_path": MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "raw_root": RAW_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "contact_sheets": contact_sheets,
        "downloaded_reference_candidates": len(downloaded_rows),
        "high_confidence_reference_candidates": len(high_rows),
        "download_failed_candidates": len(failed_rows),
        "classes_with_downloaded_references": sorted(counts_by_class),
        "counts_by_class": counts_by_class,
        "unresolved_targets": UNRESOLVED_TARGETS,
        "coverage_policy": (
            "These China/GB-style references are official-style diagrams and are not counted as "
            "realistic road-scene/photo coverage. They are class anchors for the rare assignment signs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect China/GB-style reference signs for rare assignment gaps.")
    parser.add_argument("--classes", nargs="*", default=[], help="Optional semantic_sign_id classes to collect.")
    parser.add_argument("--refresh", action="store_true", help="Re-query candidates even when the manifest already has a successful row.")
    args = parser.parse_args()
    selected = set(args.classes)

    rows_by_id = read_existing_rows()
    for index, candidate in enumerate(CANDIDATES, start=1):
        if selected and candidate.semantic_sign_id not in selected:
            continue
        candidate_id = f"CHREF01-{index:03d}"
        existing = rows_by_id.get(candidate_id)
        if (
            existing
            and existing.get("download_status") == "downloaded_reference_candidate"
            and not args.refresh
        ):
            print(f"{candidate_id} {candidate.semantic_sign_id}: already_downloaded", flush=True)
            continue
        row = build_row(candidate, index)
        rows_by_id[candidate_id] = row
        write_csv(MANIFEST_PATH, ordered_rows(rows_by_id))
        print(f"{candidate_id} {candidate.semantic_sign_id}: {row['download_status']}", flush=True)
        time.sleep(2.0)

    rows = ordered_rows(rows_by_id)
    write_csv(MANIFEST_PATH, rows)
    sheets = render_sheets(rows)
    report = build_report(rows, sheets)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    for name, path in sheets.items():
        print(f"Wrote {name}: {path}")
    print(
        json.dumps(
            {
                "downloaded_reference_candidates": report["downloaded_reference_candidates"],
                "high_confidence_reference_candidates": report["high_confidence_reference_candidates"],
                "download_failed_candidates": report["download_failed_candidates"],
                "classes_with_downloaded_references": report["classes_with_downloaded_references"],
                "unresolved_targets": UNRESOLVED_TARGETS,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
