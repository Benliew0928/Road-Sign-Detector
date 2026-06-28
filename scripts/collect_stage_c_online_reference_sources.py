from __future__ import annotations

import argparse
import csv
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
STAGE_ID = "stage_c_online_reference_sources_01"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"

RAW_ROOT = PROJECT_ROOT / "data/raw/online_sources/stage_c_reference_01/wikimedia_commons"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_online_reference_sources_01.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_online_reference_sources_01.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_online_reference_sources_01"
REALISTIC_GAP_PATH = PROJECT_ROOT / "outputs/audit/post_stage_c_realistic_gap_report.csv"


@dataclass(frozen=True)
class Candidate:
    semantic_sign_id: str
    title: str
    source_family: str
    source_modality: str
    mapping_confidence: str
    counts_for_realistic_photo_coverage: str
    notes: str


CANDIDATES = [
    Candidate(
        "straight_ahead",
        "File:Malaysia road sign RM1c.svg",
        "wikimedia_commons_malaysia_mandatory_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Exact Malaysian mandatory go-straight sign reference.",
    ),
    Candidate(
        "straight_or_right",
        "File:Malaysia road sign RM1e.svg",
        "wikimedia_commons_malaysia_mandatory_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Exact Malaysian mandatory straight-or-right sign reference.",
    ),
    Candidate(
        "turn_left",
        "File:Malaysia road sign RM1a.svg",
        "wikimedia_commons_malaysia_mandatory_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Exact Malaysian mandatory turn-left sign reference.",
    ),
    Candidate(
        "turn_right",
        "File:Malaysia road sign RM1b.svg",
        "wikimedia_commons_malaysia_mandatory_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Exact Malaysian mandatory turn-right sign reference.",
    ),
    Candidate(
        "motor_vehicles_only",
        "File:Malaysia road sign RM3a.svg",
        "wikimedia_commons_malaysia_mandatory_svg",
        "official_style_reference_diagram",
        "low_possible_mismatch",
        "no_reference_only",
        "Malaysian compulsory vehicles-track reference. Visual review shows multiple vehicle types and it may not match assignment motor-vehicles-only artwork; do not treat as solved without lecturer/manual confirmation.",
    ),
    Candidate(
        "pass_either_side",
        "File:Malaysia road sign RM4c.svg",
        "wikimedia_commons_malaysia_mandatory_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Exact Malaysian mandatory pass-either-side / keep-left-and-right reference.",
    ),
    Candidate(
        "school_zone",
        "File:School zone speed limit.png",
        "huggingface_wikipedia_malaysian_road_sign_images",
        "malaysia_reference_diagram",
        "high",
        "no_reference_only",
        "Malaysian school-zone speed-limit reference from Wikimedia-hosted image.",
    ),
    Candidate(
        "school_zone",
        "File:Sekolah Had Laju 30 kmj.png",
        "wikimedia_commons_malaysia_warning_diagram",
        "malaysia_reference_diagram",
        "high",
        "no_reference_only",
        "Malay school-zone speed-limit diagram; useful for school-zone visual/text reference.",
    ),
    Candidate(
        "steep_descent",
        "File:Malaysia road sign WD8.svg",
        "wikimedia_commons_malaysia_warning_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Exact Malaysian downhill-slope / steep-descent warning reference.",
    ),
    Candidate(
        "steep_descent",
        "File:Cerun menurun.png",
        "wikimedia_commons_malaysia_warning_diagram",
        "malaysia_reference_diagram",
        "high",
        "no_reference_only",
        "Malay steep-descent diagram from Malaysian warning-sign Commons category.",
    ),
    Candidate(
        "uneven_road",
        "File:Malaysia road sign WD1.svg",
        "wikimedia_commons_malaysia_warning_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Exact Malaysian uneven-road warning reference.",
    ),
    Candidate(
        "uneven_road",
        "File:Jalan tidak rata.png",
        "wikimedia_commons_malaysia_warning_diagram",
        "malaysia_reference_diagram",
        "high",
        "no_reference_only",
        "Malay uneven-road diagram from Malaysian warning-sign Commons category.",
    ),
    Candidate(
        "width_restriction",
        "File:Malaysia road sign RP14.svg",
        "wikimedia_commons_malaysia_regulatory_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Exact Malaysian width-limit regulatory reference.",
    ),
    Candidate(
        "no_heavy_vehicle",
        "File:Malaysia road sign RP8c.svg",
        "wikimedia_commons_malaysia_regulatory_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Exact Malaysian no-truck/no-heavy-vehicle regulatory reference.",
    ),
    Candidate(
        "side_road_right",
        "File:Malaysia road sign WD27b.svg",
        "wikimedia_commons_malaysia_warning_svg",
        "official_style_reference_diagram",
        "high",
        "no_reference_only",
        "Exact Malaysian crossroad/side-road-on-right warning reference.",
    ),
    Candidate(
        "side_road_right",
        "File:Simpang kanan.png",
        "wikimedia_commons_malaysia_warning_diagram",
        "malaysia_reference_diagram",
        "high",
        "no_reference_only",
        "Malay side-road/right-junction diagram from Malaysian warning-sign Commons category.",
    ),
]

UNRESOLVED_TARGETS = [
    {
        "semantic_sign_id": "bicycle_crossing",
        "reason": "Current online references found bicycle-only/prohibition signs, not a bicycle crossing warning.",
        "next_source": "JKR ATJ 2A/85 artwork extraction or a public real-photo dataset with exact bicycle-crossing label.",
    },
    {
        "semantic_sign_id": "no_lane_changing",
        "reason": "Search found lane-closure/keep-lane material, not the exact no-lane-changing sign.",
        "next_source": "JKR ATJ 2A/85 artwork extraction or exact public photo.",
    },
    {
        "semantic_sign_id": "no_left_or_right_turn",
        "reason": "Search found U-turn and road markings, not an exact no-left-or-right-turn sign.",
        "next_source": "JKR ATJ 2A/85 artwork extraction.",
    },
    {
        "semantic_sign_id": "no_motor_vehicles",
        "reason": "No exact Malaysian online file found yet; no-motorcycle/no-truck signs are different classes.",
        "next_source": "JKR ATJ 2A/85 artwork extraction or exact public photo.",
    },
    {
        "semantic_sign_id": "no_straight_or_left",
        "reason": "No exact Malaysian online file found yet.",
        "next_source": "JKR ATJ 2A/85 artwork extraction.",
    },
    {
        "semantic_sign_id": "no_straight_or_right",
        "reason": "No exact Malaysian online file found yet.",
        "next_source": "JKR ATJ 2A/85 artwork extraction.",
    },
    {
        "semantic_sign_id": "roundabout_mandatory",
        "reason": "Commons has yellow roundabout warning signs, not an exact blue mandatory roundabout sign.",
        "next_source": "JKR ATJ 2A/85 artwork extraction or exact public photo.",
    },
    {
        "semantic_sign_id": "slow_text",
        "reason": "Search found road marking text, not a confirmed road sign panel matching the assignment class.",
        "next_source": "JKR/coursework artwork source or exact public photo.",
    },
    {
        "semantic_sign_id": "sound_horn",
        "reason": "Search found no-horn prohibition, which is the opposite of mandatory sound-horn.",
        "next_source": "JKR ATJ 2A/85 artwork extraction.",
    },
    {
        "semantic_sign_id": "stop_for_checking",
        "reason": "Search did not find an exact Malaysian stop-for-checking sign image.",
        "next_source": "JKR ATJ 2A/85 artwork extraction or exact public photo.",
    },
    {
        "semantic_sign_id": "turn_left_or_right",
        "reason": "RM1f/RM1g are not confirmed left-or-right combined signs; not mapped to avoid label noise.",
        "next_source": "JKR ATJ 2A/85 artwork extraction.",
    },
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
    for attempt, delay in enumerate((0, 6, 20, 45), start=1):
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
        except Exception as exc:  # noqa: BLE001 - keep exact failure for manifest.
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


def build_row(candidate: Candidate, index: int) -> dict[str, str]:
    candidate_id = f"ONREF01-{index:03d}"
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
                f"Original file download failed; using API thumbnail raster for local review. "
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
        "commons_page_url": f"https://commons.wikimedia.org/wiki/{candidate.title.replace(' ', '_')}",
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
        candidate_id = f"ONREF01-{index:03d}"
        if candidate_id in rows_by_id:
            rows.append(rows_by_id[candidate_id])
    return rows


def render_sheet(rows: list[dict[str, str]], output_path: Path) -> str:
    columns = 4
    tile_width = 260
    tile_height = 245
    label_height = 62
    sheet = Image.new(
        "RGB",
        (columns * tile_width, ((len(rows) + columns - 1) // columns) * tile_height),
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
        draw.text((x + 7, y + tile_height - label_height + 24), row["semantic_sign_id"][:33], fill=(232, 239, 236))
        draw.text((x + 7, y + tile_height - label_height + 43), row["mapping_confidence"], fill=(180, 198, 190))
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline=(63, 78, 72), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def render_sheets(rows: list[dict[str, str]]) -> dict[str, str]:
    SHEET_ROOT.mkdir(parents=True, exist_ok=True)
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["download_status"] != "downloaded_reference_candidate":
            continue
        by_class.setdefault(row["semantic_sign_id"], []).append(row)

    outputs: dict[str, str] = {}
    for semantic_id, class_rows in sorted(by_class.items()):
        outputs[semantic_id] = render_sheet(class_rows, SHEET_ROOT / f"{semantic_id}.jpg")
    if by_class:
        all_rows = [row for class_rows in by_class.values() for row in class_rows]
        outputs["_all_reference_candidates"] = render_sheet(all_rows, SHEET_ROOT / "_all_reference_candidates.jpg")
    return outputs


def read_gap_rows() -> dict[str, dict[str, str]]:
    if not REALISTIC_GAP_PATH.exists():
        return {}
    with REALISTIC_GAP_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["semantic_sign_id"]: row for row in csv.DictReader(handle)}


def build_report(rows: list[dict[str, str]], contact_sheets: dict[str, str]) -> dict[str, Any]:
    downloaded_rows = [row for row in rows if row["download_status"] == "downloaded_reference_candidate"]
    exact_or_high_rows = [
        row
        for row in downloaded_rows
        if row["mapping_confidence"] in {"high", "exact"}
    ]
    counts_by_class: dict[str, int] = {}
    for row in downloaded_rows:
        counts_by_class[row["semantic_sign_id"]] = counts_by_class.get(row["semantic_sign_id"], 0) + 1
    exact_or_high_counts_by_class: dict[str, int] = {}
    for row in exact_or_high_rows:
        exact_or_high_counts_by_class[row["semantic_sign_id"]] = (
            exact_or_high_counts_by_class.get(row["semantic_sign_id"], 0) + 1
        )

    gap_rows = read_gap_rows()
    covered_classes = sorted(exact_or_high_counts_by_class)
    unresolved_with_current_gap = []
    for item in UNRESOLVED_TARGETS:
        gap_row = gap_rows.get(item["semantic_sign_id"], {})
        unresolved_with_current_gap.append(
            {
                **item,
                "realistic_candidate_total": gap_row.get("realistic_candidate_total", ""),
                "gap_to_minimum_realistic": gap_row.get("gap_to_minimum_realistic", ""),
            }
        )

    return {
        "schema_version": "1.0",
        "stage_id": STAGE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Curated Wikimedia Commons and HuggingFace/Wikipedia Malaysian road-sign references",
        "downloaded_reference_candidates": len(downloaded_rows),
        "failed_candidates": sum(1 for row in rows if row["review_status"] == "download_failed"),
        "classes_with_exact_or_high_confidence_reference": covered_classes,
        "reference_counts_by_class": counts_by_class,
        "exact_or_high_reference_counts_by_class": exact_or_high_counts_by_class,
        "unresolved_targets": unresolved_with_current_gap,
        "manifest": MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "raw_root": RAW_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "contact_sheets": contact_sheets,
        "status": "online_reference_candidates_collected_pending_visual_qc",
        "important_note": (
            "These files are online reference candidates, not real road-photo coverage. "
            "They can support class definitions and controlled training experiments, but "
            "they do not close the realistic-photo coverage gap by themselves."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect curated online Malaysian road-sign references.")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=[],
        help="Optional semantic_sign_id values to retry. Defaults to all curated candidates.",
    )
    args = parser.parse_args()

    selected_classes = set(args.classes)
    rows_by_id = read_existing_rows()
    for index, candidate in enumerate(CANDIDATES, start=1):
        if selected_classes and candidate.semantic_sign_id not in selected_classes:
            continue
        candidate_id = f"ONREF01-{index:03d}"
        rows_by_id[candidate_id] = build_row(candidate, index)
        write_csv(MANIFEST_PATH, ordered_rows(rows_by_id))
        time.sleep(2.5)
    rows = ordered_rows(rows_by_id)
    contact_sheets = render_sheets(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(build_report(rows, contact_sheets), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    for key, value in contact_sheets.items():
        print(f"Wrote {key}: {value}")


if __name__ == "__main__":
    main()
