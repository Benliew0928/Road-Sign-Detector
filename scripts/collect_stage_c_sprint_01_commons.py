from __future__ import annotations

import csv
import json
import re
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_sprint_01_commons_near_minimum_topup"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"

RAW_ROOT = PROJECT_ROOT / "data/raw/local_collection/stage_c_sprint_01/wikimedia_commons"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_sprint_01_commons_candidates.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_sprint_01_commons_candidates.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_sprint_01_commons_candidates"
POST_ROBOFLOW_GAP_PATH = PROJECT_ROOT / "outputs/audit/post_roboflow_data_gap_report.csv"


@dataclass(frozen=True)
class Candidate:
    semantic_sign_id: str
    title: str
    candidate_role: str
    geographic_relevance: str
    notes: str
    curated_review_status: str = "pending_stage_d_visual_qc"


CANDIDATES = [
    Candidate(
        "no_overtaking",
        "File:Malaysia Traffic-signs Regulatory-sign-09.jpg",
        "malaysia_reference",
        "Malaysia",
        "Malaysia-specific no-overtaking reference graphic.",
    ),
    Candidate(
        "no_overtaking",
        "File:Malaysia road sign RP17.svg",
        "malaysia_reference",
        "Malaysia",
        "Malaysia-specific no-overtaking SVG reference.",
    ),
    Candidate(
        "no_overtaking",
        "File:Dilarang memotong.jpg",
        "malaysia_reference",
        "Malaysia",
        "Downloaded for provenance, but visual QC shows this is heavy-vehicle no-overtaking, not generic no-overtaking.",
        "rejected_codex_visual_qc_label_mismatch",
    ),
    Candidate(
        "no_overtaking",
        "File:No overtaking.jpg",
        "supplemental_reference",
        "International",
        "Supplemental no-overtaking sign; needs Stage D approval before final training.",
    ),
    Candidate(
        "no_overtaking",
        "File:No Overtaking.png",
        "supplemental_reference",
        "International",
        "Supplemental no-overtaking sign; needs Stage D approval before final training.",
    ),
    Candidate(
        "keep_right",
        "File:Malaysia road sign RM4b.svg",
        "malaysia_reference",
        "Malaysia",
        "Malaysia-specific keep-right reference SVG.",
    ),
    Candidate(
        "keep_right",
        "File:B23 (Jalan Sungai Tua), Gombak 20250125 120556.jpg",
        "malaysia_real_photo",
        "Malaysia",
        "Real Malaysian road-scene photo; requires bounding-box annotation before detector use.",
    ),
    Candidate(
        "keep_right",
        "File:B23 (Jalan Sungai Tua), Gombak 20250125 120606.jpg",
        "malaysia_real_photo",
        "Malaysia",
        "Same-location real photo; keep in the same split group during dataset freeze.",
    ),
    Candidate(
        "keep_right",
        "File:Jalan Marang-Wakaf Tapai (Terengganu State Highway T2), Kampung Pengkalan Berangan 20240228 110630.jpg",
        "malaysia_real_photo",
        "Malaysia",
        "Real Malaysian road-scene photo; requires bounding-box annotation before detector use.",
    ),
    Candidate(
        "no_heavy_vehicle",
        "File:Lori dilarang.png",
        "malaysia_reference",
        "Malaysia",
        "Malay no-lorry/no-heavy-vehicle reference graphic.",
    ),
    Candidate(
        "no_heavy_vehicle",
        "File:No heavy vehicles.jpg",
        "malaysia_reference",
        "Malaysia",
        "No-heavy-vehicles reference sign.",
    ),
    Candidate(
        "no_heavy_vehicle",
        "File:Brunei road sign - No Lorries.svg",
        "regional_reference",
        "Regional non-Malaysia",
        "Regional no-lorries symbol; supplemental only unless Stage D approves.",
    ),
    Candidate(
        "no_heavy_vehicle",
        "File:Singapore road sign - Prohibitory - No lorries.svg",
        "regional_reference",
        "Regional non-Malaysia",
        "Regional no-lorries symbol; supplemental only unless Stage D approves.",
    ),
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
    "geographic_relevance",
    "candidate_role",
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
    return value[:120]


def request_with_retries(url: str, *, params: dict[str, str] | None = None) -> requests.Response:
    last_error: Exception | None = None
    for attempt, delay in enumerate((0, 4, 8), start=1):
        if delay:
            time.sleep(delay)
        try:
            response = requests.get(
                url,
                params=params,
                timeout=40,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code == 429 and attempt < 3:
                continue
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001 - preserve exact failure in manifest.
            last_error = exc
            if attempt == 3:
                raise
    raise RuntimeError("unreachable") from last_error


def commons_info(title: str) -> dict[str, Any]:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "iiurlwidth": "640",
    }
    response = request_with_retries(COMMONS_API_URL, params=params)
    pages = response.json().get("query", {}).get("pages", {})
    if not pages:
        raise ValueError(f"No Commons page returned for {title}")
    page = next(iter(pages.values()))
    image_info = (page.get("imageinfo") or [{}])[0]
    return {
        "pageid": page.get("pageid", ""),
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
    candidate_id = f"S01-{index:03d}"
    base_slug = slug(candidate.title)
    class_root = RAW_ROOT / candidate.semantic_sign_id
    original_path = class_root / "original" / base_slug
    raster_path = class_root / "raster" / f"{Path(base_slug).stem}.png"

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
        download_file(source_file_url, original_path)
        time.sleep(0.8)
        download_file(thumbnail_url, raster_path)
        is_valid, error = validate_raster(raster_path)
        if is_valid and candidate.curated_review_status.startswith("rejected_"):
            status = "downloaded_excluded"
            review_status = candidate.curated_review_status
        elif is_valid:
            status = "downloaded_candidate"
            review_status = candidate.curated_review_status
        else:
            status = f"invalid_raster: {error}"
            review_status = "rejected_download_error"
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
        "geographic_relevance": candidate.geographic_relevance,
        "candidate_role": candidate.candidate_role,
        "local_original_path": original_path.relative_to(PROJECT_ROOT).as_posix()
        if original_path.exists()
        else "",
        "local_raster_path": raster_path.relative_to(PROJECT_ROOT).as_posix()
        if raster_path.exists()
        else "",
        "download_status": status,
        "review_status": review_status,
        "notes": candidate.notes,
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
        candidate_id = f"S01-{index:03d}"
        if candidate_id in rows_by_id:
            rows.append(rows_by_id[candidate_id])
    return rows


def render_sheets(rows: list[dict[str, str]]) -> dict[str, str]:
    SHEET_ROOT.mkdir(parents=True, exist_ok=True)
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if not row["download_status"].startswith("downloaded_"):
            continue
        by_class.setdefault(row["semantic_sign_id"], []).append(row)

    outputs: dict[str, str] = {}
    for semantic_id, class_rows in sorted(by_class.items()):
        outputs[semantic_id] = render_sheet(class_rows, SHEET_ROOT / f"{semantic_id}.jpg")
    if by_class:
        all_rows = [row for class_rows in by_class.values() for row in class_rows]
        outputs["_all_candidates"] = render_sheet(all_rows, SHEET_ROOT / "_all_candidates.jpg")
    return outputs


def render_sheet(rows: list[dict[str, str]], output_path: Path) -> str:
    columns = 4
    tile_width = 260
    tile_height = 245
    label_height = 58
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
                (tile_width - 16, tile_height - label_height - 16),
            )
        sheet.paste(
            thumbnail,
            (
                x + (tile_width - thumbnail.width) // 2,
                y + 8 + (tile_height - label_height - 16 - thumbnail.height) // 2,
            ),
        )
        draw.rectangle(
            (x, y + tile_height - label_height, x + tile_width, y + tile_height),
            fill=(31, 42, 38),
        )
        status_prefix = "REJECT " if row["review_status"].startswith("rejected_") else ""
        label = f"{status_prefix}{row['candidate_id']} {row['semantic_sign_id']}"
        role = row["candidate_role"].replace("_", " ")
        draw.text((x + 7, y + tile_height - label_height + 6), label, fill=(232, 239, 236))
        draw.text((x + 7, y + tile_height - label_height + 25), role[:35], fill=(180, 198, 190))
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline=(63, 78, 72), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def read_gap_rows() -> dict[str, dict[str, str]]:
    with POST_ROBOFLOW_GAP_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["semantic_sign_id"]: row for row in csv.DictReader(handle)}


def build_report(rows: list[dict[str, str]], contact_sheets: dict[str, str]) -> dict[str, Any]:
    gap_rows = read_gap_rows()
    downloaded_by_class: dict[str, int] = {}
    malaysia_by_class: dict[str, int] = {}
    for row in rows:
        if row["download_status"] != "downloaded_candidate":
            continue
        semantic_id = row["semantic_sign_id"]
        downloaded_by_class[semantic_id] = downloaded_by_class.get(semantic_id, 0) + 1
        if row["geographic_relevance"] == "Malaysia":
            malaysia_by_class[semantic_id] = malaysia_by_class.get(semantic_id, 0) + 1

    class_impacts = []
    for semantic_id in sorted(downloaded_by_class):
        gap = gap_rows.get(semantic_id, {})
        before = int(gap.get("candidate_total_crops", 0))
        minimum = int(gap.get("minimum_clean_crops", 0))
        sprint_candidates = downloaded_by_class[semantic_id]
        class_impacts.append(
            {
                "semantic_sign_id": semantic_id,
                "candidate_total_before_sprint": before,
                "downloaded_sprint_candidates": sprint_candidates,
                "downloaded_malaysia_candidates": malaysia_by_class.get(semantic_id, 0),
                "candidate_total_with_sprint": before + sprint_candidates,
                "minimum_clean_crops": minimum,
                "gap_to_minimum_with_sprint_candidates": max(0, minimum - before - sprint_candidates),
                "stage_d_required": True,
            }
        )

    return {
        "schema_version": "1.0",
        "stage_id": STAGE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Wikimedia Commons curated candidate download",
        "total_curated_candidates": len(rows),
        "downloaded_candidates": sum(1 for row in rows if row["download_status"] == "downloaded_candidate"),
        "downloaded_excluded": sum(1 for row in rows if row["download_status"] == "downloaded_excluded"),
        "failed_candidates": sum(1 for row in rows if not row["download_status"].startswith("downloaded_")),
        "candidate_counts_by_class": downloaded_by_class,
        "malaysia_candidate_counts_by_class": malaysia_by_class,
        "class_impacts": class_impacts,
        "manifest": MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "raw_root": RAW_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "contact_sheets": contact_sheets,
        "status": "candidate_collection_complete_pending_stage_d_qc",
        "important_note": (
            "These images are newly collected Stage C candidates only. They are not in "
            "the final training split until Stage D QC, annotation, and Stage E freeze."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect curated Stage C Commons candidates.")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=[],
        help="Optional semantic_sign_id values to retry. Defaults to all candidates.",
    )
    args = parser.parse_args()

    selected_classes = set(args.classes)
    rows_by_id = read_existing_rows()
    for index, candidate in enumerate(CANDIDATES, start=1):
        if selected_classes and candidate.semantic_sign_id not in selected_classes:
            continue
        candidate_id = f"S01-{index:03d}"
        rows_by_id[candidate_id] = build_row(candidate, index)
        write_csv(MANIFEST_PATH, ordered_rows(rows_by_id))
        time.sleep(0.8)
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
