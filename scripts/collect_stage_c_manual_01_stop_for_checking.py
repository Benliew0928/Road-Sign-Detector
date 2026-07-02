from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_manual_01_visual_commons"
TARGET_ID = "stop_for_checking"
DISPLAY_NAME = "Stop for checking"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"

MANUAL_ROOT = PROJECT_ROOT / "data/raw/manual_collection/stage_c_manual_01/stop_for_checking"
ORIGINAL_ROOT = MANUAL_ROOT / "original"
CROP_ROOT = MANUAL_ROOT / "crops"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_manual_01_candidates.csv"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
CONTACT_SHEET_PATH = (
    PROJECT_ROOT
    / "outputs/review/00_CURRENT_REVIEW/stage_c_manual_stop_for_checking_contact_sheet.jpg"
)
ASSIGNMENT_REFERENCE_PATH = PROJECT_ROOT / "data/official/assignment_images/Red Signs/057_0003_j.png"


@dataclass(frozen=True)
class CandidateSpec:
    title: str
    local_slug: str
    source_modality: str
    notes: str


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        "File:CN road sign \u7981 41.svg",
        "CN_road_sign_41",
        "official_style_reference_diagram",
        "China/GB official inspection sign: red circular ring, black horizontal bar, and inspection text.",
    ),
    CandidateSpec(
        "File:\u505c\u8eca\u6aa2\u67e5\u6a19\u8a8c\u90753.png",
        "Taiwan_stop_for_checking_Zun3",
        "official_style_reference_diagram",
        "Taiwan official stop-for-checking sign with the same red-ring, black-bar, inspection-text layout.",
    ),
)

FIELDNAMES = [
    "stage_id",
    "candidate_id",
    "semantic_sign_id",
    "display_name",
    "source_title",
    "commons_page_url",
    "source_file_url",
    "download_url",
    "license_short_name",
    "license_url",
    "artist",
    "source_modality",
    "mapping_evidence",
    "assignment_reference_path",
    "local_original_path",
    "local_crop_path",
    "source_sha256",
    "crop_sha256",
    "source_width",
    "source_height",
    "crop_width",
    "crop_height",
    "crop_bbox_xyxy",
    "crop_method",
    "review_status",
    "counts_for_candidate_coverage",
    "notes",
]


def project_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def safe_reset(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    resolved = path.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(f"Refusing to reset outside project root: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def commons_image_info(title: str, width: int) -> dict[str, Any]:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": str(width),
    }
    response: requests.Response | None = None
    for attempt in range(1, 9):
        response = requests.get(
            COMMONS_API_URL,
            params=params,
            timeout=45,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code != 429:
            response.raise_for_status()
            break
        time.sleep(min(45.0, 3.0 * attempt))
    if response is None:
        raise RuntimeError(f"No Commons response for {title}")
    response.raise_for_status()
    data = response.json()
    page = next(iter(data.get("query", {}).get("pages", {}).values()), None)
    if not page or "missing" in page or "imageinfo" not in page:
        raise RuntimeError(f"Commons file was not found: {title}")
    return page


def download_commons_raster(title: str) -> tuple[dict[str, Any], dict[str, Any], str, bytes]:
    last_error = ""
    for width in (1000, 800, 600, 480):
        page = commons_image_info(title, width)
        image_info = page["imageinfo"][0]
        urls = [image_info.get("thumburl"), image_info.get("url")]
        for url in [candidate for candidate in urls if candidate]:
            if image_info.get("mime") == "image/svg+xml" and url == image_info.get("url"):
                continue
            try:
                response: requests.Response | None = None
                for attempt in range(1, 7):
                    response = requests.get(
                        url,
                        timeout=60,
                        headers={"User-Agent": USER_AGENT},
                    )
                    if response.status_code != 429:
                        break
                    last_error = "HTTP 429 from upload.wikimedia.org"
                    time.sleep(min(30.0, 2.0 * attempt))
                if response is None or response.status_code == 429:
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    last_error = f"Unexpected content type {content_type}"
                    continue
                Image.open(BytesIO(response.content)).verify()
                return page, image_info, url, response.content
            except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
                last_error = str(exc)
                time.sleep(0.4)
    raise RuntimeError(f"Could not download raster for {title}: {last_error}")


def image_from_bytes(content: bytes) -> Image.Image:
    image = Image.open(BytesIO(content))
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def save_image_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def save_crop(image: Image.Image, path: Path) -> tuple[str, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95, optimize=True)
    content = buffer.getvalue()
    path.write_bytes(content)
    return sha256_bytes(content), image.width, image.height


def full_sign_crop(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], str]:
    bbox = (0, 0, image.width, image.height)
    return image.copy(), bbox, "full_sign_diagram_keep_text"


def materialize_candidates() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, spec in enumerate(CANDIDATES, start=1):
        page, image_info, download_url, content = download_commons_raster(spec.title)
        image = image_from_bytes(content)
        crop, bbox, crop_method = full_sign_crop(image)
        candidate_id = f"MAN01-SFC-{index:04d}"
        extension = (
            ".png"
            if "png" in download_url.lower() or image_info.get("mime") == "image/svg+xml"
            else ".jpg"
        )
        original_path = ORIGINAL_ROOT / f"{candidate_id}_{spec.local_slug}{extension}"
        crop_path = CROP_ROOT / f"{candidate_id}_{spec.local_slug}.jpg"
        save_image_bytes(content, original_path)
        crop_sha, crop_width, crop_height = save_crop(crop, crop_path)

        metadata = image_info.get("extmetadata", {})
        rows.append(
            {
                "stage_id": STAGE_ID,
                "candidate_id": candidate_id,
                "semantic_sign_id": TARGET_ID,
                "display_name": DISPLAY_NAME,
                "source_title": page["title"],
                "commons_page_url": image_info.get("descriptionurl", ""),
                "source_file_url": image_info.get("url", ""),
                "download_url": download_url,
                "license_short_name": strip_html(
                    metadata.get("LicenseShortName", {}).get("value", "")
                ),
                "license_url": strip_html(metadata.get("LicenseUrl", {}).get("value", "")),
                "artist": strip_html(metadata.get("Artist", {}).get("value", "")),
                "source_modality": spec.source_modality,
                "mapping_evidence": (
                    "Visual match against assignment reference sign_057: red circular "
                    "inspection sign, black horizontal bar, and Chinese inspection text."
                ),
                "assignment_reference_path": project_rel(ASSIGNMENT_REFERENCE_PATH),
                "local_original_path": project_rel(original_path),
                "local_crop_path": project_rel(crop_path),
                "source_sha256": sha256_bytes(content),
                "crop_sha256": crop_sha,
                "source_width": str(image.width),
                "source_height": str(image.height),
                "crop_width": str(crop_width),
                "crop_height": str(crop_height),
                "crop_bbox_xyxy": ",".join(str(value) for value in bbox),
                "crop_method": crop_method,
                "review_status": "visual_match_pending_stage_d_qc",
                "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
                "notes": spec.notes,
            }
        )
        time.sleep(1.0)
    return rows


def write_manifest(rows: list[dict[str, str]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if not (
                    row.get("stage_id") == STAGE_ID
                    and row.get("semantic_sign_id") == TARGET_ID
                ):
                    existing_rows.append(row)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing_rows + rows)


def make_contact_sheet(rows: list[dict[str, str]]) -> None:
    cols = 2
    tile_width, tile_height = 320, 300
    sheet = Image.new("RGB", (cols * tile_width, tile_height), (236, 236, 236))
    for index, row in enumerate(rows):
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        try:
            image = Image.open(PROJECT_ROOT / row["local_crop_path"]).convert("RGB")
            image = ImageOps.contain(image, (250, 220), Image.Resampling.LANCZOS)
            tile.paste(image, ((tile_width - image.width) // 2, 8))
        except OSError:
            pass
        draw = ImageDraw.Draw(tile)
        draw.text((10, 236), row["candidate_id"], fill="black")
        draw.text((10, 253), row["source_modality"][:45], fill=(60, 60, 60))
        draw.text((10, 270), row["source_title"].replace("File:", "")[:45], fill=(60, 60, 60))
        sheet.paste(tile, (index * tile_width, 0))
    CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(CONTACT_SHEET_PATH, quality=92)


def update_tracker(accepted_count: int) -> dict[str, str]:
    with TRACKER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        tracker_rows = list(reader)

    updated_row: dict[str, str] | None = None
    for row in tracker_rows:
        if row["semantic_sign_id"] != TARGET_ID:
            continue
        current_total = int(row["realistic_candidate_total"])
        minimum = int(row["minimum_clean_crops"])
        needed = max(0, minimum - current_total)
        credited_count = min(accepted_count, needed)
        new_total = current_total + credited_count
        row["realistic_candidate_total"] = str(new_total)
        row["gap_to_minimum"] = str(max(0, minimum - new_total))
        row["cleaning_status"] = "stage_d_qc_needed"
        row["collection_status"] = (
            "meets_minimum_pending_qc" if new_total >= minimum else "still_below_minimum"
        )
        row["next_action"] = (
            "Review the Stage C manual stop_for_checking contact sheet, then run Stage D QC "
            "before Stage E split freeze."
        )
        updated_row = dict(row)
        break

    if updated_row is None:
        raise RuntimeError(f"{TARGET_ID} was not found in {TRACKER_PATH}")

    with TRACKER_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tracker_rows)
    return updated_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-tracker", action="store_true")
    args = parser.parse_args()

    if args.reset:
        safe_reset(MANUAL_ROOT)

    rows = materialize_candidates()
    if len(rows) != len(CANDIDATES):
        raise RuntimeError(f"Expected {len(CANDIDATES)} rows, got {len(rows)}")
    write_manifest(rows)
    make_contact_sheet(rows)
    tracker_row = None if args.skip_tracker else update_tracker(len(rows))

    print(f"Wrote {len(rows)} {TARGET_ID} candidates")
    print(f"Manifest: {project_rel(MANIFEST_PATH)}")
    print(f"Contact sheet: {project_rel(CONTACT_SHEET_PATH)}")
    if tracker_row:
        print(
            "Tracker: "
            f"{tracker_row['realistic_candidate_total']}/{tracker_row['minimum_clean_crops']} "
            f"(gap {tracker_row['gap_to_minimum']})"
        )
    print(f"Generated at: {datetime.now(UTC).isoformat()}")


if __name__ == "__main__":
    main()
