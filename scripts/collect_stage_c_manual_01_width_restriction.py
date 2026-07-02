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

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_manual_01_visual_commons"
TARGET_ID = "width_restriction"
DISPLAY_NAME = "Width restriction"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"

MANUAL_ROOT = PROJECT_ROOT / "data/raw/manual_collection/stage_c_manual_01/width_restriction"
ORIGINAL_ROOT = MANUAL_ROOT / "original"
CROP_ROOT = MANUAL_ROOT / "crops"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_manual_01_candidates.csv"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
CONTACT_SHEET_PATH = (
    PROJECT_ROOT
    / "outputs/review/00_CURRENT_REVIEW/stage_c_manual_width_restriction_contact_sheet.jpg"
)
REFERENCE_PATH = (
    PROJECT_ROOT
    / "outputs/review/90_ARCHIVE_OLD_REVIEW_FILES_20260630/02_p5_emtd_dataset_qc/"
    "p5_class_contact_sheets/width_restriction.jpg"
)


@dataclass(frozen=True)
class CandidateSpec:
    title: str
    source_modality: str
    notes: str


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        "File:China road sign \u7981 34.svg",
        "official_style_reference_diagram",
        "China/GB-style maximum-width sign with metric width value.",
    ),
    CandidateSpec(
        "File:Korean Traffic sign (Maximum Width Limit).svg",
        "official_style_reference_diagram",
        "Korean official-style maximum-width sign with side arrows and metric value.",
    ),
    CandidateSpec(
        "File:Japan road sign 322.svg",
        "official_style_reference_diagram",
        "Japan official-style maximum-width sign with side arrows and metric value.",
    ),
    CandidateSpec(
        "File:SA road sign - Maximum Width.svg",
        "official_style_reference_diagram",
        "Saudi Arabia official-style maximum-width sign with metric value.",
    ),
    CandidateSpec(
        "File:Brunei road sign - Width Restriction.svg",
        "official_style_reference_diagram",
        "Brunei official-style width restriction sign.",
    ),
    CandidateSpec(
        "File:Philippines road sign R6-2.svg",
        "official_style_reference_diagram",
        "Philippines official-style width-limit sign.",
    ),
    CandidateSpec(
        "File:Singapore road sign - Prohibitory - Width limit.svg",
        "official_style_reference_diagram",
        "Singapore official-style prohibitory width-limit sign.",
    ),
    CandidateSpec(
        "File:Indian Road Sign width limit.svg",
        "official_style_reference_diagram",
        "India official-style width-limit sign with metric value.",
    ),
    CandidateSpec(
        "File:Finland road sign C21-2.0.svg",
        "official_style_reference_diagram",
        "Finland C21 maximum-width sign, 2.0 m value.",
    ),
    CandidateSpec(
        "File:Finland road sign C21-2.1.svg",
        "official_style_reference_diagram",
        "Finland C21 maximum-width sign, 2.1 m value.",
    ),
    CandidateSpec(
        "File:Finland road sign C21-2.2.svg",
        "official_style_reference_diagram",
        "Finland C21 maximum-width sign, 2.2 m value.",
    ),
    CandidateSpec(
        "File:Finland road sign C21-2.3.svg",
        "official_style_reference_diagram",
        "Finland C21 maximum-width sign, 2.3 m value.",
    ),
    CandidateSpec(
        "File:Finland road sign C21-2.4.svg",
        "official_style_reference_diagram",
        "Finland C21 maximum-width sign, 2.4 m value.",
    ),
    CandidateSpec(
        "File:Finland road sign C21-2.5.svg",
        "official_style_reference_diagram",
        "Finland C21 maximum-width sign, 2.5 m value.",
    ),
    CandidateSpec(
        "File:Belgian road sign C27.svg",
        "official_style_reference_diagram",
        "Belgian maximum-width sign with metric value.",
    ),
    CandidateSpec(
        "File:Nederlands verkeersbord C18 2023.svg",
        "official_style_reference_diagram",
        "Netherlands C18 maximum-width sign with metric value.",
    ),
    CandidateSpec(
        "File:France road sign B11.svg",
        "official_style_reference_diagram",
        "France B11 maximum-width sign with metric value.",
    ),
    CandidateSpec(
        "File:Mexico road sign SR-16.svg",
        "official_style_reference_diagram",
        "Mexico official-style maximum-width sign with metric value.",
    ),
    CandidateSpec(
        "File:UA road sign 3.17.svg",
        "official_style_reference_diagram",
        "Ukraine official-style maximum-width sign with metric value.",
    ),
    CandidateSpec(
        "File:Jamaica road sign R21.svg",
        "official_style_reference_diagram",
        "Jamaica maximum-width sign with metric value.",
    ),
    CandidateSpec(
        "File:Width restriction - geograph.org.uk - 804033.jpg",
        "real_road_photo_visual_match",
        "Real road photo of a width restriction sign.",
    ),
    CandidateSpec(
        "File:Maximum width.jpg",
        "official_style_reference_diagram",
        "Clear maximum-width sign with side arrows and metric value.",
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


def slug(value: str) -> str:
    value = value.replace("File:", "")
    value = value.encode("ascii", errors="ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")[:120] or "candidate"


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


def commons_pages(width: int = 1000) -> dict[str, dict[str, Any]]:
    params = {
        "action": "query",
        "format": "json",
        "titles": "|".join(spec.title for spec in CANDIDATES),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": str(width),
    }
    response: requests.Response | None = None
    for attempt in range(1, 9):
        response = requests.get(
            COMMONS_API_URL,
            params=params,
            timeout=60,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code != 429:
            response.raise_for_status()
            break
        time.sleep(min(45.0, 3.0 * attempt))
    if response is None:
        raise RuntimeError("No Commons response for width_restriction candidates")
    response.raise_for_status()
    data = response.json()
    pages = {
        page["title"]: page
        for page in data.get("query", {}).get("pages", {}).values()
        if "missing" not in page and "imageinfo" in page
    }
    missing = [spec.title for spec in CANDIDATES if spec.title not in pages]
    if missing:
        raise RuntimeError(f"Commons files were not found: {missing}")
    return pages


def candidate_download_url(image_info: dict[str, Any]) -> str:
    thumb_url = image_info.get("thumburl", "")
    if thumb_url:
        return thumb_url
    source_url = image_info.get("url", "")
    if not source_url:
        raise RuntimeError("Commons image info did not include a downloadable URL")
    if image_info.get("mime") == "image/svg+xml":
        raise RuntimeError("Commons SVG image info did not include a raster thumbnail URL")
    return source_url


def download_url_content(url: str) -> bytes:
    last_error = ""
    for attempt in range(1, 9):
        try:
            response = requests.get(
                url,
                timeout=60,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code == 429:
                last_error = "HTTP 429 from upload.wikimedia.org"
                time.sleep(min(45.0, 3.0 * attempt))
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                last_error = f"Unexpected content type {content_type}"
                time.sleep(0.5)
                continue
            Image.open(BytesIO(response.content)).verify()
            return response.content
        except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
            last_error = str(exc)
            time.sleep(min(20.0, 1.5 * attempt))
    raise RuntimeError(f"Could not download raster URL {url}: {last_error}")


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
    image.convert("RGB").save(buffer, format="JPEG", quality=94, optimize=True)
    content = buffer.getvalue()
    path.write_bytes(content)
    return sha256_bytes(content), image.width, image.height


def red_sign_crop(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], str]:
    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = 1.0
    work = rgb
    if max(height, width) > 1600:
        scale = 1600 / max(height, width)
        work = cv2.resize(
            rgb,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
    low_red = cv2.inRange(hsv, np.array([0, 45, 45]), np.array([16, 255, 255]))
    high_red = cv2.inRange(hsv, np.array([164, 45, 45]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(low_red, high_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, int, int, int, int] | None = None
    image_area = work.shape[0] * work.shape[1]
    for contour in contours:
        x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if bbox_width < 18 or bbox_height < 18 or area < 70:
            continue
        aspect = bbox_width / bbox_height
        density = area / (bbox_width * bbox_height)
        if aspect < 0.45 or aspect > 2.1 or density < 0.025:
            continue
        score = area * (1.25 - min(abs(aspect - 1.0), 1.0) * 0.3) * (0.8 + min(density, 0.55))
        if bbox_width * bbox_height > image_area * 0.4:
            score *= 0.2
        if best is None or score > best[0]:
            best = (score, x, y, bbox_width, bbox_height)

    if best is None:
        side = min(width, height)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        bbox = (x1, y1, x1 + side, y1 + side)
        return image.crop(bbox), bbox, "center_fallback_no_red_component"

    _, x, y, bbox_width, bbox_height = best
    x = int(x / scale)
    y = int(y / scale)
    bbox_width = int(bbox_width / scale)
    bbox_height = int(bbox_height / scale)
    pad = int(max(bbox_width, bbox_height) * 0.45)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + bbox_width + pad)
    y2 = min(height, y + bbox_height + pad)

    side = max(x2 - x1, y2 - y1)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)
    bbox = (x1, y1, x2, y2)
    return image.crop(bbox), bbox, "red_component_auto_crop"


def materialize_candidates() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pages = commons_pages()
    for index, spec in enumerate(CANDIDATES, start=1):
        page = pages[spec.title]
        image_info = page["imageinfo"][0]
        download_url = candidate_download_url(image_info)
        candidate_id = f"MAN01-WR-{index:04d}"
        extension = ".png" if "png" in download_url.lower() or image_info.get("mime") == "image/svg+xml" else ".jpg"
        original_path = ORIGINAL_ROOT / f"{candidate_id}_{slug(spec.title)}{extension}"
        if original_path.exists() and original_path.stat().st_size > 0:
            content = original_path.read_bytes()
        else:
            content = download_url_content(download_url)
        image = image_from_bytes(content)
        crop, bbox, crop_method = red_sign_crop(image)
        crop_path = CROP_ROOT / f"{candidate_id}_{slug(spec.title)}.jpg"
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
                    "Visual match against width_restriction reference: red circular "
                    "maximum-width sign with inward side arrows, metric width value, or "
                    "clear road-photo equivalent."
                ),
                "assignment_reference_path": project_rel(REFERENCE_PATH),
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
    cols = 6
    tile_width, tile_height = 220, 230
    sheet_rows = max(1, (len(rows) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tile_width, sheet_rows * tile_height), (236, 236, 236))
    for index, row in enumerate(rows):
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        try:
            image = Image.open(PROJECT_ROOT / row["local_crop_path"]).convert("RGB")
            image = ImageOps.contain(image, (192, 152), Image.Resampling.LANCZOS)
            tile.paste(image, ((tile_width - image.width) // 2, 7))
        except OSError:
            pass
        draw = ImageDraw.Draw(tile)
        draw.text((7, 164), row["candidate_id"], fill="black")
        draw.text((7, 181), row["source_modality"][:31], fill=(60, 60, 60))
        draw.text((7, 198), row["source_title"].replace("File:", "")[:31], fill=(60, 60, 60))
        sheet.paste(tile, ((index % cols) * tile_width, (index // cols) * tile_height))
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
            "Review the Stage C manual width_restriction contact sheet, then run Stage D QC "
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
