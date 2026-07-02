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
from urllib.parse import quote

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_manual_01_visual_commons"
TARGET_ID = "tractors_ahead"
DISPLAY_NAME = "Tractors ahead"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"

MANUAL_ROOT = PROJECT_ROOT / "data/raw/manual_collection/stage_c_manual_01/tractors_ahead"
ORIGINAL_ROOT = MANUAL_ROOT / "original"
CROP_ROOT = MANUAL_ROOT / "crops"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_manual_01_candidates.csv"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
CONTACT_SHEET_PATH = (
    PROJECT_ROOT
    / "outputs/review/00_CURRENT_REVIEW/stage_c_manual_tractors_ahead_contact_sheet.jpg"
)
ASSIGNMENT_REFERENCE_PATH = PROJECT_ROOT / "data/official/assignment_images/Yellow Signs/051_0005_j.png"


@dataclass(frozen=True)
class CandidateSpec:
    title: str
    source_modality: str
    notes: str


DIAGRAM_TITLES: tuple[str, ...] = (
    "File:1999 Brazil road sign A-31.svg",
    "File:Argentina MSV 2017 road sign P-29(b).svg",
    "File:Argentina P-29B.svg",
    "File:Australia road sign W5-50.svg",
    "File:BO road sign SP-51.svg",
    "File:Brasil A-31.svg",
    "File:CA-BC road sign W-313-1.svg",
    "File:CA-BC road sign W-314-L.svg",
    "File:CA-BC road sign W-314-R.svg",
    "File:CL road sign PO-3.svg",
    "File:Colombia road sign SP-45.svg",
    "File:COPACA Road sign P-30.svg",
    "File:Ecuador road sign P6-12.svg",
    "File:HR road sign A30.svg",
    "File:IE road sign W-168.svg",
    "File:Israel road sign 147.svg",
    "File:MA road sign 126 (blue).svg",
    "File:MA road sign 126.svg",
    "File:Maltese road sign I.E10.svg",
    "File:Mexico road sign SP-36.svg",
    "File:MUTCD W11-5.svg",
    "File:MUTCD W11-5a.svg",
    "File:MX road sign SP-36.svg",
    "File:Panama P-31.svg",
    "File:Paraguay road sign P-53.svg",
    "File:Peru road sign P-51.svg",
    "File:Portugal road sign A34.svg",
    "File:UK traffic sign 553.1.svg",
    "File:RO road sign A53.svg",
    "File:SADC road sign TW352-RHT.svg",
    "File:SADC road sign TW352.svg",
    "File:SADC road sign W352-RHT.svg",
    "File:SADC road sign W352.svg",
    "File:SIECA road sign P-10-5.svg",
    "File:Sweden road sign A31.svg",
    "File:UG road sign W47.svg",
)

CANDIDATES: tuple[CandidateSpec, ...] = tuple(
    CandidateSpec(
        title,
        "official_style_reference_diagram",
        "Official-style tractor/farm-vehicle warning sign diagram.",
    )
    for title in DIAGRAM_TITLES
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


def commons_file_page_url(title: str) -> str:
    return f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'), safe=':/()._-')}"


def cached_commons_info(title: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    page_url = commons_file_page_url(title)
    return (
        {"title": title},
        {
            "descriptionurl": page_url,
            "url": "",
            "thumburl": "",
            "mime": "image/png",
            "extmetadata": {},
        },
        page_url,
    )


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
    for _attempt in range(1, 4):
        response = requests.get(
            COMMONS_API_URL,
            params=params,
            timeout=45,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code != 429:
            response.raise_for_status()
            break
        retry_after = response.headers.get("retry-after")
        wait_seconds = float(retry_after) + 2.0 if retry_after and retry_after.isdigit() else 5.0
        time.sleep(min(30.0, wait_seconds))
    if response is None:
        raise RuntimeError(f"No Commons response for {title}")
    response.raise_for_status()
    data = response.json()
    page = next(iter(data.get("query", {}).get("pages", {}).values()), None)
    if not page or "missing" in page or "imageinfo" not in page:
        raise RuntimeError(f"Commons file was not found: {title}")
    return page


def download_url_content(url: str) -> bytes:
    last_error = ""
    for attempt in range(1, 7):
        try:
            response = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
            if response.status_code == 429:
                last_error = "HTTP 429 from upload.wikimedia.org"
                retry_after = response.headers.get("retry-after")
                wait_seconds = (
                    float(retry_after) + 2.0
                    if retry_after and retry_after.isdigit()
                    else 5.0 * attempt
                )
                time.sleep(min(660.0, wait_seconds))
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                last_error = f"Unexpected content type {content_type}"
                continue
            Image.open(BytesIO(response.content)).verify()
            return response.content
        except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
            last_error = str(exc)
            time.sleep(min(20.0, 1.5 * attempt))
    raise RuntimeError(f"Could not download raster URL {url}: {last_error}")


def download_commons_raster(title: str) -> tuple[dict[str, Any], dict[str, Any], str, bytes]:
    page = commons_image_info(title, 900)
    image_info = page["imageinfo"][0]
    download_url = image_info.get("thumburl") or image_info.get("url", "")
    if image_info.get("mime") != "image/svg+xml" and image_info.get("url"):
        download_url = image_info["url"]
    if not download_url:
        raise RuntimeError(f"Commons image info did not include a URL for {title}")
    if image_info.get("mime") == "image/svg+xml" and download_url == image_info.get("url"):
        raise RuntimeError(f"Commons SVG image info did not include a thumbnail for {title}")
    return page, image_info, download_url, download_url_content(download_url)


def image_from_bytes(content: bytes) -> Image.Image:
    image = Image.open(BytesIO(content))
    image = ImageOps.exif_transpose(image)
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


def warning_sign_crop(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], str]:
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
    red_low = cv2.inRange(hsv, np.array([0, 45, 45]), np.array([15, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([165, 45, 45]), np.array([179, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([12, 35, 55]), np.array([50, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(red_low, red_high), yellow)
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
        if aspect < 0.4 or aspect > 2.4 or density < 0.025:
            continue
        score = area * (0.8 + min(density, 0.75))
        if bbox_width * bbox_height > image_area * 0.75:
            score *= 0.4
        if best is None or score > best[0]:
            best = (score, x, y, bbox_width, bbox_height)

    if best is None:
        side = min(width, height)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        bbox = (x1, y1, x1 + side, y1 + side)
        return image.crop(bbox), bbox, "center_fallback_no_warning_component"

    _, x, y, bbox_width, bbox_height = best
    x = int(x / scale)
    y = int(y / scale)
    bbox_width = int(bbox_width / scale)
    bbox_height = int(bbox_height / scale)
    pad = int(max(bbox_width, bbox_height) * 0.55)
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
    return image.crop(bbox), bbox, "warning_color_component_auto_crop"


def materialize_candidates() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, spec in enumerate(CANDIDATES, start=1):
        print(
            f"Tractor candidate {index}/{len(CANDIDATES)}: "
            f"{spec.title.encode('unicode_escape').decode()}",
            flush=True,
        )
        candidate_id = f"MAN01-TA-{index:04d}"
        expected_slug = slug(spec.title)
        cached_originals = [
            ORIGINAL_ROOT / f"{candidate_id}_{expected_slug}.png",
            ORIGINAL_ROOT / f"{candidate_id}_{expected_slug}.jpg",
            ORIGINAL_ROOT / f"{candidate_id}_{expected_slug}.jpeg",
        ]
        original_path = next(
            (path for path in cached_originals if path.exists() and path.stat().st_size > 0),
            cached_originals[0],
        )
        crop_path = CROP_ROOT / f"{candidate_id}_{expected_slug}.jpg"
        if original_path.exists() and original_path.stat().st_size > 0:
            content = original_path.read_bytes()
            page, image_info, download_url = cached_commons_info(spec.title)
        else:
            page, image_info, download_url, content = download_commons_raster(spec.title)
            extension = (
                ".png"
                if "png" in download_url.lower() or image_info.get("mime") == "image/svg+xml"
                else ".jpg"
            )
            original_path = ORIGINAL_ROOT / f"{candidate_id}_{slug(page['title'])}{extension}"
            crop_path = CROP_ROOT / f"{candidate_id}_{slug(page['title'])}.jpg"
        image = image_from_bytes(content)
        crop, bbox, crop_method = warning_sign_crop(image)
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
                    "Visual match against assignment reference sign_051: tractor or "
                    "farm-vehicle warning sign pictogram."
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
        time.sleep(0.2)
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
    tile_width, tile_height = 215, 230
    sheet_rows = max(1, (len(rows) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tile_width, sheet_rows * tile_height), (236, 236, 236))
    for index, row in enumerate(rows):
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        try:
            image = Image.open(PROJECT_ROOT / row["local_crop_path"]).convert("RGB")
            image = ImageOps.contain(image, (190, 152), Image.Resampling.LANCZOS)
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
            "Review the Stage C manual tractors_ahead contact sheet, then run Stage D QC "
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
