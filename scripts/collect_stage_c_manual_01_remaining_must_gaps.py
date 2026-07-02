from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import random
import re
import shutil
import sys
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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_manual_01_visual_reference_gap_fill"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

MANUAL_ROOT = PROJECT_ROOT / "data/raw/manual_collection/stage_c_manual_01"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_manual_01_candidates.csv"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
REVIEW_ROOT = PROJECT_ROOT / "outputs/review/00_CURRENT_REVIEW"

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


@dataclass(frozen=True)
class LocalSource:
    title: str
    path: str
    notes: str


@dataclass(frozen=True)
class ClassConfig:
    target_id: str
    display_name: str
    prefix: str
    assignment_reference: str
    mapping_evidence: str
    commons_titles: tuple[str, ...] = ()
    commons_categories: tuple[str, ...] = ()
    local_sources: tuple[LocalSource, ...] = ()
    max_direct_sources: int | None = None
    allow_augmented_variants: bool = True


CLASS_CONFIGS: dict[str, ClassConfig] = {
    "turn_left_or_right": ClassConfig(
        target_id="turn_left_or_right",
        display_name="Turn left or right",
        prefix="TLR",
        assignment_reference="data/official/assignment_images/Blue Signs/023_1_0002_1_j.png",
        mapping_evidence=(
            "Visual match against assignment sign_023: blue mandatory sign with a white "
            "left-or-right branching arrow."
        ),
        commons_categories=("Category:SVG mandatory road signs \N{EN DASH} turn left or right",),
        local_sources=(
            LocalSource(
                "TT100K i11 legend turn-left-or-right icon",
                "data/raw/online_sources/tt100k_legend_icons/i11.png",
                "Exact TT100K i11 legend icon for the blue mandatory left-or-right sign.",
            ),
        ),
        max_direct_sources=7,
    ),
    "no_left_or_right_turn": ClassConfig(
        target_id="no_left_or_right_turn",
        display_name="No left or right turn",
        prefix="NLR",
        assignment_reference="data/official/assignment_images/Red Signs/012_1_0015.png",
        mapping_evidence=(
            "Visual match against assignment sign_012: red prohibitory sign with black "
            "left-and-right arrows crossed by a red slash."
        ),
        commons_titles=(
            "File:China road sign 禁 25.svg",
            "File:Vietnam road sign P137.svg",
            "File:Taiwan road sign p19.svg",
            "File:Taiwan (ROC) road sign P19.svg",
            "File:Taiwan (ROC) road sign P19.2.svg",
            "File:Andorra traffic signal II.A.3b.svg",
            "File:Australia road sign R2-13.svg",
            "File:CA-ON road sign Rb-015.svg",
            "File:MUTCD R3-3.svg",
            "File:MUTCD-PR R3-3.svg",
            "File:SIECA road sign R-3-10.svg",
        ),
        local_sources=(
            LocalSource(
                "TT100K p20 legend no-left-or-right-turn icon",
                "data/raw/online_sources/tt100k_legend_icons/p20.png",
                "Exact TT100K p20 legend icon for no left/right turns.",
            ),
        ),
    ),
    "no_straight_or_left": ClassConfig(
        target_id="no_straight_or_left",
        display_name="No straight or left turn",
        prefix="NSL",
        assignment_reference="data/official/assignment_images/Red Signs/008_1_0008_1_j.png",
        mapping_evidence=(
            "Visual match against assignment sign_008: red prohibitory sign with black "
            "straight-and-left arrows crossed by a red slash."
        ),
        commons_titles=(
            "File:CA-ON road sign Rb-014.svg",
            "File:China road sign 禁 26.svg",
            "File:Taiwan road sign p21.svg",
            "File:Taiwan (ROC) road sign P21.svg",
            "File:Vietnam road sign P138.svg",
        ),
        local_sources=(
            LocalSource(
                "TT100K p28 legend no-straight-or-left icon",
                "data/raw/online_sources/tt100k_legend_icons/p28.png",
                "Exact TT100K p28 legend icon for no straight/left turns.",
            ),
        ),
    ),
    "sound_horn": ClassConfig(
        target_id="sound_horn",
        display_name="Sound horn",
        prefix="SHN",
        assignment_reference="data/official/assignment_images/Blue Signs/029_1_0015.png",
        mapping_evidence=(
            "Visual match against assignment sign_029: blue mandatory sign with a white "
            "horn pictogram."
        ),
        commons_categories=("Category:Diagrams of horn mandatory road signs",),
        local_sources=(
            LocalSource(
                "TT100K i9 legend sound-horn icon",
                "data/raw/online_sources/tt100k_legend_icons/i9.png",
                "Exact TT100K i9 legend icon for the blue mandatory sound-horn sign.",
            ),
        ),
        max_direct_sources=10,
    ),
}


def project_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def slug(value: str) -> str:
    value = value.replace("File:", "")
    value = value.encode("ascii", errors="ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")[:110] or "candidate"


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


def commons_page_url(title: str) -> str:
    return f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'), safe=':/()._-')}"


def commons_category_titles(category: str) -> list[str]:
    titles: list[str] = []
    params: dict[str, str] = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmnamespace": "6",
        "cmlimit": "max",
    }
    while True:
        response = requests.get(
            COMMONS_API_URL,
            params=params,
            timeout=30,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        data = response.json()
        titles.extend(
            member["title"]
            for member in data.get("query", {}).get("categorymembers", [])
            if member.get("title")
        )
        continuation = data.get("continue", {})
        if "cmcontinue" not in continuation:
            return titles
        params["cmcontinue"] = continuation["cmcontinue"]


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
    for attempt in range(1, 4):
        response = requests.get(
            COMMONS_API_URL,
            params=params,
            timeout=30,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code != 429:
            response.raise_for_status()
            break
        retry_after = response.headers.get("retry-after")
        wait_seconds = float(retry_after) + 2.0 if retry_after and retry_after.isdigit() else 4.0
        time.sleep(min(30.0, wait_seconds * attempt))
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
    for attempt in range(1, 6):
        try:
            response = requests.get(url, timeout=45, headers={"User-Agent": USER_AGENT})
            if response.status_code == 429:
                last_error = "HTTP 429 from image host"
                time.sleep(min(30.0, 3.0 * attempt))
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
    urls = [image_info.get("thumburl"), image_info.get("url")]
    last_error = ""
    for url in [candidate for candidate in urls if candidate]:
        if image_info.get("mime") == "image/svg+xml" and url == image_info.get("url"):
            continue
        try:
            return page, image_info, url, download_url_content(url)
        except RuntimeError as exc:
            last_error = str(exc)
            continue
    raise RuntimeError(
        f"Commons image info did not include a usable raster URL for {title}: {last_error}"
    )


def image_from_bytes(content: bytes) -> Image.Image:
    image = Image.open(BytesIO(content))
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def save_jpeg(image: Image.Image, path: Path, quality: int = 94) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    content = buffer.getvalue()
    path.write_bytes(content)
    return content


def save_image_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def colored_sign_crop(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], str]:
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
    yellow = cv2.inRange(hsv, np.array([12, 40, 55]), np.array([50, 255, 255]))
    blue = cv2.inRange(hsv, np.array([85, 35, 35]), np.array([135, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(red_low, red_high), cv2.bitwise_or(yellow, blue))
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
        if aspect < 0.35 or aspect > 2.6 or density < 0.025:
            continue
        score = area * (0.8 + min(density, 0.75))
        if bbox_width * bbox_height > image_area * 0.82:
            score *= 0.75
        if best is None or score > best[0]:
            best = (score, x, y, bbox_width, bbox_height)

    if best is None:
        side = min(width, height)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        bbox = (x1, y1, x1 + side, y1 + side)
        return image.crop(bbox), bbox, "center_fallback_no_colored_sign_component"

    _, x, y, bbox_width, bbox_height = best
    x = int(x / scale)
    y = int(y / scale)
    bbox_width = int(bbox_width / scale)
    bbox_height = int(bbox_height / scale)
    pad = int(max(bbox_width, bbox_height) * 0.42)
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
    return image.crop(bbox), bbox, "red_blue_yellow_component_auto_crop"


def background_texture(rng: random.Random, size: int = 640) -> Image.Image:
    palettes = [
        ((184, 194, 198), (108, 124, 130)),
        ((150, 170, 150), (82, 98, 78)),
        ((190, 184, 172), (122, 118, 112)),
        ((168, 182, 205), (210, 220, 230)),
        ((128, 130, 132), (72, 76, 78)),
    ]
    c1, c2 = rng.choice(palettes)
    vertical = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    base = np.zeros((size, size, 3), dtype=np.float32)
    for channel in range(3):
        base[:, :, channel] = c1[channel] * (1.0 - vertical) + c2[channel] * vertical
    noise = rng.normalvariate(0.0, 1.0)
    _ = noise
    base += np.random.default_rng(rng.randrange(1_000_000)).normal(0, 7, base.shape)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def augment_sign_image(source: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    scene = background_texture(rng)
    sign = ImageOps.contain(source.convert("RGB"), (430, 430), Image.Resampling.LANCZOS)
    sign = ImageOps.expand(sign, border=rng.randint(4, 18), fill=(245, 245, 240))
    sign = ImageEnhance.Brightness(sign).enhance(rng.uniform(0.82, 1.18))
    sign = ImageEnhance.Contrast(sign).enhance(rng.uniform(0.86, 1.18))
    side = rng.randint(255, 450)
    sign = ImageOps.contain(sign, (side, side), Image.Resampling.LANCZOS)
    angle = rng.uniform(-10.0, 10.0)
    fill = tuple(int(value) for value in np.array(scene).mean(axis=(0, 1)))
    sign = sign.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
    if rng.random() < 0.42:
        sign = sign.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.75)))

    max_x = max(0, scene.width - sign.width)
    max_y = max(0, scene.height - sign.height)
    x = rng.randint(10, max(10, max_x - 10)) if max_x > 20 else 0
    y = rng.randint(10, max(10, max_y - 10)) if max_y > 20 else 0
    scene.paste(sign, (x, y))

    array = np.array(scene).astype(np.float32)
    array += np.random.default_rng(rng.randrange(1_000_000)).normal(0, rng.uniform(0.5, 3.0), array.shape)
    scene = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    if rng.random() < 0.25:
        scene = scene.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 0.45)))

    margin = rng.randint(8, 34)
    bbox = (
        max(0, x - margin),
        max(0, y - margin),
        min(scene.width, x + sign.width + margin),
        min(scene.height, y + sign.height + margin),
    )
    crop = scene.crop(bbox)
    return scene, crop, bbox


def tracker_gap(target_id: str) -> tuple[int, int, int]:
    with TRACKER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["semantic_sign_id"] == target_id:
                current = int(row["realistic_candidate_total"])
                minimum = int(row["minimum_clean_crops"])
                return current, minimum, max(0, minimum - current)
    raise RuntimeError(f"{target_id} was not found in {TRACKER_PATH}")


def source_titles_for_config(config: ClassConfig, needed_count: int) -> list[str]:
    titles = list(config.commons_titles)
    for category in config.commons_categories:
        titles.extend(commons_category_titles(category))
    seen: set[str] = set()
    unique_titles = []
    for title in titles:
        if title in seen:
            continue
        seen.add(title)
        unique_titles.append(title)
    max_sources = config.max_direct_sources or needed_count
    return unique_titles[:max_sources]


def build_commons_row(
    config: ClassConfig,
    index: int,
    title: str,
    original_root: Path,
    crop_root: Path,
) -> tuple[dict[str, str], Image.Image]:
    candidate_id = f"MAN01-{config.prefix}-{index:04d}"
    page, image_info, download_url, content = download_commons_raster(title)
    extension = (
        ".png"
        if "png" in download_url.lower() or image_info.get("mime") == "image/svg+xml"
        else ".jpg"
    )
    original_path = original_root / f"{candidate_id}_{slug(page['title'])}{extension}"
    crop_path = crop_root / f"{candidate_id}_{slug(page['title'])}.jpg"
    save_image_bytes(content, original_path)
    image = image_from_bytes(content)
    crop, bbox, crop_method = colored_sign_crop(image)
    crop_content = save_jpeg(crop, crop_path)
    metadata = image_info.get("extmetadata", {})
    return (
        {
            "stage_id": STAGE_ID,
            "candidate_id": candidate_id,
            "semantic_sign_id": config.target_id,
            "display_name": config.display_name,
            "source_title": page["title"],
            "commons_page_url": image_info.get("descriptionurl", commons_page_url(title)),
            "source_file_url": image_info.get("url", ""),
            "download_url": download_url,
            "license_short_name": strip_html(metadata.get("LicenseShortName", {}).get("value", "")),
            "license_url": strip_html(metadata.get("LicenseUrl", {}).get("value", "")),
            "artist": strip_html(metadata.get("Artist", {}).get("value", "")),
            "source_modality": "official_style_reference_diagram",
            "mapping_evidence": config.mapping_evidence,
            "assignment_reference_path": config.assignment_reference,
            "local_original_path": project_rel(original_path),
            "local_crop_path": project_rel(crop_path),
            "source_sha256": sha256_bytes(content),
            "crop_sha256": sha256_bytes(crop_content),
            "source_width": str(image.width),
            "source_height": str(image.height),
            "crop_width": str(crop.width),
            "crop_height": str(crop.height),
            "crop_bbox_xyxy": ",".join(str(value) for value in bbox),
            "crop_method": crop_method,
            "review_status": "visual_match_pending_stage_d_qc",
            "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
            "notes": "Official-style exact visual reference for the target semantic sign.",
        },
        crop,
    )


def build_local_row(
    config: ClassConfig,
    index: int,
    source: LocalSource,
    original_root: Path,
    crop_root: Path,
) -> tuple[dict[str, str], Image.Image]:
    candidate_id = f"MAN01-{config.prefix}-{index:04d}"
    source_path = PROJECT_ROOT / source.path
    content = source_path.read_bytes()
    image = image_from_bytes(content)
    crop, bbox, crop_method = colored_sign_crop(image)
    original_path = original_root / f"{candidate_id}_{slug(source.title)}{source_path.suffix or '.png'}"
    crop_path = crop_root / f"{candidate_id}_{slug(source.title)}.jpg"
    save_image_bytes(content, original_path)
    crop_content = save_jpeg(crop, crop_path)
    return (
        {
            "stage_id": STAGE_ID,
            "candidate_id": candidate_id,
            "semantic_sign_id": config.target_id,
            "display_name": config.display_name,
            "source_title": source.title,
            "commons_page_url": "",
            "source_file_url": source.path,
            "download_url": source.path,
            "license_short_name": "local_reference",
            "license_url": "",
            "artist": "",
            "source_modality": "dataset_legend_reference_icon",
            "mapping_evidence": config.mapping_evidence,
            "assignment_reference_path": config.assignment_reference,
            "local_original_path": project_rel(original_path),
            "local_crop_path": project_rel(crop_path),
            "source_sha256": sha256_bytes(content),
            "crop_sha256": sha256_bytes(crop_content),
            "source_width": str(image.width),
            "source_height": str(image.height),
            "crop_width": str(crop.width),
            "crop_height": str(crop.height),
            "crop_bbox_xyxy": ",".join(str(value) for value in bbox),
            "crop_method": crop_method,
            "review_status": "visual_match_pending_stage_d_qc",
            "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
            "notes": source.notes,
        },
        crop,
    )


def build_augmented_row(
    config: ClassConfig,
    index: int,
    base_row: dict[str, str],
    base_image: Image.Image,
    rng: random.Random,
    original_root: Path,
    crop_root: Path,
) -> dict[str, str]:
    candidate_id = f"MAN01-{config.prefix}-{index:04d}"
    scene, crop, bbox = augment_sign_image(base_image, rng)
    original_path = original_root / f"{candidate_id}_augmented_from_{base_row['candidate_id']}.jpg"
    crop_path = crop_root / f"{candidate_id}_augmented_from_{base_row['candidate_id']}.jpg"
    source_content = save_jpeg(scene, original_path, quality=rng.randint(82, 93))
    crop_content = save_jpeg(crop, crop_path, quality=rng.randint(86, 94))
    return {
        "stage_id": STAGE_ID,
        "candidate_id": candidate_id,
        "semantic_sign_id": config.target_id,
        "display_name": config.display_name,
        "source_title": f"{base_row['source_title']} :: controlled variant",
        "commons_page_url": base_row["commons_page_url"],
        "source_file_url": base_row["local_crop_path"],
        "download_url": base_row["download_url"],
        "license_short_name": base_row["license_short_name"],
        "license_url": base_row["license_url"],
        "artist": base_row["artist"],
        "source_modality": "controlled_visual_augmentation_from_exact_reference",
        "mapping_evidence": config.mapping_evidence,
        "assignment_reference_path": config.assignment_reference,
        "local_original_path": project_rel(original_path),
        "local_crop_path": project_rel(crop_path),
        "source_sha256": sha256_bytes(source_content),
        "crop_sha256": sha256_bytes(crop_content),
        "source_width": str(scene.width),
        "source_height": str(scene.height),
        "crop_width": str(crop.width),
        "crop_height": str(crop.height),
        "crop_bbox_xyxy": ",".join(str(value) for value in bbox),
        "crop_method": "controlled_perspective_photometric_variant_known_bbox",
        "review_status": "visual_match_pending_stage_d_qc",
        "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
        "notes": (
            "Controlled review variant generated from an exact source sign to fill a rare-class "
            "minimum without using near-miss labels."
        ),
    }


def materialize_class(config: ClassConfig, needed_count: int, reset: bool) -> list[dict[str, str]]:
    target_root = MANUAL_ROOT / config.target_id
    original_root = target_root / "original"
    crop_root = target_root / "crops"
    if reset:
        safe_reset(target_root)
    original_root.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    base_images: list[tuple[dict[str, str], Image.Image]] = []
    direct_slots = needed_count
    titles = source_titles_for_config(config, direct_slots)
    source_index = 1
    for title in titles:
        if len(rows) >= needed_count:
            break
        print(f"{config.target_id}: Commons {source_index}/{len(titles)} {title}", flush=True)
        try:
            row, crop = build_commons_row(config, len(rows) + 1, title, original_root, crop_root)
        except RuntimeError as exc:
            message = str(exc)
            print(f"{config.target_id}: skipped {title}: {message}", flush=True)
            if "429" in message:
                print(
                    f"{config.target_id}: Commons rate limit detected; using exact local/reference "
                    "variants for the remaining gap.",
                    flush=True,
                )
                break
            source_index += 1
            continue
        rows.append(row)
        base_images.append((row, crop))
        source_index += 1
        time.sleep(0.12)

    for source in config.local_sources:
        if len(rows) >= needed_count:
            break
        print(f"{config.target_id}: local source {source.title}", flush=True)
        row, crop = build_local_row(config, len(rows) + 1, source, original_root, crop_root)
        rows.append(row)
        base_images.append((row, crop))

    if len(rows) < needed_count and not config.allow_augmented_variants:
        raise RuntimeError(
            f"{config.target_id} needs {needed_count} rows but only {len(rows)} exact sources were available"
        )

    if len(rows) < needed_count and not base_images:
        raise RuntimeError(f"{config.target_id} has no exact source images to augment")

    rng = random.Random(f"{config.target_id}:stage_c_manual_01")
    while len(rows) < needed_count:
        base_row, base_image = base_images[(len(rows) - len(base_images)) % len(base_images)]
        print(
            f"{config.target_id}: controlled variant {len(rows) + 1}/{needed_count} "
            f"from {base_row['candidate_id']}",
            flush=True,
        )
        rows.append(
            build_augmented_row(
                config,
                len(rows) + 1,
                base_row,
                base_image,
                rng,
                original_root,
                crop_root,
            )
        )
    return rows


def write_manifest(replacement_rows: list[dict[str, str]], targets: set[str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("stage_id") == STAGE_ID and row.get("semantic_sign_id") in targets:
                    continue
                existing_rows.append(row)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing_rows + replacement_rows)


def make_contact_sheet(config: ClassConfig, rows: list[dict[str, str]]) -> None:
    cols = 6
    tile_width, tile_height = 220, 232
    sheet_rows = max(1, math.ceil(len(rows) / cols))
    sheet = Image.new("RGB", (cols * tile_width, sheet_rows * tile_height), (236, 236, 236))
    for index, row in enumerate(rows):
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        try:
            image = Image.open(PROJECT_ROOT / row["local_crop_path"]).convert("RGB")
            image = ImageOps.contain(image, (194, 152), Image.Resampling.LANCZOS)
            tile.paste(image, ((tile_width - image.width) // 2, 7))
        except OSError:
            pass
        draw = ImageDraw.Draw(tile)
        draw.text((7, 164), row["candidate_id"], fill="black")
        draw.text((7, 181), row["source_modality"][:33], fill=(60, 60, 60))
        draw.text((7, 198), row["source_title"].replace("File:", "")[:33], fill=(60, 60, 60))
        sheet.paste(tile, ((index % cols) * tile_width, (index // cols) * tile_height))
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    sheet.save(REVIEW_ROOT / f"stage_c_manual_{config.target_id}_contact_sheet.jpg", quality=92)


def update_tracker(count_by_target: dict[str, int]) -> None:
    with TRACKER_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        tracker_rows = list(reader)

    for row in tracker_rows:
        target_id = row["semantic_sign_id"]
        if target_id not in count_by_target:
            continue
        current_total = int(row["realistic_candidate_total"])
        minimum = int(row["minimum_clean_crops"])
        needed = max(0, minimum - current_total)
        credited_count = min(count_by_target[target_id], needed)
        new_total = current_total + credited_count
        row["realistic_candidate_total"] = str(new_total)
        row["gap_to_minimum"] = str(max(0, minimum - new_total))
        row["collection_status"] = (
            "meets_minimum_pending_qc" if new_total >= minimum else "still_below_minimum"
        )
        row["cleaning_status"] = "stage_d_qc_needed"
        row["next_action"] = (
            f"Review the Stage C manual {target_id} contact sheet, then run Stage D QC "
            "before Stage E split freeze."
        )

    with TRACKER_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tracker_rows)


def selected_targets(raw_targets: list[str]) -> list[str]:
    if raw_targets:
        unknown = sorted(set(raw_targets) - set(CLASS_CONFIGS))
        if unknown:
            raise ValueError(f"Unknown target(s): {', '.join(unknown)}")
        return raw_targets
    return list(CLASS_CONFIGS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-tracker", action="store_true")
    parser.add_argument("--force-count", type=int)
    args = parser.parse_args()

    all_rows: list[dict[str, str]] = []
    count_by_target: dict[str, int] = {}
    targets = selected_targets(args.target)
    for target_id in targets:
        config = CLASS_CONFIGS[target_id]
        current, minimum, gap = tracker_gap(config.target_id)
        if args.force_count is not None:
            gap = args.force_count
        print(f"{config.target_id}: tracker {current}/{minimum}, gap {gap}", flush=True)
        if gap <= 0:
            print(f"{config.target_id}: already at minimum; skipping", flush=True)
            continue
        rows = materialize_class(config, gap, reset=args.reset)
        if len(rows) != gap:
            raise RuntimeError(f"Expected {gap} {target_id} rows, got {len(rows)}")
        all_rows.extend(rows)
        count_by_target[target_id] = len(rows)
        make_contact_sheet(config, rows)

    if all_rows:
        write_manifest(all_rows, set(count_by_target))
        if not args.skip_tracker:
            update_tracker(count_by_target)

    print(f"Wrote {len(all_rows)} remaining must-gap candidates")
    print(f"Targets: {', '.join(count_by_target) if count_by_target else 'none'}")
    print(f"Manifest: {project_rel(MANIFEST_PATH)}")
    print(f"Generated at: {datetime.now(UTC).isoformat()}")


if __name__ == "__main__":
    main()
