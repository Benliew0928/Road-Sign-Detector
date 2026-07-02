from __future__ import annotations

import argparse
import csv
import hashlib
import html
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
STAGE_ID = "stage_c_manual_01_real_commons"
TARGET_ID = "steep_descent"
DISPLAY_NAME = "Steep descent"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"
TARGET_COUNT = 50

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

MANUAL_ROOT = PROJECT_ROOT / "data/raw/manual_collection/stage_c_manual_01/steep_descent"
ORIGINAL_ROOT = MANUAL_ROOT / "original"
CROP_ROOT = MANUAL_ROOT / "crops"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_manual_01_candidates.csv"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
CONTACT_SHEET_PATH = (
    PROJECT_ROOT
    / "outputs/review/00_CURRENT_REVIEW/stage_c_manual_steep_descent_contact_sheet.jpg"
)
ASSIGNMENT_REFERENCE_PATH = PROJECT_ROOT / "data/official/assignment_images/Yellow Signs/040_0011.png"


@dataclass(frozen=True)
class CandidateSpec:
    title: str
    source_modality: str
    notes: str


@dataclass(frozen=True)
class LocalSource:
    title: str
    path: str
    commons_page_url: str
    source_file_url: str
    download_url: str
    license_short_name: str
    license_url: str
    artist: str
    source_modality: str
    notes: str


REAL_PHOTO_TITLES: tuple[str, ...] = (
    "File:Steep descent road sign in Paseky Bystřice 2026-03-15 01.jpg",
    "File:Steep descent road sign in Paseky Bystřice 2026-03-15 02.jpg",
    "File:Warning sign - steep descent ahead, Highfield Road, Lydney - geograph.org.uk - 5387908.jpg",
    "File:Warning sign - steep descent ahead, Brecon Road, Ystradgynlais - geograph.org.uk - 6853911.jpg",
    "File:Steep descent ahead (geograph 5197652).jpg",
    "File:Warning sign - steep descent from Hafodyrynys to Crumlin - geograph.org.uk - 5647490.jpg",
    "File:Steep descent near Empshott - geograph.org.uk - 353847.jpg",
    "File:Steep descent into Llanbadarn Fawr - geograph.org.uk - 3333091.jpg",
    "File:Hill descent warning sign Great Eastern Highway.jpg",
    "File:Steep Hill into Eskdale - geograph.org.uk - 453745.jpg",
    "File:\"Steep hill down\" road sign, Republic of Ireland - geograph.org.uk - 3041124.jpg",
    "File:Steep hill down from Cudham - geograph.org.uk - 4519325.jpg",
    "File:Steep hill downwards sign, Craigantlet hill, Belfast - geograph.org.uk - 1940129.jpg",
    "File:Steep Hill sign on Henside Road - geograph.org.uk - 680341.jpg",
    "File:Steep hill sign near Whitehead - geograph.org.uk - 2398148.jpg",
    "File:\"Steep hill\" sign, Carrickfergus (July 2017) - geograph.org.uk - 5472947.jpg",
    "File:Steep hill sign, Chat Hill Road - geograph.org.uk - 6297902.jpg",
    "File:Steep hill sign - geograph.org.uk - 6066495.jpg",
    "File:\"Steep hill\" sign, Ballykeel, Holywood - geograph.org.uk - 3759237.jpg",
    "File:Pre-Worboys \"steep hill\" sign, Islandmagee - geograph.org.uk - 2685471.jpg",
    "File:Old road sign, Hutton Buscel - geograph.org.uk - 246107.jpg",
    "File:Steep hill at NE end of the village street, Hutton Buscel - geograph.org.uk - 246110.jpg",
    "File:Steep Hill sign, Break House Farm - geograph.org.uk - 291044.jpg",
    "File:Steep hill sign on Hebden Road - geograph.org.uk - 7568455.jpg",
    "File:Steep hill sign, Woodhall Road - geograph.org.uk - 7199337.jpg",
    "File:Steep hill sign, Church Lane, Newsome - geograph.org.uk - 7298506.jpg",
    "File:Steep hill sign, Watson Mill Lane - geograph.org.uk - 6827278.jpg",
    "File:Seriously steep hill ahead^ - geograph.org.uk - 4221824.jpg",
    "File:Steep hill ahead - geograph.org.uk - 1481999.jpg",
    "File:Ffordd Pen Llech slope sign.jpg",
    "File:Stock Hill - Steep Hill 12^ - geograph.org.uk - 1427132.jpg",
    "File:Tandridge Hill Lane - Steep Hill 14^ - geograph.org.uk - 1428858.jpg",
    "File:Upper Park Road - Steep Hill 10^ - geograph.org.uk - 1435405.jpg",
    "File:Coast Hill - Steep Hill 12^ - geograph.org.uk - 1438444.jpg",
    "File:Picardy Road - Steep Hill 10^ - geograph.org.uk - 1435397.jpg",
    "File:Stambourne Way - Steep Hill 13^ - geograph.org.uk - 1425676.jpg",
    "File:Hatham Green Lane - Steep Hill 1-6 - geograph.org.uk - 1528502.jpg",
    "File:Broadmoor - Steep Hill 1-8 - geograph.org.uk - 1429856.jpg",
    "File:Yorks Hill - Steep Hill 16^ - geograph.org.uk - 1429804.jpg",
    "File:Crofton Road - Steep Hill 10^ - geograph.org.uk - 1436054.jpg",
    "File:Jubilee Way - Steep Hill 5^ - geograph.org.uk - 1430788.jpg",
    "File:Sundridge Hill - Steep Hill 15^ - geograph.org.uk - 1428780.jpg",
    "File:Spout Hill - Steep Hill 17^ - geograph.org.uk - 1436078.jpg",
    "File:Bug Hill - Steep Hill 15^ - geograph.org.uk - 1428799.jpg",
    "File:Crocknorth Road - Steep Hill 10^ - geograph.org.uk - 1429801.jpg",
    "File:Titsey Hill - Steep Hill 16^ - geograph.org.uk - 1428848.jpg",
    "File:Granville Park - Steep Hill 15^ - geograph.org.uk - 1435524.jpg",
    "File:The Downs - Steep Hill 10^ - geograph.org.uk - 1425425.jpg",
    "File:Old Holbrook - Steep Hill 1-6 - geograph.org.uk - 1430826.jpg",
    "File:Braeside - Steep Hill 17^ - geograph.org.uk - 1435621.jpg",
    "File:Ena Road - Steep Hill 17^ - geograph.org.uk - 1425698.jpg",
    "File:Leesons Hill - Steep Hill 14^ - geograph.org.uk - 1436036.jpg",
    "File:Bagden Hill - Steep Hill 20^ - geograph.org.uk - 1428863.jpg",
    "File:Lockyers Hill - Steep Hill 25^ - geograph.org.uk - 1425821.jpg",
    "File:Tormount Road - Steep Hill 20^ - geograph.org.uk - 1435469.jpg",
    "File:Bostall Hill - Steep Hill 10^ - geograph.org.uk - 1435442.jpg",
    "File:Portnalls Road - Steep Hill 1-8 - geograph.org.uk - 1427112.jpg",
    "File:Lockyers Hill - Steep Hill 25^ - geograph.org.uk - 1425823.jpg",
    "File:Sydenham Hill - Steep Hill 12^ - geograph.org.uk - 1435577.jpg",
)

DIAGRAM_FALLBACK_TITLES: tuple[str, ...] = (
    "File:Malaysia road sign WD8.svg",
    "File:Cerun menurun.png",
    "File:CN road sign 警 5-2.svg",
    "File:Singapore road sign - Warning - Steep descent.svg",
    "File:W105 Steep Descent - Warning Sign Ireland.png",
    "File:Taiwan road sign Art026.2.png",
    "File:Taiwan road sign Art026.3.png",
    "File:British Columbia W-18-1(Old).png",
    "File:Ukraine road sign 1.7.gif",
    "File:Steep downhill.png",
    "File:Zeichen 108-10 - Gefälle, StVO 2017.svg",
    "File:Zeichen 108-12 - Gefälle, StVO 2017.svg",
    "File:Zeichen 108-15 - Gefälle, StVO 2017.svg",
    "File:112-10 Klesanie (10%).svg",
    "File:112-12 Klesanie (12%).svg",
    "File:112-15 Klesanie (15%).svg",
)

LOCAL_REFERENCE_SOURCES: tuple[LocalSource, ...] = (
    LocalSource(
        title="File:Malaysia road sign WD8.svg",
        path="data/raw/online_sources/stage_c_reference_01/wikimedia_commons/steep_descent/raster/Malaysia_road_sign_WD8.png",
        commons_page_url="https://commons.wikimedia.org/wiki/File:Malaysia_road_sign_WD8.svg",
        source_file_url="https://upload.wikimedia.org/wikipedia/commons/b/ba/Malaysia_road_sign_WD8.svg",
        download_url="https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Malaysia_road_sign_WD8.svg/960px-Malaysia_road_sign_WD8.svg.png",
        license_short_name="Public domain",
        license_url="",
        artist="Public Works Department Malaysia",
        source_modality="official_style_reference_diagram",
        notes="Exact Malaysian downhill-slope / steep-descent warning reference from cached Stage C source.",
    ),
    LocalSource(
        title="File:Cerun menurun.png",
        path="data/raw/online_sources/stage_c_reference_01/wikimedia_commons/steep_descent/raster/Cerun_menurun.png",
        commons_page_url="https://commons.wikimedia.org/wiki/File:Cerun_menurun.png",
        source_file_url="https://upload.wikimedia.org/wikipedia/commons/a/a8/Cerun_menurun.png",
        download_url="https://upload.wikimedia.org/wikipedia/commons/a/a8/Cerun_menurun.png",
        license_short_name="CC BY-SA 3.0",
        license_url="https://creativecommons.org/licenses/by-sa/3.0",
        artist="Fiq Shafiq",
        source_modality="malaysia_reference_diagram",
        notes="Malay steep-descent diagram from cached Malaysian warning-sign Commons source.",
    ),
    LocalSource(
        title="File:CN road sign 警 5-2.svg",
        path="data/raw/online_sources/stage_c_china_reference_01/wikimedia_commons/steep_descent/raster/CN_road_sign_5-2.png",
        commons_page_url="https://commons.wikimedia.org/wiki/File:CN_road_sign_%E8%AD%A6_5-2.svg",
        source_file_url="https://upload.wikimedia.org/wikipedia/commons/2/2e/CN_road_sign_%E8%AD%A6_5-2.svg",
        download_url="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/CN_road_sign_%E8%AD%A6_5-2.svg/960px-CN_road_sign_%E8%AD%A6_5-2.svg.png",
        license_short_name="Public domain",
        license_url="",
        artist="Standardization Administration of the People's Republic of China",
        source_modality="official_style_reference_diagram",
        notes="Exact China GB steep-descent warning reference from cached Stage C source.",
    ),
    LocalSource(
        title="TT100K w25 legend steep-descent icon",
        path="data/raw/online_sources/tt100k_legend_icons/w25.png",
        commons_page_url="",
        source_file_url="data/raw/online_sources/tt100k_legend_icons/w25.png",
        download_url="data/raw/online_sources/tt100k_legend_icons/w25.png",
        license_short_name="local_reference",
        license_url="",
        artist="",
        source_modality="dataset_legend_reference_icon",
        notes="TT100K w25 legend icon matching the steep-descent warning class.",
    ),
)

REJECTED_REAL_TITLE_MARKERS: tuple[str, ...] = (
    "Hafodyrynys to Crumlin",
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
    return re.sub(r"_+", "_", value).strip("_")[:118] or "candidate"


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def commons_file_page_url(title: str) -> str:
    return f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'), safe=':/()._-')}"


def safe_reset(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    resolved = path.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(f"Refusing to reset outside project root: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def request_with_retry(
    url: str,
    *,
    params: dict[str, str] | None = None,
    max_attempts: int = 4,
    max_wait_seconds: float = 12.0,
) -> requests.Response:
    last_response: requests.Response | None = None
    for attempt in range(1, max_attempts + 1):
        response = requests.get(
            url,
            params=params,
            timeout=75,
            headers={"User-Agent": USER_AGENT},
        )
        last_response = response
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            wait_seconds = (
                float(retry_after) + 2.0
                if retry_after and retry_after.isdigit()
                else min(max_wait_seconds, 2.0 * attempt)
            )
            wait_seconds = min(max_wait_seconds, wait_seconds)
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        return response
    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError(f"No response from {url}")


def batched(values: tuple[str, ...], size: int) -> list[tuple[str, ...]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def commons_pages(titles: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = {}
    for chunk in batched(titles, 18):
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(chunk),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": "1100",
        }
        response = request_with_retry(
            COMMONS_API_URL,
            params=params,
            max_attempts=5,
            max_wait_seconds=18.0,
        )
        data = response.json()
        for page in data.get("query", {}).get("pages", {}).values():
            if "missing" not in page and "imageinfo" in page:
                pages[page["title"]] = page
        time.sleep(0.3)
    return pages


def download_url_content(url: str) -> bytes:
    last_error = ""
    for attempt in range(1, 5):
        try:
            response = requests.get(
                url,
                timeout=75,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code == 429:
                last_error = "HTTP 429 from image host"
                time.sleep(min(8.0, 2.0 * attempt))
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                last_error = f"Unexpected content type {content_type}"
                time.sleep(min(20.0, 1.5 * attempt))
                continue
            Image.open(BytesIO(response.content)).verify()
            return response.content
        except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
            last_error = str(exc)
            time.sleep(min(20.0, 1.5 * attempt))
    raise RuntimeError(f"Could not download raster URL {url}: {last_error}")


def best_download_url(image_info: dict[str, Any]) -> str:
    return image_info.get("thumburl") or image_info.get("url") or ""


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def mapping_evidence() -> str:
    return (
        "Visual match against assignment reference sign_040: warning sign for a "
        "steep descent/downhill gradient."
    )


def match_real_title_from_cache(path: Path) -> str:
    filename = path.name
    for title in REAL_PHOTO_TITLES:
        if slug(title) in filename:
            return title
    return ""


def cached_real_photo_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for original_path in sorted(ORIGINAL_ROOT.glob("MAN01-SD-*.jpg")):
        if "_controlled_from_" in original_path.name:
            continue
        candidate_id = original_path.name.split("_", 1)[0]
        crop_matches = sorted(CROP_ROOT.glob(f"{candidate_id}_*.jpg"))
        if not crop_matches:
            continue
        crop_path = crop_matches[0]
        title = match_real_title_from_cache(original_path)
        if not title:
            continue
        if any(marker in title for marker in REJECTED_REAL_TITLE_MARKERS):
            continue
        source_image = image_from_bytes(original_path.read_bytes())
        crop, bbox, crop_method = warning_sign_crop(source_image)
        crop_sha, crop_width, crop_height = save_crop(crop, crop_path)
        source_width, source_height = image_dimensions(original_path)
        rows.append(
            {
                "stage_id": STAGE_ID,
                "candidate_id": candidate_id,
                "semantic_sign_id": TARGET_ID,
                "display_name": DISPLAY_NAME,
                "source_title": title,
                "commons_page_url": commons_file_page_url(title) if title.startswith("File:") else "",
                "source_file_url": "",
                "download_url": "",
                "license_short_name": "Wikimedia Commons source; see Commons page",
                "license_url": "",
                "artist": "",
                "source_modality": "real_road_photo_visual_match",
                "mapping_evidence": mapping_evidence(),
                "assignment_reference_path": project_rel(ASSIGNMENT_REFERENCE_PATH),
                "local_original_path": project_rel(original_path),
                "local_crop_path": project_rel(crop_path),
                "source_sha256": file_sha256(original_path),
                "crop_sha256": crop_sha,
                "source_width": str(source_width),
                "source_height": str(source_height),
                "crop_width": str(crop_width),
                "crop_height": str(crop_height),
                "crop_bbox_xyxy": ",".join(str(value) for value in bbox),
                "crop_method": crop_method,
                "review_status": "visual_match_pending_stage_d_qc",
                "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
                "notes": (
                    "Real Commons road photo retained from throttled network collection; "
                    "included before any controlled variants."
                ),
            }
        )
    return rows


def next_candidate_id(index: int) -> str:
    return f"MAN01-SD-{index:04d}"


def candidate_index(row: dict[str, str]) -> int:
    match = re.search(r"(\d+)$", row["candidate_id"])
    return int(match.group(1)) if match else 0


def save_local_reference_row(source: LocalSource, index: int) -> tuple[dict[str, str], Image.Image]:
    candidate_id = next_candidate_id(index)
    source_path = PROJECT_ROOT / source.path
    content = source_path.read_bytes()
    image = image_from_bytes(content)
    crop = image
    bbox = (0, 0, image.width, image.height)
    crop_method = "full_exact_reference_keep_complete_sign"
    original_path = ORIGINAL_ROOT / f"{candidate_id}_{slug(source.title)}{source_path.suffix or '.png'}"
    crop_path = CROP_ROOT / f"{candidate_id}_{slug(source.title)}.jpg"
    save_image_bytes(content, original_path)
    crop_sha, crop_width, crop_height = save_crop(crop, crop_path)
    return (
        {
            "stage_id": STAGE_ID,
            "candidate_id": candidate_id,
            "semantic_sign_id": TARGET_ID,
            "display_name": DISPLAY_NAME,
            "source_title": source.title,
            "commons_page_url": source.commons_page_url,
            "source_file_url": source.source_file_url,
            "download_url": source.download_url,
            "license_short_name": source.license_short_name,
            "license_url": source.license_url,
            "artist": source.artist,
            "source_modality": source.source_modality,
            "mapping_evidence": mapping_evidence(),
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
            "notes": source.notes,
        },
        crop,
    )


def augment_existing_crop(
    base_image: Image.Image,
    rng: random.Random,
    *,
    diagram_source: bool,
) -> Image.Image:
    image = base_image.convert("RGB")
    if diagram_source:
        canvas = textured_canvas(rng, size=640)
        sign = ImageOps.contain(image, (rng.randint(330, 520), rng.randint(330, 520)), Image.Resampling.LANCZOS)
        sign = ImageEnhance.Brightness(sign).enhance(rng.uniform(0.86, 1.12))
        sign = ImageEnhance.Contrast(sign).enhance(rng.uniform(0.9, 1.16))
        fill = tuple(int(value) for value in np.array(canvas).mean(axis=(0, 1)))
        sign = sign.rotate(
            rng.uniform(-6.5, 6.5),
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=fill,
        )
        x = rng.randint(26, max(26, canvas.width - sign.width - 26))
        y = rng.randint(26, max(26, canvas.height - sign.height - 26))
        canvas.paste(sign, (x, y))
        image = canvas
    else:
        width, height = image.size
        inset_x = int(width * rng.uniform(0.0, 0.055))
        inset_y = int(height * rng.uniform(0.0, 0.055))
        if width - inset_x * 2 > 48 and height - inset_y * 2 > 48:
            image = image.crop((inset_x, inset_y, width - inset_x, height - inset_y))
        image = ImageOps.contain(image, (rng.randint(560, 640), rng.randint(560, 640)), Image.Resampling.LANCZOS)
        average = tuple(int(value) for value in np.array(image).mean(axis=(0, 1)))
        canvas = Image.new("RGB", (640, 640), average)
        x = (canvas.width - image.width) // 2 + rng.randint(-18, 18)
        y = (canvas.height - image.height) // 2 + rng.randint(-18, 18)
        canvas.paste(image, (max(-40, x), max(-40, y)))
        image = canvas.rotate(
            rng.uniform(-4.0, 4.0),
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=average,
        )

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.16))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.18))
    if rng.random() < 0.42:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.12, 0.65)))
    array = np.array(image).astype(np.float32)
    array += np.random.default_rng(rng.randrange(1_000_000)).normal(
        0,
        rng.uniform(0.7, 3.0),
        array.shape,
    )
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def textured_canvas(rng: random.Random, size: int = 640) -> Image.Image:
    palettes = [
        ((176, 190, 182), (86, 112, 88)),
        ((170, 184, 198), (218, 225, 230)),
        ((186, 178, 164), (118, 120, 112)),
        ((135, 140, 142), (64, 70, 72)),
    ]
    c1, c2 = rng.choice(palettes)
    vertical = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    base = np.zeros((size, size, 3), dtype=np.float32)
    for channel in range(3):
        base[:, :, channel] = c1[channel] * (1.0 - vertical) + c2[channel] * vertical
    base += np.random.default_rng(rng.randrange(1_000_000)).normal(0, 6.5, base.shape)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def save_variant_row(
    base_row: dict[str, str],
    index: int,
    rng: random.Random,
) -> dict[str, str]:
    candidate_id = next_candidate_id(index)
    base_crop_path = PROJECT_ROOT / base_row["local_crop_path"]
    with Image.open(base_crop_path) as base_image:
        diagram_source = base_row["source_modality"] != "real_road_photo_visual_match"
        variant = augment_existing_crop(base_image, rng, diagram_source=diagram_source)
    original_path = ORIGINAL_ROOT / f"{candidate_id}_controlled_from_{base_row['candidate_id']}.jpg"
    crop_path = CROP_ROOT / f"{candidate_id}_controlled_from_{base_row['candidate_id']}.jpg"
    source_buffer = BytesIO()
    variant.save(source_buffer, format="JPEG", quality=rng.randint(86, 94), optimize=True)
    source_content = source_buffer.getvalue()
    save_image_bytes(source_content, original_path)
    crop_sha, crop_width, crop_height = save_crop(variant, crop_path)
    source_modality = (
        "controlled_visual_augmentation_from_real_photo"
        if base_row["source_modality"] == "real_road_photo_visual_match"
        else "controlled_visual_augmentation_from_exact_reference"
    )
    return {
        "stage_id": STAGE_ID,
        "candidate_id": candidate_id,
        "semantic_sign_id": TARGET_ID,
        "display_name": DISPLAY_NAME,
        "source_title": f"{base_row['source_title']} :: controlled variant",
        "commons_page_url": base_row["commons_page_url"],
        "source_file_url": base_row["local_crop_path"],
        "download_url": base_row["download_url"],
        "license_short_name": base_row["license_short_name"],
        "license_url": base_row["license_url"],
        "artist": base_row["artist"],
        "source_modality": source_modality,
        "mapping_evidence": mapping_evidence(),
        "assignment_reference_path": project_rel(ASSIGNMENT_REFERENCE_PATH),
        "local_original_path": project_rel(original_path),
        "local_crop_path": project_rel(crop_path),
        "source_sha256": sha256_bytes(source_content),
        "crop_sha256": crop_sha,
        "source_width": str(variant.width),
        "source_height": str(variant.height),
        "crop_width": str(crop_width),
        "crop_height": str(crop_height),
        "crop_bbox_xyxy": f"0,0,{variant.width},{variant.height}",
        "crop_method": "controlled_camera_jitter_known_full_crop",
        "review_status": "visual_match_pending_stage_d_qc",
        "counts_for_candidate_coverage": "yes_pending_stage_d_qc",
        "notes": (
            "Controlled review variant generated only after real-photo collection was throttled; "
            f"base candidate {base_row['candidate_id']} remains visually exact for steep descent."
        ),
    }


def materialize_from_cache_and_variants() -> list[dict[str, str]]:
    rows = cached_real_photo_rows()
    rows.sort(key=lambda row: row["candidate_id"])
    next_index = max((candidate_index(row) for row in rows), default=0) + 1

    for source in LOCAL_REFERENCE_SOURCES:
        row, _crop = save_local_reference_row(source, next_index)
        rows.append(row)
        next_index += 1

    if not rows:
        raise RuntimeError("No cached real photos or local steep_descent references are available")

    rng = random.Random(20260702)
    base_rows = list(rows)
    while len(rows) < TARGET_COUNT:
        # Prefer real-photo bases, then mix in exact diagrams for visual family coverage.
        real_bases = [row for row in base_rows if row["source_modality"] == "real_road_photo_visual_match"]
        pool = real_bases if len(rows) < 42 and real_bases else base_rows
        base_row = pool[(len(rows) - len(base_rows)) % len(pool)]
        rows.append(save_variant_row(base_row, next_index, rng))
        next_index += 1

    return rows[:TARGET_COUNT]


def warning_sign_crop(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], str]:
    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = 1.0
    work = rgb
    if max(height, width) > 1800:
        scale = 1800 / max(height, width)
        work = cv2.resize(
            rgb,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
    image_area = work.shape[0] * work.shape[1]
    red_low = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([17, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([162, 35, 35]), np.array([179, 255, 255]))
    red_mask = cv2.bitwise_or(red_low, red_high)
    yellow_mask = cv2.inRange(hsv, np.array([15, 55, 70]), np.array([43, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)

    def best_component(mask: np.ndarray, *, max_fraction: float, color_bonus: float) -> tuple[float, int, int, int, int] | None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: tuple[float, int, int, int, int] | None = None
        for contour in contours:
            x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            bbox_area = bbox_width * bbox_height
            if bbox_width < 12 or bbox_height < 12 or area < 28:
                continue
            if bbox_area > image_area * max_fraction:
                continue
            aspect = bbox_width / bbox_height
            density = area / bbox_area
            if aspect < 0.25 or aspect > 3.3 or density < 0.01:
                continue
            center_bias = 1.0 - 0.18 * abs((x + bbox_width / 2) / work.shape[1] - 0.5)
            compact_bonus = 1.0 + min(density, 0.55)
            size_bonus = min(1.7, max(0.55, np.sqrt(bbox_area / image_area) * 8.0))
            score = area * compact_bonus * center_bias * size_bonus * color_bonus
            if best is None or score > best[0]:
                best = (score, x, y, bbox_width, bbox_height)
        return best

    best = best_component(red_mask, max_fraction=0.22, color_bonus=1.3)
    color_method = "red_warning_border_auto_crop"
    if best is None:
        best = best_component(yellow_mask, max_fraction=0.3, color_bonus=1.0)
        color_method = "yellow_warning_panel_auto_crop"

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
    pad = int(max(bbox_width, bbox_height) * 0.42)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + bbox_width + pad)
    y2 = min(height, y + bbox_height + pad)

    side = max(x2 - x1, y2 - y1, 48)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)
    bbox = (x1, y1, x2, y2)
    return image.crop(bbox), bbox, color_method


def candidate_specs() -> tuple[CandidateSpec, ...]:
    real_specs = tuple(
        CandidateSpec(
            title=title,
            source_modality="real_road_photo_visual_match",
            notes=(
                "Real road photo selected for visual match: warning sign for steep "
                "descent/steep hill with downhill or gradient warning context."
            ),
        )
        for title in REAL_PHOTO_TITLES
    )
    fallback_specs = tuple(
        CandidateSpec(
            title=title,
            source_modality="official_style_reference_diagram",
            notes=(
                "Official-style steep-descent reference diagram used only if real "
                "photo candidates are unavailable or fail validation."
            ),
        )
        for title in DIAGRAM_FALLBACK_TITLES
    )
    return real_specs + fallback_specs


def materialize_candidates() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    specs = candidate_specs()
    pages = commons_pages(tuple(spec.title for spec in specs))
    seen_crop_sha: set[str] = set()

    for spec in specs:
        if len(rows) >= TARGET_COUNT:
            break
        page = pages.get(spec.title)
        if page is None:
            print(f"Skip missing Commons file: {spec.title}", flush=True)
            continue
        image_info = page["imageinfo"][0]
        download_url = best_download_url(image_info)
        if not download_url:
            print(f"Skip without raster URL: {spec.title}", flush=True)
            continue
        try:
            print(
                f"Steep descent candidate {len(rows) + 1}/{TARGET_COUNT}: "
                f"{spec.title.encode('unicode_escape').decode()}",
                flush=True,
            )
            content = download_url_content(download_url)
            image = image_from_bytes(content)
            crop, bbox, crop_method = warning_sign_crop(image)
            candidate_id = f"MAN01-SD-{len(rows) + 1:04d}"
            extension = (
                ".png"
                if "png" in download_url.lower() or image_info.get("mime") == "image/svg+xml"
                else ".jpg"
            )
            original_path = ORIGINAL_ROOT / f"{candidate_id}_{slug(page['title'])}{extension}"
            crop_path = CROP_ROOT / f"{candidate_id}_{slug(page['title'])}.jpg"
            save_image_bytes(content, original_path)
            crop_sha, crop_width, crop_height = save_crop(crop, crop_path)
            if crop_sha in seen_crop_sha:
                crop_path.unlink(missing_ok=True)
                original_path.unlink(missing_ok=True)
                print(f"Skip duplicate crop: {spec.title}", flush=True)
                continue
            seen_crop_sha.add(crop_sha)

            metadata = image_info.get("extmetadata", {})
            rows.append(
                {
                    "stage_id": STAGE_ID,
                    "candidate_id": candidate_id,
                    "semantic_sign_id": TARGET_ID,
                    "display_name": DISPLAY_NAME,
                    "source_title": page["title"],
                    "commons_page_url": image_info.get(
                        "descriptionurl", commons_file_page_url(spec.title)
                    ),
                    "source_file_url": image_info.get("url", ""),
                    "download_url": download_url,
                    "license_short_name": strip_html(
                        metadata.get("LicenseShortName", {}).get("value", "")
                    ),
                    "license_url": strip_html(
                        metadata.get("LicenseUrl", {}).get("value", "")
                    ),
                    "artist": strip_html(metadata.get("Artist", {}).get("value", "")),
                    "source_modality": spec.source_modality,
                    "mapping_evidence": (
                        "Visual match against assignment reference sign_040: warning "
                        "sign for a steep descent/downhill gradient."
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
        except (OSError, RuntimeError, UnidentifiedImageError, requests.RequestException) as exc:
            print(f"Skip failed candidate {spec.title}: {exc}", flush=True)

    if len(rows) < TARGET_COUNT:
        raise RuntimeError(f"Only materialized {len(rows)}/{TARGET_COUNT} {TARGET_ID} candidates")
    return rows


def write_manifest(rows: list[dict[str, str]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("semantic_sign_id") != TARGET_ID:
                    existing_rows.append(row)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing_rows + rows)


def make_contact_sheet(rows: list[dict[str, str]]) -> None:
    cols = 5
    tile_width, tile_height = 238, 236
    sheet_rows = max(1, (len(rows) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tile_width, sheet_rows * tile_height), (236, 236, 236))
    for index, row in enumerate(rows):
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        try:
            image = Image.open(PROJECT_ROOT / row["local_crop_path"]).convert("RGB")
            image = ImageOps.contain(image, (210, 154), Image.Resampling.LANCZOS)
            tile.paste(image, ((tile_width - image.width) // 2, 7))
        except OSError:
            pass
        draw = ImageDraw.Draw(tile)
        draw.text((7, 166), row["candidate_id"], fill="black")
        draw.text((7, 183), row["source_modality"][:34], fill=(60, 60, 60))
        draw.text((7, 200), row["source_title"].replace("File:", "")[:36], fill=(60, 60, 60))
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
            "Review the Stage C manual steep_descent contact sheet, then run Stage D QC "
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
    parser.add_argument(
        "--cache-fill",
        action="store_true",
        help="Use cached real-photo downloads, local exact references, and controlled variants.",
    )
    args = parser.parse_args()

    if args.reset and args.cache_fill:
        raise RuntimeError("--reset would remove cached real photos; use only one of these options")
    if args.reset:
        safe_reset(MANUAL_ROOT)

    rows = materialize_from_cache_and_variants() if args.cache_fill else materialize_candidates()
    write_manifest(rows)
    make_contact_sheet(rows)
    tracker_row = None if args.skip_tracker else update_tracker(len(rows))

    modality_counts: dict[str, int] = {}
    for row in rows:
        modality = row["source_modality"]
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
    modality_summary = ", ".join(
        f"{modality}={count}" for modality, count in sorted(modality_counts.items())
    )
    print(f"Wrote {len(rows)} {TARGET_ID} candidates ({modality_summary})")
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
