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
TARGET_ID = "residential_area_ahead"
DISPLAY_NAME = "Residential area ahead"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
COMMONS_CATEGORY = "Category:Road sign \u8b66-20 (China)"
USER_AGENT = "MiniProjectRoadSignResearch/1.0 (academic coursework data provenance)"

MANUAL_ROOT = PROJECT_ROOT / "data/raw/manual_collection/stage_c_manual_01/residential_area_ahead"
ORIGINAL_ROOT = MANUAL_ROOT / "original"
CROP_ROOT = MANUAL_ROOT / "crops"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_manual_01_candidates.csv"
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
CONTACT_SHEET_PATH = (
    PROJECT_ROOT
    / "outputs/review/00_CURRENT_REVIEW/stage_c_manual_residential_area_ahead_contact_sheet.jpg"
)
ASSIGNMENT_REFERENCE_PATH = PROJECT_ROOT / "data/official/assignment_images/Yellow Signs/045_1_0004.png"


@dataclass(frozen=True)
class ExternalImageSpec:
    source_title: str
    source_page_url: str
    image_url: str
    local_slug: str
    source_modality: str
    license_note: str
    notes: str
    crop_box_xyxy: tuple[int, int, int, int] | None = None


LOCAL_COMMONS_SPECS: tuple[tuple[int, str], ...] = (
    (1, "File:CN road sign \u8b66 23.svg"),
    (2, "File:104\u56fd\u9053 - National Highway 104 - 2015.08 - panoramio (1).jpg"),
    (3, "File:201812 Village Sign on Anshan Road near Anyuan Village.jpg"),
    (5, "File:Changping, Beijing, China - panoramio (155).jpg"),
    (6, "File:Dajiaoyu Village (20230125150536).jpg"),
    (7, "File:East Liangcun Village 01.jpg"),
    (8, "File:End of musical road at G108 Xiayunling section (20230122123921).jpg"),
    (9, "File:Mengzhai Village 01.jpg"),
    (10, "File:Village road sign beside road in China.jpg"),
    (11, "File:Xiaoniu Village 02.jpg"),
    (13, "File:\u4e1c\u6881\u6751\u897f\u6751\u53e3.jpg"),
    (14, "File:\u5c0f\u725b\u6751.jpg"),
    (15, "File:\u5c0f\u725b\u6751\u6751\u53e3 2020-04.jpg"),
    (16, "File:\u5c0f\u725b\u6751\u6751\u53e3.jpg"),
    (17, "File:\u5dde\u57ce\u8857\u9053\u8363\u82b1\u6811\u6751.jpg"),
    (18, "File:\u5f20\u5e84 02.jpg"),
    (19, "File:\u6881\u6751.jpg"),
    (20, "File:\u77f3\u9a6c\u6751\u9644\u8fd1\u7684\u6c49\u5893.jpg"),
    (21, "File:\u8001\u6caa\u676d\u516c\u8def Zhejiang - panoramio.jpg"),
)

EXTERNAL_IMAGES: tuple[ExternalImageSpec, ...] = (
    ExternalImageSpec(
        "ExcuseMyEnglish Mu Shan roadside village warning sign",
        "https://excusemyenglish.fr/Images/ImagesChine/Panneaux/InSitu/Content/Panneaux9.htm",
        "https://excusemyenglish.fr/Images/ImagesChine/Panneaux/InSitu/MaxiLiugong31a.jpg",
        "excusemyenglish_mu_shan",
        "real_road_photo_visual_match",
        "Public web page; license not stated. Keep for local academic candidate review only.",
        "Roadside China village warning sign above blue village name plate.",
    ),
    ExternalImageSpec(
        "Sohu Pasha Old Village roadside village warning sign",
        "https://www.sohu.com/a/419865166_120865726",
        "http://p0.itc.cn/images01/20200921/8a2766b8b8504d5085bb7489f6e7eb61.png",
        "sohu_pasha_old_village",
        "real_road_photo_visual_match",
        "Public web page; license not stated. Keep for local academic candidate review only.",
        "Roadside village warning sign above a blue Pasha Old Village sign.",
    ),
    ExternalImageSpec(
        "360 Baike Shuangtangjian village roadside sign",
        "https://baike.so.com/gallery/list?eid=9876047&ghid=first&pic_idx=1&sid=10223130",
        "https://so1.360tres.com/t01384a89c45000d1c5.jpg",
        "baike360_shuangtangjian",
        "real_road_photo_visual_match",
        "Public web page; license not stated. Keep for local academic candidate review only.",
        "Rural road photo with the China village warning sign beside a village name plate.",
    ),
    ExternalImageSpec(
        "Made-in-China Xinlu reflective village warning sign product photo",
        "https://es.made-in-china.com/co_xinluwiremesh/product_60X60cm-Reflective-Road-Traffic-Sign-For-Road-Safety-With-Aluminum-Plate_hrrryyeey.html",
        "https://image.made-in-china.com/202f0j00OJpBYbNqlckt/60X60cm-Reflective-Road-Traffic-Sign-for-Road-Safety-with-Aluminum-Plate.webp",
        "made_in_china_xinlu_product",
        "commercial_product_photo_visual_match",
        "Commercial product page preview; license not stated. Keep for local academic candidate review only.",
        "Physical reflective plate using the village/residential warning pictogram.",
    ),
    ExternalImageSpec(
        "Made-in-China RunKun multi-sign product photo foreground village sign",
        "https://runkunaluminum.en.made-in-china.com/product/XwbGOhAMGdYN/China-Aluminum-Road-Signs-Manufacturers-for-Safety-Warning-Signal-Board-Reflective-Traffic-Sign.html",
        "https://image.made-in-china.com/2f0j00rEakbQocFHgf/Aluminum-Road-Signs-Manufacturers-for-Safety-Warning-Signal-Board-Reflective-Traffic-Sign.jpg",
        "made_in_china_runkun_multi_foreground",
        "commercial_product_photo_visual_match",
        "Commercial product page preview; license not stated. Keep for local academic candidate review only.",
        "Foreground physical plate in a multi-sign product photo.",
        (190, 260, 580, 690),
    ),
    ExternalImageSpec(
        "NiPic village warning sign diagram",
        "https://www.nipic.com/show/38798795.html",
        "https://pic.nximg.cn/file/20220315/33650540_141740779109_2.jpg",
        "nipic_village_warning_diagram",
        "public_web_reference_diagram",
        "Stock/design page preview; license requires site terms. Keep for local academic candidate review only.",
        "Clear village warning sign diagram with the house/tree pictogram.",
    ),
    ExternalImageSpec(
        "NiPic village traffic warning sign diagram",
        "https://www.nipic.com/show/38751744.html",
        "https://pic.nximg.cn/file/20220311/31359956_155512809100_2.jpg",
        "nipic_village_traffic_diagram",
        "public_web_reference_diagram",
        "Stock/design page preview; license requires site terms. Keep for local academic candidate review only.",
        "Village warning sign diagram with Chinese text below the sign.",
    ),
    ExternalImageSpec(
        "NiPic village traffic identifier diagram",
        "https://www.nipic.com/show/24816143.html",
        "https://pic.nximg.cn/file/20190711/29541948_154656785000_2.jpg",
        "nipic_village_identifier_diagram",
        "public_web_reference_diagram",
        "Stock/design page preview; license requires site terms. Keep for local academic candidate review only.",
        "Clean village warning sign diagram with the same house/tree pictogram.",
    ),
    ExternalImageSpec(
        "GMW village entrance road warning sign",
        "https://difang.gmw.cn/nmg/2026-01/17/content_38542362.htm",
        "https://imgdifang.gmw.cn/attachement/jpg/site2/20260117/f44d305ea12e2ad53f8815.jpg",
        "gmw_village_entrance",
        "real_road_photo_visual_match",
        "Public news page; license not stated. Keep for local academic candidate review only.",
        "Real roadside village-warning sign on a blue village name plate.",
    ),
    ExternalImageSpec(
        "Sina Gengzhuang village road sign",
        "https://k.sina.com.cn/article_2920531853_ae13c78d00100dvyx.html",
        "http://n.sinaimg.cn/front/40/w480h360/20180914/pwv7-hkahyhw9954469.jpg",
        "sina_gengzhuang_village",
        "real_road_photo_visual_match",
        "Public news page; license not stated. Keep for local academic candidate review only.",
        "Real overhead roadside sign with the village-warning icon inside a blue village plate.",
    ),
    ExternalImageSpec(
        "QDZY product photo village warning sign 01",
        "http://www.qdzyjtgc.com/ddbzp/339.html",
        "http://www.qdzyjtgc.com/uploads/allimg/190706/1-1ZF61G1220-L.jpg",
        "qdzy_product_01",
        "commercial_product_photo_visual_match",
        "Commercial product page preview; license not stated. Keep for local academic candidate review only.",
        "Physical village-warning triangular sign product photo.",
    ),
    ExternalImageSpec(
        "QDZY product photo village warning sign 02",
        "http://www.qdzyjtgc.com/ddbzp/339.html",
        "http://www.qdzyjtgc.com/uploads/allimg/190706/1-1ZF61G135353.jpg",
        "qdzy_product_02",
        "commercial_product_photo_visual_match",
        "Commercial product page preview; license not stated. Keep for local academic candidate review only.",
        "Second physical village-warning triangular sign product photo.",
    ),
    ExternalImageSpec(
        "TT100K w3 legend residential/village warning icon",
        "https://lijiancheng0614.github.io/2019/04/16/2019_04_16_TT100K/",
        "data/raw/online_sources/tt100k_legend_icons/w3.png",
        "tt100k_w3_legend_icon",
        "dataset_legend_reference_icon",
        "TT100K legend reference icon already stored locally; original dataset use is academic/non-commercial.",
        "Exact w3 legend icon for the village/residential-area-ahead warning class.",
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


def commons_category_titles() -> list[str]:
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": COMMONS_CATEGORY,
        "cmtype": "file",
        "cmlimit": "100",
    }
    response = requests.get(
        COMMONS_API_URL,
        params=params,
        timeout=45,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    titles = [
        item["title"]
        for item in response.json().get("query", {}).get("categorymembers", [])
    ]
    if len(titles) != 21:
        raise RuntimeError(f"Expected 21 Commons category files, got {len(titles)}")
    return titles


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
        if image_info.get("mime") == "image/svg+xml":
            urls = [image_info.get("thumburl")]
        else:
            urls = [image_info.get("url") or image_info.get("thumburl")]
        for url in [candidate for candidate in urls if candidate]:
            if image_info.get("mime") == "image/svg+xml" and url == image_info.get("url"):
                continue
            try:
                response: requests.Response | None = None
                for attempt in range(1, 7):
                    response = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
                    if response.status_code != 429:
                        break
                    last_error = "HTTP 429 from upload.wikimedia.org"
                    retry_after = response.headers.get("retry-after")
                    if retry_after and retry_after.isdigit():
                        wait_seconds = min(660.0, float(retry_after) + 2.0)
                    else:
                        wait_seconds = min(60.0, 5.0 * attempt)
                    print(
                        f"Commons rate limit for {title.encode('unicode_escape').decode()}; "
                        f"waiting {wait_seconds:.0f}s",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
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


def download_external_image(spec: ExternalImageSpec) -> bytes:
    if not spec.image_url.startswith(("http://", "https://")):
        local_path = PROJECT_ROOT / spec.image_url
        if not local_path.exists():
            raise FileNotFoundError(f"Local source image is missing: {local_path}")
        return local_path.read_bytes()

    headers = {"User-Agent": USER_AGENT, "Referer": spec.source_page_url}
    last_error = ""
    for attempt in range(1, 5):
        try:
            response = requests.get(spec.image_url, headers=headers, timeout=45)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise RuntimeError(f"Unexpected content type {content_type}")
            Image.open(BytesIO(response.content)).verify()
            return response.content
        except (requests.RequestException, RuntimeError, UnidentifiedImageError, OSError) as exc:
            last_error = str(exc)
            time.sleep(min(10.0, 1.5 * attempt))
    raise RuntimeError(f"Could not download {spec.source_title}: {last_error}")


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


def yellow_warning_crop(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], str]:
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
    yellow = cv2.inRange(hsv, np.array([12, 45, 55]), np.array([48, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, int, int, int, int] | None = None
    image_area = work.shape[0] * work.shape[1]
    for contour in contours:
        x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if bbox_width < 18 or bbox_height < 18 or area < 80:
            continue
        aspect = bbox_width / bbox_height
        density = area / (bbox_width * bbox_height)
        if aspect < 0.45 or aspect > 1.9 or density < 0.08:
            continue
        score = area * (0.8 + min(density, 0.75))
        if bbox_width * bbox_height > image_area * 0.75:
            score *= 0.35
        if best is None or score > best[0]:
            best = (score, x, y, bbox_width, bbox_height)

    if best is None:
        side = min(width, height)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        bbox = (x1, y1, x1 + side, y1 + side)
        return image.crop(bbox), bbox, "center_fallback_no_yellow_warning_component"

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
    return image.crop(bbox), bbox, "yellow_warning_sign_auto_crop"


def crop_external_image(
    image: Image.Image,
    spec: ExternalImageSpec,
) -> tuple[Image.Image, tuple[int, int, int, int], str]:
    if spec.crop_box_xyxy is None:
        return yellow_warning_crop(image)

    x1, y1, x2, y2 = spec.crop_box_xyxy
    x1 = max(0, min(image.width - 1, x1))
    y1 = max(0, min(image.height - 1, y1))
    x2 = max(x1 + 1, min(image.width, x2))
    y2 = max(y1 + 1, min(image.height, y2))
    region = image.crop((x1, y1, x2, y2))
    crop, inner_bbox, _ = yellow_warning_crop(region)
    ix1, iy1, ix2, iy2 = inner_bbox
    bbox = (x1 + ix1, y1 + iy1, x1 + ix2, y1 + iy2)
    return crop, bbox, "manual_region_then_yellow_warning_auto_crop"


def commons_source_modality(title: str) -> str:
    lower = title.lower()
    if lower.endswith((".jpg", ".jpeg", ".png")) and "road sign" not in lower:
        return "real_road_photo_visual_match"
    if lower.endswith((".jpg", ".jpeg")) and "village" in lower:
        return "real_road_photo_visual_match"
    return "official_style_reference_diagram"


def materialize_commons_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for offset, (index, title) in enumerate(LOCAL_COMMONS_SPECS):
        print(
            f"Commons {offset + 1}/{len(LOCAL_COMMONS_SPECS)}: "
            f"{title.encode('unicode_escape').decode()}",
            flush=True,
        )
        page = commons_image_info(title, 1000)
        image_info = page["imageinfo"][0]
        download_url = image_info.get("thumburl") or image_info.get("url", "")
        candidate_id = f"MAN01-RAA-{index:04d}"
        extension = (
            ".png"
            if "png" in download_url.lower() or image_info.get("mime") == "image/svg+xml"
            else ".jpg"
        )
        original_path = ORIGINAL_ROOT / f"{candidate_id}_{slug(page['title'])}{extension}"
        crop_path = CROP_ROOT / f"{candidate_id}_{slug(page['title'])}.jpg"
        if original_path.exists() and original_path.stat().st_size > 0:
            content = original_path.read_bytes()
            download_url = image_info.get("url", download_url)
        else:
            page, image_info, download_url, content = download_commons_raster(title)
        image = image_from_bytes(content)
        crop, bbox, crop_method = yellow_warning_crop(image)
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
                "source_modality": commons_source_modality(page["title"]),
                "mapping_evidence": (
                    "Visual match against assignment reference sign_045: yellow triangular "
                    "China/GB village or residential-area warning with house/tree pictogram."
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
                "notes": "Exact China/GB village-warning category or close official reference.",
            }
        )
        time.sleep(0.2)
    return rows


def materialize_external_rows(start_index: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for offset, spec in enumerate(EXTERNAL_IMAGES):
        index = start_index + offset
        print(
            f"External {offset + 1}/{len(EXTERNAL_IMAGES)}: {spec.source_title}",
            flush=True,
        )
        content = download_external_image(spec)
        image = image_from_bytes(content)
        crop, bbox, crop_method = crop_external_image(image, spec)
        candidate_id = f"MAN01-RAA-{index:04d}"
        extension = ".png" if "png" in spec.image_url.lower() else ".jpg"
        if "webp" in spec.image_url.lower():
            extension = ".webp"
        original_path = ORIGINAL_ROOT / f"{candidate_id}_{spec.local_slug}{extension}"
        crop_path = CROP_ROOT / f"{candidate_id}_{spec.local_slug}.jpg"
        save_image_bytes(content, original_path)
        crop_sha, crop_width, crop_height = save_crop(crop, crop_path)

        rows.append(
            {
                "stage_id": STAGE_ID,
                "candidate_id": candidate_id,
                "semantic_sign_id": TARGET_ID,
                "display_name": DISPLAY_NAME,
                "source_title": spec.source_title,
                "commons_page_url": spec.source_page_url,
                "source_file_url": spec.image_url,
                "download_url": spec.image_url,
                "license_short_name": "license_not_stated",
                "license_url": spec.source_page_url,
                "artist": "",
                "source_modality": spec.source_modality,
                "mapping_evidence": (
                    "Manual web visual match against assignment reference sign_045: "
                    "yellow triangular village/residential warning with house/tree pictogram."
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
                "notes": f"{spec.notes} {spec.license_note}",
            }
        )
        time.sleep(0.2)
    return rows


def materialize_candidates() -> list[dict[str, str]]:
    commons_rows = materialize_commons_rows()
    external_rows = materialize_external_rows(22)
    rows = commons_rows + external_rows
    if len(rows) != 32:
        raise RuntimeError(f"Expected 32 rows, got {len(rows)}")
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
    cols = 8
    tile_width, tile_height = 190, 215
    sheet_rows = max(1, (len(rows) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tile_width, sheet_rows * tile_height), (236, 236, 236))
    for index, row in enumerate(rows):
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        try:
            image = Image.open(PROJECT_ROOT / row["local_crop_path"]).convert("RGB")
            image = ImageOps.contain(image, (168, 138), Image.Resampling.LANCZOS)
            tile.paste(image, ((tile_width - image.width) // 2, 6))
        except OSError:
            pass
        draw = ImageDraw.Draw(tile)
        draw.text((6, 150), row["candidate_id"], fill="black")
        draw.text((6, 167), row["source_modality"][:29], fill=(60, 60, 60))
        draw.text((6, 184), row["source_title"].replace("File:", "")[:30], fill=(60, 60, 60))
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
            "Review the Stage C manual residential_area_ahead contact sheet, then run "
            "Stage D QC before Stage E split freeze."
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
