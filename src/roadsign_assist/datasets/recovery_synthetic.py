"""Phase B5 deterministic synthetic/composited presentation candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from roadsign_assist.datasets.recovery_acquisition import (
    DEFAULT_ACQUISITION_ROOT,
    DEFAULT_AUDIT_ROOT,
    check_disk_budget,
)
from roadsign_assist.paths import project_path

SYNTHETIC_VERSION = "recovery_presentation_synthetic_v1"
DEFAULT_BASE_MANIFEST = Path("data/manifests/classifier_production_78_v3_20260829.csv")
NEGATIVE_KINDS = ("icons_or_logos", "yellow_diamond", "unrelated_traffic_imagery")
DEVICES = ("simulated_phone", "simulated_tablet", "simulated_monitor")
DISTANCES = ("near", "medium", "far")
BRIGHTNESS_LEVELS = ("low", "medium", "high")
GLARE_LEVELS = ("none", "glare", "reflection")
BACKGROUNDS = ("room", "vehicle", "outdoor")
COMPOSITIONS = ("single_sign", "search_results", "multi_sign")
MATERIALS = ("screen", "matte", "glossy")


class BaseArtwork(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artwork_id: str
    semantic_sign_id: str
    path: str
    sha256: str
    source_group: str
    source_dataset: str
    source_url: str
    licence_status: str
    review_decision: str


class SyntheticRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    generated_at: datetime
    sample_id: str
    instance_id: str
    split: str = "train"
    expected_kind: str
    semantic_sign_id: str = ""
    capture_domain: str
    image_path: str
    image_sha256: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    bbox_x1: int = Field(ge=0)
    bbox_y1: int = Field(ge=0)
    bbox_x2: int = Field(ge=0)
    bbox_y2: int = Field(ge=0)
    area_ratio: float = Field(ge=0.0, le=1.0)
    size_bucket: str
    source_id: str = "deterministic_synthetic"
    source_artwork_id: str = ""
    source_artwork_sha256: str = ""
    source_artwork_group: str = ""
    base_rendering_id: str
    related_capture_group_id: str
    simulated_device_id: str
    screen_orientation: str
    orientation_degrees: int
    presentation_distance: str
    presentation_brightness: str
    presentation_glare: str
    presentation_background: str
    scene_composition: str
    presentation_material: str
    negative_kind: str = ""
    transform_seed: int
    generation_recipe: dict[str, object]
    annotation_status: str = "synthetic_exact_geometry_unreviewed"
    review_decision: str = "pending"
    licence_status: str = "derived_from_reviewed_internal_academic_source_pending_batch_review"
    permitted_use: str = "none_pending_review"
    collection_state: str = "quarantined"


def _now() -> datetime:
    return datetime.now(UTC)


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)) and value != "":
        return int(value)
    return 0


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    value: Any = rng.integers(low, high)
    return int(value)


def _load_bases(path: Path, *, per_class: int = 3) -> dict[str, list[BaseArtwork]]:
    selected: dict[str, dict[str, BaseArtwork]] = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "train" or row.get("review_decision") != "accept":
                continue
            label = row.get("semantic_sign_id", "").strip()
            image_path = project_path(row.get("dataset_image_path", ""))
            digest = row.get("crop_sha256", "").strip().lower()
            if not label or not digest or not image_path.is_file():
                continue
            selected[label].setdefault(
                digest,
                BaseArtwork(
                    artwork_id=row.get("sample_id", "") or f"sha256:{digest}",
                    semantic_sign_id=label,
                    path=str(image_path),
                    sha256=digest,
                    source_group=row.get("source_group", "") or f"sha256:{digest}",
                    source_dataset=row.get("source_dataset", ""),
                    source_url=row.get("source_url", ""),
                    licence_status=row.get("licence_status", "") or "unreviewed",
                    review_decision=row.get("review_decision", ""),
                ),
            )
    return {
        label: [values[key] for key in sorted(values)[:per_class]]
        for label, values in sorted(selected.items())
    }


def _road_surface(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = round(height * 0.43)
    sky_top = np.asarray((170, 120, 70), dtype=np.float32)
    sky_bottom = np.asarray((230, 200, 150), dtype=np.float32)
    for y in range(horizon):
        mix = y / max(1, horizon - 1)
        image[y, :] = (sky_top * (1 - mix) + sky_bottom * mix).astype(np.uint8)
    image[horizon:, :] = (55, 95, 70)
    road = np.array(
        [
            [round(width * 0.42), horizon],
            [round(width * 0.58), horizon],
            [round(width * 0.94), height],
            [round(width * 0.06), height],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, road, (72, 72, 76))
    for offset in (-0.14, 0.14):
        top_x = round(width * (0.5 + offset * 0.15))
        bottom_x = round(width * (0.5 + offset))
        cv2.line(image, (top_x, horizon), (bottom_x, height), (220, 220, 215), 3)
    for _ in range(18):
        x = int(rng.integers(0, width))
        y = int(rng.integers(max(0, horizon - 40), height))
        radius = int(rng.integers(3, 16))
        color = (int(rng.integers(35, 75)), int(rng.integers(75, 145)), int(rng.integers(25, 65)))
        cv2.circle(image, (x, y), radius, color, -1)
    noise = rng.normal(0, 4, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _surface(
    width: int,
    height: int,
    *,
    material: str,
    background: str,
    device: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = _road_surface(width, height, rng)
    if background == "room":
        canvas[:] = (112, 106, 98)
        for x in range(0, width, 48):
            cv2.line(canvas, (x, 0), (x, height), (120, 114, 106), 1)
    elif background == "vehicle":
        cv2.rectangle(canvas, (0, round(height * 0.72)), (width, height), (35, 35, 40), -1)
    margin_x = round(width * (0.08 if device == "simulated_monitor" else 0.14))
    margin_y = round(height * (0.08 if device == "simulated_phone" else 0.13))
    x1, y1, x2, y2 = margin_x, margin_y, width - margin_x, height - margin_y
    if material == "screen":
        cv2.rectangle(canvas, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (12, 12, 14), -1)
        screen = _road_surface(x2 - x1, y2 - y1, rng)
        canvas[y1:y2, x1:x2] = screen
        cv2.circle(canvas, ((x1 + x2) // 2, y2 + 5), 3, (75, 75, 75), -1)
    else:
        paper = 235 if material == "glossy" else 215
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (paper, paper, paper), -1)
        texture = rng.normal(0, 4 if material == "matte" else 1, (y2 - y1, x2 - x1, 1))
        canvas[y1:y2, x1:x2] = np.clip(
            canvas[y1:y2, x1:x2].astype(np.float32) + texture, 0, 255
        ).astype(np.uint8)
    return canvas, (x1, y1, x2, y2)


def _paste_rotated(
    canvas: np.ndarray,
    artwork: np.ndarray,
    *,
    center: tuple[int, int],
    target_side: int,
    angle: int,
) -> tuple[int, int, int, int]:
    source_height, source_width = artwork.shape[:2]
    scale = target_side / max(source_width, source_height)
    resized = cv2.resize(
        artwork,
        (max(8, round(source_width * scale)), max(8, round(source_height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    height, width = resized.shape[:2]
    diagonal = max(width, height) + 16
    patch = np.zeros((diagonal, diagonal, 3), dtype=np.uint8)
    mask = np.zeros((diagonal, diagonal), dtype=np.uint8)
    x0 = (diagonal - width) // 2
    y0 = (diagonal - height) // 2
    patch[y0 : y0 + height, x0 : x0 + width] = resized
    mask[y0 : y0 + height, x0 : x0 + width] = 255
    matrix = cv2.getRotationMatrix2D((diagonal / 2, diagonal / 2), angle, 1.0)
    rotated = cv2.warpAffine(patch, matrix, (diagonal, diagonal))
    rotated_mask = cv2.warpAffine(mask, matrix, (diagonal, diagonal))
    left = center[0] - diagonal // 2
    top = center[1] - diagonal // 2
    right = left + diagonal
    bottom = top + diagonal
    canvas_left, canvas_top = max(0, left), max(0, top)
    canvas_right, canvas_bottom = min(canvas.shape[1], right), min(canvas.shape[0], bottom)
    patch_left, patch_top = canvas_left - left, canvas_top - top
    patch_right = patch_left + canvas_right - canvas_left
    patch_bottom = patch_top + canvas_bottom - canvas_top
    region_mask = rotated_mask[patch_top:patch_bottom, patch_left:patch_right]
    region = canvas[canvas_top:canvas_bottom, canvas_left:canvas_right]
    source = rotated[patch_top:patch_bottom, patch_left:patch_right]
    region[region_mask > 0] = source[region_mask > 0]
    points: Any = cv2.findNonZero((rotated_mask > 0).astype(np.uint8))
    if points is None:
        return canvas_left, canvas_top, canvas_right, canvas_bottom
    bx, by, bw, bh = cv2.boundingRect(points)
    return (
        max(0, left + bx),
        max(0, top + by),
        min(canvas.shape[1], left + bx + bw),
        min(canvas.shape[0], top + by + bh),
    )


def _effects(
    image: np.ndarray,
    *,
    brightness: str,
    glare: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    factors = {"low": 0.58, "medium": 0.9, "high": 1.16}
    output = np.clip(image.astype(np.float32) * factors[brightness], 0, 255).astype(np.uint8)
    temperature = int(rng.integers(-18, 19))
    adjusted = output.astype(np.int16)
    adjusted[:, :, 0] -= temperature
    adjusted[:, :, 2] += temperature
    output = np.clip(adjusted, 0, 255).astype(np.uint8)
    weather = ("clear", "rain", "haze", "shadow")[seed % 4]
    if weather == "rain":
        for _ in range(45):
            x = _randint(rng, 0, output.shape[1])
            y = _randint(rng, 0, output.shape[0])
            cv2.line(output, (x, y), (x + 4, y + 14), (190, 190, 190), 1)
    elif weather == "haze":
        output = cv2.addWeighted(output, 0.62, np.full_like(output, 205), 0.38, 0)
    elif weather == "shadow":
        overlay = output.copy()
        points = np.array(
            [[0, output.shape[0] // 3], [output.shape[1], output.shape[0] // 2], [output.shape[1], output.shape[0]]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(overlay, points, (10, 10, 10))
        output = cv2.addWeighted(output, 0.78, overlay, 0.22, 0)
    if glare != "none":
        overlay = output.copy()
        center = (int(output.shape[1] * 0.72), int(output.shape[0] * 0.25))
        axes = (max(20, output.shape[1] // 5), max(12, output.shape[0] // 12))
        cv2.ellipse(overlay, center, axes, -25, 0, 360, (255, 255, 255), -1)
        output = cv2.addWeighted(output, 0.82, overlay, 0.18 if glare == "glare" else 0.10, 0)
    if seed % 3 == 0:
        for y in range(seed % 7, output.shape[0], 6):
            cv2.line(output, (0, y), (output.shape[1], y), (25, 25, 25), 1)
    blur_kernel = (1, 3, 5)[seed % 3]
    if blur_kernel > 1:
        output = cv2.GaussianBlur(output, (blur_kernel, blur_kernel), 0)
    return output, {
        "weather_effect": weather,
        "colour_temperature_shift": temperature,
        "moire_lines": seed % 3 == 0,
        "blur_kernel": blur_kernel,
        "jpeg_quality": 72 + seed % 23,
    }


def _negative_content(
    canvas: np.ndarray,
    surface: tuple[int, int, int, int],
    *,
    kind: str,
    rng: np.random.Generator,
) -> None:
    x1, y1, x2, y2 = surface
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    if kind == "icons_or_logos":
        for index, color in enumerate(((220, 80, 40), (45, 180, 220), (80, 210, 80))):
            offset = (index - 1) * max(35, (x2 - x1) // 5)
            cv2.circle(canvas, (center[0] + offset, center[1]), 28, color, -1)
        cv2.putText(canvas, "APP", (center[0] - 45, center[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (245, 245, 245), 2)
    elif kind == "yellow_diamond":
        side = max(30, min(x2 - x1, y2 - y1) // 5)
        points = np.array(
            [[center[0], center[1] - side], [center[0] + side, center[1]], [center[0], center[1] + side], [center[0] - side, center[1]]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(canvas, points, (20, 220, 240))
        cv2.polylines(canvas, [points], True, (30, 30, 30), 5)
    else:
        cv2.rectangle(canvas, (center[0] - 100, center[1] - 40), (center[0] + 100, center[1] + 45), (35, 70, 160), -1)
        cv2.circle(canvas, (center[0] - 55, center[1] + 45), 22, (20, 20, 20), -1)
        cv2.circle(canvas, (center[0] + 55, center[1] + 45), 22, (20, 20, 20), -1)
        for _ in range(4):
            point = (int(rng.integers(x1, x2)), int(rng.integers(y1, y2)))
            cv2.circle(canvas, point, 8, (40, 170, 220), -1)


def _write_jpeg(path: Path, image: np.ndarray, quality: int) -> tuple[str, int]:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError("synthetic JPEG encoding failed")
    data = encoded.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest(), len(data)


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    arrow: Any = pa
    parquet: Any = pq
    table: Any = arrow.Table.from_pylist(rows)
    parquet.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def generate_presentation_candidates(
    *,
    batch_id: str,
    total_images: int = 20_000,
    base_manifest: str | Path = DEFAULT_BASE_MANIFEST,
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    minimum_free_gib: float = 15.0,
    maximum_acquisition_gib: float = 10.0,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    """Generate a balanced, reproducible presentation pool from reviewed train art."""
    bases = _load_bases(project_path(base_manifest))
    labels = sorted(bases)
    if len(labels) != 78:
        raise ValueError(f"Expected reviewed train artwork for 78 classes, found {len(labels)}")
    if any(len(values) < 3 for values in bases.values()):
        raise ValueError("Every class requires at least three distinct reviewed train artworks")
    if total_images < len(labels) + len(NEGATIVE_KINDS):
        raise ValueError("total_images is too small for balanced class and negative coverage")
    negative_count = max(len(NEGATIVE_KINDS), round(total_images * 0.064))
    positive_count = total_images - negative_count
    per_class, positive_remainder = divmod(positive_count, len(labels))
    root = project_path(acquisition_root)
    report_path = root / "manifests" / "synthetic" / f"{batch_id}.json"
    progress_path = root / "manifests" / "synthetic" / f"{batch_id}.jsonl"
    parquet_path = root / "manifests" / "synthetic" / f"{batch_id}.parquet"
    output_dir = root / "synthetic" / batch_id / "images"
    audit_path = project_path(audit_root) / "synthetic" / f"{batch_id}.json"
    if report_path.exists():
        if resume:
            value: object = json.loads(report_path.read_text(encoding="utf-8"))
            return dict(_mapping(value)) | {"resume_reused": True}
        raise FileExistsError(f"Refusing to overwrite synthetic report: {report_path}")
    estimated_bytes = total_images * 120_000
    budget = check_disk_budget(
        root,
        acquisition_root=root,
        minimum_free_gib=minimum_free_gib,
        maximum_acquisition_gib=maximum_acquisition_gib,
        planned_additional_bytes=estimated_bytes,
    )
    preflight: dict[str, object] = {
        "batch_id": batch_id,
        "classes": len(labels),
        "base_artworks": sum(len(values) for values in bases.values()),
        "total_images": total_images,
        "positive_images": positive_count,
        "negative_images": negative_count,
        "per_class_floor": per_class,
        "estimated_maximum_bytes": estimated_bytes,
        "disk_allowed": budget.allowed,
        "disk_reasons": list(budget.reasons),
        "dry_run": dry_run,
    }
    if dry_run:
        return preflight
    if not budget.allowed:
        raise RuntimeError(f"Disk budget guard blocked synthesis: {', '.join(budget.reasons)}")
    existing: dict[str, SyntheticRow] = {}
    if progress_path.is_file():
        with progress_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = SyntheticRow.model_validate_json(line)
                except ValueError:
                    continue
                existing[row.sample_id] = row
    rows = dict(existing)
    image_cache: dict[str, np.ndarray] = {}
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as progress:
        global_index = 0
        for label_index, label in enumerate(labels):
            class_count = per_class + (1 if label_index < positive_remainder else 0)
            for variant in range(class_count):
                seed_hex = _stable_id(batch_id, label, variant)
                seed = int(seed_hex[:8], 16)
                sample_id = f"syn_{label}_{variant:04d}_{seed_hex[:10]}"
                global_index += 1
                if sample_id in rows:
                    continue
                rng = np.random.default_rng(seed)
                base = bases[label][variant % len(bases[label])]
                if base.path not in image_cache:
                    image_raw: Any = cv2.imread(base.path, cv2.IMREAD_COLOR)
                    if image_raw is None:
                        raise OSError(f"Unable to decode reviewed base artwork: {base.path}")
                    image_cache[base.path] = cast(np.ndarray[Any, Any], image_raw)
                artwork = image_cache[base.path]
                orientation = "portrait" if variant % 2 == 0 else "landscape"
                width, height = ((384, 512) if orientation == "portrait" else (512, 384))
                device = DEVICES[variant % len(DEVICES)]
                distance = DISTANCES[(variant // 3) % len(DISTANCES)]
                brightness = BRIGHTNESS_LEVELS[(variant // 9) % len(BRIGHTNESS_LEVELS)]
                glare = GLARE_LEVELS[(variant // 27) % len(GLARE_LEVELS)]
                background = BACKGROUNDS[(variant // 5) % len(BACKGROUNDS)]
                composition = COMPOSITIONS[(variant // 7) % len(COMPOSITIONS)]
                material = MATERIALS[(variant // 11) % len(MATERIALS)]
                angle = -45 + (variant * 15) % 91
                canvas, surface = _surface(
                    width,
                    height,
                    material=material,
                    background=background,
                    device=device,
                    rng=rng,
                )
                x1, y1, x2, y2 = surface
                scales = {"near": 0.42, "medium": 0.24, "far": 0.11}
                side = max(18, round(min(x2 - x1, y2 - y1) * scales[distance]))
                center = (
                    _randint(rng, x1 + side // 2, max(x1 + side // 2 + 1, x2 - side // 2)),
                    _randint(rng, y1 + side // 2, max(y1 + side // 2 + 1, y2 - side // 2)),
                )
                bbox = _paste_rotated(
                    canvas, artwork, center=center, target_side=side, angle=angle
                )
                if composition in {"search_results", "multi_sign"}:
                    for distractor in range(3 if composition == "search_results" else 1):
                        dx = x1 + 20 + distractor * 42
                        dy = y1 + 20
                        cv2.rectangle(canvas, (dx, dy), (dx + 28, dy + 28), (180, 180, 180), -1)
                transformed, effect_recipe = _effects(
                    canvas, brightness=brightness, glare=glare, seed=seed
                )
                image_path = output_dir / f"{sample_id}.jpg"
                image_sha256, _size = _write_jpeg(
                    image_path, transformed, _integer(effect_recipe["jpeg_quality"])
                )
                bx1, by1, bx2, by2 = bbox
                area_ratio = ((bx2 - bx1) * (by2 - by1)) / (width * height)
                size_bucket = "very_small" if area_ratio <= 0.001 else "small" if area_ratio <= 0.01 else "medium" if area_ratio <= 0.05 else "large"
                base_rendering_id = _stable_id(base.artwork_id, material, composition)
                row = SyntheticRow(
                    batch_id=batch_id,
                    generated_at=_now(),
                    sample_id=sample_id,
                    instance_id=f"{sample_id}:0",
                    expected_kind="sign",
                    semantic_sign_id=label,
                    capture_domain="electronic_screen" if material == "screen" else "printed_card",
                    image_path=str(image_path),
                    image_sha256=image_sha256,
                    image_width=width,
                    image_height=height,
                    bbox_x1=bx1,
                    bbox_y1=by1,
                    bbox_x2=bx2,
                    bbox_y2=by2,
                    area_ratio=round(area_ratio, 8),
                    size_bucket=size_bucket,
                    source_artwork_id=base.artwork_id,
                    source_artwork_sha256=base.sha256,
                    source_artwork_group=base.source_group,
                    base_rendering_id=base_rendering_id,
                    related_capture_group_id=f"synthetic-base:{base.artwork_id}",
                    simulated_device_id=device,
                    screen_orientation=orientation,
                    orientation_degrees=angle,
                    presentation_distance=distance,
                    presentation_brightness=brightness,
                    presentation_glare=glare,
                    presentation_background=background,
                    scene_composition=composition,
                    presentation_material=material,
                    transform_seed=seed,
                    generation_recipe={
                        "version": SYNTHETIC_VERSION,
                        "surface": {"width": width, "height": height},
                        "target_side": side,
                        "target_center": list(center),
                        "source_dataset": base.source_dataset,
                        "source_url": base.source_url,
                        "source_licence_status": base.licence_status,
                        **effect_recipe,
                    },
                )
                progress.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")
                progress.flush()
                rows[sample_id] = row

        for negative_index in range(negative_count):
            kind = NEGATIVE_KINDS[negative_index % len(NEGATIVE_KINDS)]
            seed_hex = _stable_id(batch_id, "negative", kind, negative_index)
            seed = int(seed_hex[:8], 16)
            sample_id = f"syn_negative_{kind}_{negative_index:04d}_{seed_hex[:10]}"
            global_index += 1
            if sample_id in rows:
                continue
            rng = np.random.default_rng(seed)
            orientation = "portrait" if negative_index % 2 == 0 else "landscape"
            width, height = ((384, 512) if orientation == "portrait" else (512, 384))
            device = DEVICES[negative_index % len(DEVICES)]
            brightness = BRIGHTNESS_LEVELS[(negative_index // 3) % len(BRIGHTNESS_LEVELS)]
            glare = GLARE_LEVELS[(negative_index // 9) % len(GLARE_LEVELS)]
            background = BACKGROUNDS[(negative_index // 5) % len(BACKGROUNDS)]
            canvas, surface = _surface(
                width,
                height,
                material="screen",
                background=background,
                device=device,
                rng=rng,
            )
            _negative_content(canvas, surface, kind=kind, rng=rng)
            transformed, effect_recipe = _effects(
                canvas, brightness=brightness, glare=glare, seed=seed
            )
            image_path = output_dir / f"{sample_id}.jpg"
            image_sha256, _size = _write_jpeg(
                image_path, transformed, _integer(effect_recipe["jpeg_quality"])
            )
            row = SyntheticRow(
                batch_id=batch_id,
                generated_at=_now(),
                sample_id=sample_id,
                instance_id=f"{sample_id}:no-sign",
                expected_kind="no_sign",
                capture_domain="electronic_screen",
                image_path=str(image_path),
                image_sha256=image_sha256,
                image_width=width,
                image_height=height,
                bbox_x1=0,
                bbox_y1=0,
                bbox_x2=0,
                bbox_y2=0,
                area_ratio=0.0,
                size_bucket="",
                base_rendering_id=_stable_id("negative", kind),
                related_capture_group_id=f"synthetic-negative:{kind}",
                simulated_device_id=device,
                screen_orientation=orientation,
                orientation_degrees=-45 + (negative_index * 15) % 91,
                presentation_distance=DISTANCES[negative_index % len(DISTANCES)],
                presentation_brightness=brightness,
                presentation_glare=glare,
                presentation_background=background,
                scene_composition="negative_screen",
                presentation_material="screen",
                negative_kind=kind,
                transform_seed=seed,
                generation_recipe={"version": SYNTHETIC_VERSION, **effect_recipe},
            )
            progress.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")
            progress.flush()
            rows[sample_id] = row
    ordered = [rows[key] for key in sorted(rows)]
    _write_parquet(parquet_path, [row.model_dump(mode="json") for row in ordered])
    class_counts = Counter(row.semantic_sign_id for row in ordered if row.expected_kind == "sign")
    negative_counts = Counter(row.negative_kind for row in ordered if row.expected_kind == "no_sign")
    devices_by_class: dict[str, set[str]] = defaultdict(set)
    orientations_by_class: dict[str, set[str]] = defaultdict(set)
    distances_by_class: dict[str, set[str]] = defaultdict(set)
    brightness_by_class: dict[str, set[str]] = defaultdict(set)
    glare_by_class: dict[str, set[str]] = defaultdict(set)
    backgrounds_by_class: dict[str, set[str]] = defaultdict(set)
    compositions_by_class: dict[str, set[str]] = defaultdict(set)
    materials_by_class: dict[str, set[str]] = defaultdict(set)
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in ordered:
        splits_by_group[row.related_capture_group_id].add(row.split)
        if row.expected_kind != "sign":
            continue
        devices_by_class[row.semantic_sign_id].add(row.simulated_device_id)
        orientations_by_class[row.semantic_sign_id].add(row.screen_orientation)
        distances_by_class[row.semantic_sign_id].add(row.presentation_distance)
        brightness_by_class[row.semantic_sign_id].add(row.presentation_brightness)
        glare_by_class[row.semantic_sign_id].add(row.presentation_glare)
        backgrounds_by_class[row.semantic_sign_id].add(row.presentation_background)
        compositions_by_class[row.semantic_sign_id].add(row.scene_composition)
        materials_by_class[row.semantic_sign_id].add(row.presentation_material)
    per_class_complete = {
        label: (
            set(DEVICES).issubset(devices_by_class[label])
            and {"portrait", "landscape"}.issubset(orientations_by_class[label])
            and set(DISTANCES).issubset(distances_by_class[label])
            and set(BRIGHTNESS_LEVELS).issubset(brightness_by_class[label])
            and {"none", "glare"}.issubset(glare_by_class[label])
            and len(backgrounds_by_class[label]) >= 2
            and set(COMPOSITIONS).issubset(compositions_by_class[label])
            and {"screen", "matte", "glossy"}.issubset(materials_by_class[label])
        )
        for label in labels
    }
    actual_bytes = sum(Path(row.image_path).stat().st_size for row in ordered)
    report: dict[str, object] = preflight | {
        "dry_run": False,
        "synthetic_version": SYNTHETIC_VERSION,
        "generated_rows": len(ordered),
        "actual_bytes": actual_bytes,
        "class_counts": dict(sorted(class_counts.items())),
        "negative_counts": dict(sorted(negative_counts.items())),
        "class_balance_range": [min(class_counts.values()), max(class_counts.values())],
        "classes_with_complete_recipe_coverage": sum(per_class_complete.values()),
        "group_split_crossings": sum(len(splits) > 1 for splits in splits_by_group.values()),
        "all_rows_quarantined": all(
            row.collection_state == "quarantined"
            and row.permitted_use == "none_pending_review"
            and row.review_decision == "pending"
            for row in ordered
        ),
        "outputs": {
            "report": str(report_path),
            "audit_report": str(audit_path),
            "manifest_jsonl": str(progress_path),
            "manifest_parquet": str(parquet_path),
            "images": str(output_dir),
        },
        "exit_condition": {
            "passed": len(ordered) == total_images
            and len(class_counts) == 78
            and max(class_counts.values()) - min(class_counts.values()) <= 1
            and set(negative_counts) == set(NEGATIVE_KINDS)
            and all(per_class_complete.values())
            and all(len(splits) == 1 for splits in splits_by_group.values()),
            "basis": "All 78 classes and required negative compositions have balanced, recipe-traceable, group-safe synthetic training coverage.",
        },
        "resume_reused": False,
    }
    for path in (report_path, audit_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["SYNTHETIC_VERSION", "BaseArtwork", "SyntheticRow", "generate_presentation_candidates"]
