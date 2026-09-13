"""Render the bounded manual-review package for Phase C teammate detector data."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from roadsign_assist.paths import PROJECT_ROOT

RELEASE_ID = "detector_production_assignment_v1_20260829"
DEFAULT_QUEUE = Path(f"data/manifests/{RELEASE_ID}_reviews/teammate_visual_review.csv")
DEFAULT_OUTPUT = Path(f"outputs/review/{RELEASE_ID}/teammate_boxes")
DEFAULT_EMTD_QUEUE = Path(f"data/manifests/{RELEASE_ID}_reviews/emtd_box_review.csv")
DEFAULT_EMTD_OUTPUT = Path(f"outputs/review/{RELEASE_ID}/emtd_boxes")
SHEET_COLUMNS = 6
SHEET_ROWS = 5
SHEET_SIZE = SHEET_COLUMNS * SHEET_ROWS
INDEX_FIELDS = [
    "sample_id",
    "layout_root_id",
    "review_reason",
    "source_image_id",
    "source_path",
    "review_sheet",
    "sheet_position",
    "review_decision",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _reason(row: dict[str, str]) -> str:
    return row.get("review_reasons", "").strip() or "stratified_sample_only"


def _safe_name(value: str) -> str:
    return "_".join("".join(char if char.isalnum() else "_" for char in value).split("_"))


def _label_for_image(image: Path) -> Path:
    try:
        images_index = image.parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Expected an image path below an images directory: {image}") from exc
    prefix = Path(*image.parts[:images_index])
    suffix = Path(*image.parts[images_index + 1 :]).with_suffix(".txt")
    return prefix / "labels" / suffix


def _read_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing YOLO label: {path}")
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number} must have five YOLO fields")
        try:
            _, x, y, width, height = (float(value) for value in fields)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} has invalid YOLO values") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number} has non-positive geometry")
        boxes.append((x, y, width, height))
    if not boxes:
        raise ValueError(f"{path} is empty")
    return boxes


def _overlay_thumbnail(
    image_path: Path,
    boxes: list[tuple[float, float, float, float]],
    *,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for x, y, box_width, box_height in boxes:
        left = max(0, round((x - box_width / 2) * width))
        top = max(0, round((y - box_height / 2) * height))
        right = min(width - 1, round((x + box_width / 2) * width))
        bottom = min(height - 1, round((y + box_height / 2) * height))
        draw.rectangle((left, top, right, bottom), outline=(255, 73, 73), width=max(2, width // 500))
    return ImageOps.contain(image, (tile_width - 12, tile_height - 42))


def _render_sheet(rows: list[dict[str, str]], output: Path) -> None:
    tile_width = 240
    tile_height = 190
    sheet = Image.new("RGB", (SHEET_COLUMNS * tile_width, SHEET_ROWS * tile_height), (20, 25, 24))
    draw = ImageDraw.Draw(sheet)
    font = _font(13)
    for index, row in enumerate(rows):
        column = index % SHEET_COLUMNS
        grid_row = index // SHEET_COLUMNS
        left = column * tile_width
        top = grid_row * tile_height
        image_path = Path(row["source_path"])
        thumbnail = _overlay_thumbnail(
            image_path,
            _read_boxes(_label_for_image(image_path)),
            tile_width=tile_width,
            tile_height=tile_height,
        )
        image_left = left + (tile_width - thumbnail.width) // 2
        image_top = top + (tile_height - 38 - thumbnail.height) // 2 + 4
        sheet.paste(thumbnail, (image_left, image_top))
        draw.rectangle((left, top + tile_height - 38, left + tile_width, top + tile_height), fill=(31, 42, 38))
        reason = {"large_box": "LB", "tiny_box": "TB"}.get(_reason(row), "S")
        label = f"{index + 1:02d} {row['sample_id'][:10]} {reason}"
        draw.text((left + 6, top + tile_height - 27), label, fill=(232, 239, 236), font=font)
        draw.rectangle((left, top, left + tile_width - 1, top + tile_height - 1), outline=(63, 78, 72))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite review package: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _build_detector_review_package(
    *,
    project_root: Path,
    queue_path: Path,
    output_root: Path,
    index_name: str,
    package_name: str,
    review_instruction: str,
    overwrite: bool,
) -> dict[str, Any]:
    root = project_root.resolve()
    queue = (root / queue_path).resolve()
    output = (root / output_root).resolve()
    rows = _read_csv(queue)
    if not rows:
        raise ValueError(f"{package_name} review queue is empty: {queue}")
    if any(row.get("review_decision", "").strip() for row in rows):
        raise ValueError(f"{package_name} review package must be generated before any decision is recorded")
    _prepare_output(output, overwrite=overwrite)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["layout_root_id"], _reason(row))].append(row)
    index_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    sheet_count = 0
    for (layout, reason), group_rows in sorted(grouped.items()):
        group_rows.sort(key=lambda row: row["sample_id"])
        group_root = output / "sheets" / _safe_name(layout) / _safe_name(reason)
        for start in range(0, len(group_rows), SHEET_SIZE):
            page = group_rows[start : start + SHEET_SIZE]
            sheet_path = group_root / f"sheet_{start // SHEET_SIZE + 1:03d}.jpg"
            try:
                _render_sheet(page, sheet_path)
            except (OSError, ValueError) as exc:
                failures.extend(
                    {"sample_id": row["sample_id"], "source_path": row["source_path"], "reason": str(exc)}
                    for row in page
                )
                continue
            sheet_count += 1
            relative_sheet = sheet_path.relative_to(output).as_posix()
            for position, row in enumerate(page, start=1):
                index_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "layout_root_id": row["layout_root_id"],
                        "review_reason": _reason(row),
                        "source_image_id": row["source_image_id"],
                        "source_path": row["source_path"],
                        "review_sheet": relative_sheet,
                        "sheet_position": str(position),
                        "review_decision": row.get("review_decision", ""),
                    }
                )
    _write_csv(output / index_name, index_rows, INDEX_FIELDS)
    _write_csv(
        output / "render_failures.csv",
        failures,
        ["sample_id", "source_path", "reason"],
    )
    summary = {
        "release_id": RELEASE_ID,
        "queue_path": queue.as_posix(),
        "required_rows": len(rows),
        "indexed_rows": len(index_rows),
        "render_failures": len(failures),
        "sheet_count": sheet_count,
        "by_layout": dict(sorted(Counter(row["layout_root_id"] for row in rows).items())),
        "by_reason": dict(sorted(Counter(_reason(row) for row in rows).items())),
        "review_instruction": review_instruction,
        "use_policy": "assignment_only_no_redistribution_no_shared_dvc",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Unable to render {len(failures)} {package_name} review rows")
    return summary


def build_teammate_detector_review_package(
    *,
    project_root: Path = PROJECT_ROOT,
    queue_path: Path = DEFAULT_QUEUE,
    output_root: Path = DEFAULT_OUTPUT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render every required teammate review row as a labelled overlay sheet."""
    return _build_detector_review_package(
        project_root=project_root,
        queue_path=queue_path,
        output_root=output_root,
        index_name="teammate_review_index.csv",
        package_name="Teammate",
        review_instruction=(
            "Inspect every overlay tile. Record accept or reject only in the canonical "
            "teammate_visual_review.csv queue."
        ),
        overwrite=overwrite,
    )


def build_emtd_detector_review_package(
    *,
    project_root: Path = PROJECT_ROOT,
    queue_path: Path = DEFAULT_EMTD_QUEUE,
    output_root: Path = DEFAULT_EMTD_OUTPUT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render every EMTD merge-eligibility row as a labelled overlay sheet."""
    return _build_detector_review_package(
        project_root=project_root,
        queue_path=queue_path,
        output_root=output_root,
        index_name="emtd_review_index.csv",
        package_name="EMTD",
        review_instruction=(
            "Inspect every overlay tile. Record accept or reject only in the canonical "
            "emtd_box_review.csv queue; accept an image only when every box is correct."
        ),
        overwrite=overwrite,
    )
