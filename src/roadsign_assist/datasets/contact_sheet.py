from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from roadsign_assist.paths import OFFICIAL_ROOT, project_path


@dataclass(frozen=True)
class ReviewTile:
    label: str
    image_path: Path
    crop: tuple[int, int, int, int] | None = None


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_contact_sheet(
    tiles: list[ReviewTile],
    output_path: str | Path,
    *,
    columns: int = 8,
    tile_width: int = 180,
    tile_height: int = 170,
) -> Path:
    if not tiles:
        raise ValueError("Cannot render an empty contact sheet")
    if columns < 1:
        raise ValueError("columns must be positive")

    rows = math.ceil(len(tiles) / columns)
    label_height = 30
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        color=(20, 25, 24),
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(15)

    for index, tile in enumerate(tiles):
        column = index % columns
        row = index // columns
        x = column * tile_width
        y = row * tile_height
        with Image.open(tile.image_path) as source:
            image = source.convert("RGB")
            if tile.crop is not None:
                image = image.crop(tile.crop)
            thumbnail = ImageOps.contain(
                image,
                (tile_width - 16, tile_height - label_height - 16),
            )
        image_x = x + (tile_width - thumbnail.width) // 2
        image_y = y + 8 + (tile_height - label_height - 16 - thumbnail.height) // 2
        sheet.paste(thumbnail, (image_x, image_y))
        draw.rectangle(
            (x, y + tile_height - label_height, x + tile_width, y + tile_height),
            fill=(31, 42, 38),
        )
        draw.text(
            (x + 7, y + tile_height - label_height + 6),
            tile.label,
            fill=(232, 239, 236),
            font=font,
        )
        draw.rectangle(
            (x, y, x + tile_width - 1, y + tile_height - 1),
            outline=(63, 78, 72),
            width=1,
        )

    resolved = project_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(resolved, quality=92)
    return resolved


def coursework_tiles(
    manifest_path: str | Path = "data/manifests/official_images.csv",
    *,
    representatives_only: bool = False,
) -> list[ReviewTile]:
    path = project_path(manifest_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if representatives_only:
        selected: dict[str, dict[str, str]] = {}
        for row in rows:
            selected.setdefault(row["coursework_id_candidate"], row)
        rows = [selected[key] for key in sorted(selected)]
    else:
        rows.sort(key=lambda row: (row["coursework_id_candidate"], row["filename"]))
    root = OFFICIAL_ROOT / "assignment_images"
    return [
        ReviewTile(
            label=(
                row["coursework_id_candidate"]
                if representatives_only
                else f"{row['coursework_id_candidate']} {row['filename']}"
            ),
            image_path=root / row["relative_path"],
        )
        for row in rows
    ]


def emtd_class_tiles(
    ground_truth_path: str | Path = "data/raw/emtd/metadata/GT.csv",
    image_root: str | Path = "data/raw/emtd/images",
) -> list[ReviewTile]:
    ground_truth = project_path(ground_truth_path)
    root = project_path(image_root)
    by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
    with ground_truth.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            by_class[int(row["Class ID"])].append(row)

    available = {path.name.casefold(): path for path in root.iterdir() if path.is_file()}
    tiles: list[ReviewTile] = []
    for class_id, rows in sorted(by_class.items()):
        candidate = next(
            (row for row in rows if row["filename"].casefold() in available),
            None,
        )
        if candidate is None:
            continue
        image_path = available[candidate["filename"].casefold()]
        with Image.open(image_path) as image:
            width, height = image.size
        x1 = int(candidate["xmin"])
        y1 = int(candidate["ymin"])
        x2 = int(candidate["xmax"])
        y2 = int(candidate["ymax"])
        margin_x = max(4, round((x2 - x1) * 0.20))
        margin_y = max(4, round((y2 - y1) * 0.20))
        crop = (
            max(0, x1 - margin_x),
            max(0, y1 - margin_y),
            min(width, x2 + margin_x),
            min(height, y2 + margin_y),
        )
        tiles.append(
            ReviewTile(
                label=f"EMTD class {class_id:02d}",
                image_path=image_path,
                crop=crop,
            )
        )
    return tiles


def directory_tiles(root: str | Path) -> list[ReviewTile]:
    resolved = project_path(root)
    return [ReviewTile(label=path.stem, image_path=path) for path in sorted(resolved.glob("*.jpg"))]
