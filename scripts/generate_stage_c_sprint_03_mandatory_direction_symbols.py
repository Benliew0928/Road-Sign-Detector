from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_sprint_03_mandatory_direction_symbols"
OUTPUT_ROOT = PROJECT_ROOT / "data/generated/stage_c_sprint_03_mandatory_direction_symbols"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_sprint_03_mandatory_direction_symbols.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_sprint_03_mandatory_direction_symbols.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_sprint_03_mandatory_direction_symbols"

CLASSES = {
    "straight_ahead": "up",
    "turn_left": "left",
    "turn_right": "right",
}
VARIANTS_PER_CLASS = 50
OUTPUT_SIZE = 224
DRAW_SIZE = 512

FIELDNAMES = [
    "stage_id",
    "synthetic_id",
    "semantic_sign_id",
    "symbol_family",
    "output_path",
    "generation_status",
    "review_status",
    "transform_summary",
    "sha256",
    "notes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_mandatory_symbol(direction: str, *, blue: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGBA", (DRAW_SIZE, DRAW_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = 38
    draw.ellipse((margin, margin, DRAW_SIZE - margin, DRAW_SIZE - margin), fill=blue)

    if direction == "up":
        points = [
            (256, 82),
            (142, 222),
            (214, 222),
            (214, 374),
            (298, 374),
            (298, 222),
            (370, 222),
        ]
    elif direction == "left":
        points = [
            (82, 256),
            (222, 142),
            (222, 214),
            (374, 214),
            (374, 298),
            (222, 298),
            (222, 370),
        ]
    elif direction == "right":
        points = [
            (430, 256),
            (290, 142),
            (290, 214),
            (138, 214),
            (138, 298),
            (290, 298),
            (290, 370),
        ]
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    draw.polygon(points, fill=(255, 255, 255, 255))
    return image


def make_background(rng: random.Random) -> Image.Image:
    palette = rng.choice(
        [
            ((142, 151, 145), (108, 118, 111)),
            ((178, 182, 172), (126, 139, 128)),
            ((118, 143, 155), (86, 103, 113)),
            ((170, 166, 154), (119, 115, 105)),
        ]
    )
    image = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE), palette[0])
    draw = ImageDraw.Draw(image)
    for _ in range(28):
        x1 = rng.randint(-50, OUTPUT_SIZE)
        y1 = rng.randint(-50, OUTPUT_SIZE)
        x2 = x1 + rng.randint(50, 170)
        y2 = y1 + rng.randint(10, 42)
        mix = rng.random()
        color = tuple(round(a * mix + b * (1 - mix)) for a, b in zip(*palette))
        draw.rectangle((x1, y1, x2, y2), fill=color)
    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.45, 1.3)))


def augment_symbol(semantic_id: str, direction: str, variant_index: int) -> tuple[Image.Image, str]:
    rng = random.Random(f"{STAGE_ID}:{semantic_id}:{variant_index}")
    blue = (
        rng.randint(0, 14),
        rng.randint(76, 116),
        rng.randint(165, 215),
    )
    symbol = draw_mandatory_symbol(direction, blue=blue)

    target_size = rng.randint(154, 204)
    symbol = ImageOps.contain(symbol, (target_size, target_size))
    angle = rng.uniform(-8.0, 8.0)
    symbol = symbol.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    brightness = rng.uniform(0.88, 1.12)
    contrast = rng.uniform(0.88, 1.14)
    rgb = symbol.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    symbol = Image.merge("RGBA", (*rgb.split(), symbol.getchannel("A")))

    background = make_background(rng).convert("RGBA")
    x = (OUTPUT_SIZE - symbol.width) // 2 + rng.randint(-9, 9)
    y = (OUTPUT_SIZE - symbol.height) // 2 + rng.randint(-9, 9)
    shadow = Image.new("RGBA", symbol.size, (0, 0, 0, 0))
    shadow_alpha = symbol.getchannel("A").filter(ImageFilter.GaussianBlur(radius=3))
    shadow.putalpha(shadow_alpha.point(lambda value: int(value * 0.20)))
    background.alpha_composite(shadow, (x + 4, y + 5))
    background.alpha_composite(symbol, (x, y))
    output = background.convert("RGB")

    if rng.random() < 0.35:
        output = output.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.08, 0.25)))

    summary = (
        f"direction={direction}; size={target_size}; angle={angle:.2f}; "
        f"blue={blue}; brightness={brightness:.2f}; contrast={contrast:.2f}; offset=({x},{y})"
    )
    return output, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def render_sheet(rows: list[dict[str, str]], output_path: Path, *, max_rows: int | None = None) -> str:
    rows = rows[:max_rows] if max_rows else rows
    columns = 5
    tile_width = 210
    tile_height = 225
    label_height = 48
    sheet = Image.new(
        "RGB",
        (columns * tile_width, math.ceil(len(rows) / columns) * tile_height),
        color=(20, 25, 24),
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        with Image.open(PROJECT_ROOT / row["output_path"]) as source:
            thumbnail = ImageOps.contain(source.convert("RGB"), (tile_width - 16, tile_height - label_height - 16))
        sheet.paste(
            thumbnail,
            (
                x + (tile_width - thumbnail.width) // 2,
                y + 8 + (tile_height - label_height - 16 - thumbnail.height) // 2,
            ),
        )
        draw.rectangle((x, y + tile_height - label_height, x + tile_width, y + tile_height), fill=(31, 42, 38))
        draw.text((x + 7, y + tile_height - label_height + 6), row["synthetic_id"], fill=(232, 239, 236))
        draw.text((x + 7, y + tile_height - label_height + 24), row["semantic_sign_id"], fill=(180, 198, 190))
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline=(63, 78, 72), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    if OUTPUT_ROOT.exists():
        resolved = OUTPUT_ROOT.resolve()
        expected = (PROJECT_ROOT / "data/generated").resolve()
        if expected not in resolved.parents:
            raise RuntimeError(f"Refusing to clear unexpected output root: {resolved}")
        shutil.rmtree(OUTPUT_ROOT)

    rows: list[dict[str, str]] = []
    counter = 1
    for semantic_id, direction in CLASSES.items():
        for variant_index in range(1, VARIANTS_PER_CLASS + 1):
            synthetic_id = f"S03-{counter:03d}_{semantic_id}"
            output_path = OUTPUT_ROOT / semantic_id / f"{synthetic_id}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image, transform_summary = augment_symbol(semantic_id, direction, variant_index)
            image.save(output_path, quality=91)
            rows.append(
                {
                    "stage_id": STAGE_ID,
                    "synthetic_id": synthetic_id,
                    "semantic_sign_id": semantic_id,
                    "symbol_family": "mandatory_direction_arrow",
                    "output_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                    "generation_status": "generated_candidate",
                    "review_status": "pending_stage_d_visual_qc",
                    "transform_summary": transform_summary,
                    "sha256": sha256(output_path),
                    "notes": "Project-owned generated reference-symbol candidate; not real camera coverage.",
                }
            )
            counter += 1

    write_csv(MANIFEST_PATH, rows)
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["semantic_sign_id"], []).append(row)

    contact_sheets = {
        label: render_sheet(label_rows, SHEET_ROOT / f"{label}.jpg", max_rows=30)
        for label, label_rows in sorted(by_class.items())
    }
    contact_sheets["_sample_all_candidates"] = render_sheet(rows, SHEET_ROOT / "_sample_all_candidates.jpg", max_rows=45)

    report = {
        "schema_version": "1.0",
        "stage_id": STAGE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_candidates": len(rows),
        "candidate_counts_by_class": {
            label: len(label_rows) for label, label_rows in sorted(by_class.items())
        },
        "manifest": MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "output_root": OUTPUT_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "contact_sheets": contact_sheets,
        "status": "generated_reference_symbols_pending_stage_d_qc",
        "important_note": (
            "These are project-owned reference-symbol candidates for zero-data mandatory "
            "direction classes. They close candidate count gaps but still need real/demo "
            "photos for final presentation reliability."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    for key, value in contact_sheets.items():
        print(f"Wrote {key}: {value}")


if __name__ == "__main__":
    main()
