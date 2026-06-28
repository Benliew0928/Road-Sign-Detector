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
STAGE_ID = "stage_c_sprint_04_compound_mandatory_symbols"
OUTPUT_ROOT = PROJECT_ROOT / "data/generated/stage_c_sprint_04_compound_mandatory_symbols"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_sprint_04_compound_mandatory_symbols.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_sprint_04_compound_mandatory_symbols.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_sprint_04_compound_mandatory_symbols"

CLASSES = {
    "straight_or_right": ("straight_or_right", 50),
    "turn_left_or_right": ("left_or_right", 50),
    "roundabout_mandatory": ("roundabout", 50),
    "pass_either_side": ("pass_either_side", 40),
}
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


def arrow_head(tip: tuple[float, float], direction: float, *, length: float, width: float) -> list[tuple[float, float]]:
    tx, ty = tip
    bx = tx - math.cos(direction) * length
    by = ty - math.sin(direction) * length
    px = -math.sin(direction)
    py = math.cos(direction)
    return [
        (tx, ty),
        (bx + px * width / 2, by + py * width / 2),
        (bx - px * width / 2, by - py * width / 2),
    ]


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    width: int = 42,
    head_length: int = 58,
    head_width: int = 88,
) -> None:
    sx, sy = start
    ex, ey = end
    direction = math.atan2(ey - sy, ex - sx)
    line_end = (
        ex - math.cos(direction) * head_length * 0.58,
        ey - math.sin(direction) * head_length * 0.58,
    )
    draw.line((sx, sy, line_end[0], line_end[1]), fill=(255, 255, 255, 255), width=width)
    draw.polygon(arrow_head(end, direction, length=head_length, width=head_width), fill=(255, 255, 255, 255))


def draw_roundabout(draw: ImageDraw.ImageDraw) -> None:
    box = (145, 145, 367, 367)
    width = 32
    arcs = [(8, 104), (128, 224), (248, 344)]
    for start, end in arcs:
        draw.arc(box, start=start, end=end, fill=(255, 255, 255, 255), width=width)
        angle = math.radians(end)
        center = (256, 256)
        radius = 111
        tip = (center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius)
        tangent = angle + math.pi / 2
        draw.polygon(arrow_head(tip, tangent, length=46, width=62), fill=(255, 255, 255, 255))


def draw_symbol(symbol: str, *, blue: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGBA", (DRAW_SIZE, DRAW_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = 38
    draw.ellipse((margin, margin, DRAW_SIZE - margin, DRAW_SIZE - margin), fill=blue)

    if symbol == "straight_or_right":
        draw_arrow(draw, (226, 365), (226, 110), width=42, head_length=58, head_width=86)
        draw_arrow(draw, (226, 292), (365, 180), width=40, head_length=56, head_width=82)
    elif symbol == "left_or_right":
        draw_arrow(draw, (258, 256), (112, 256), width=42, head_length=58, head_width=86)
        draw_arrow(draw, (254, 256), (400, 256), width=42, head_length=58, head_width=86)
    elif symbol == "pass_either_side":
        draw_arrow(draw, (224, 138), (136, 374), width=38, head_length=48, head_width=70)
        draw_arrow(draw, (288, 138), (376, 374), width=38, head_length=48, head_width=70)
    elif symbol == "roundabout":
        draw_roundabout(draw)
    else:
        raise ValueError(f"Unsupported symbol: {symbol}")
    return image


def make_background(rng: random.Random) -> Image.Image:
    palette = rng.choice(
        [
            ((136, 148, 142), (104, 115, 110)),
            ((178, 176, 166), (127, 124, 115)),
            ((124, 144, 158), (86, 103, 116)),
            ((148, 160, 134), (105, 118, 98)),
        ]
    )
    image = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE), palette[0])
    draw = ImageDraw.Draw(image)
    for _ in range(30):
        x1 = rng.randint(-45, OUTPUT_SIZE)
        y1 = rng.randint(-45, OUTPUT_SIZE)
        x2 = x1 + rng.randint(45, 165)
        y2 = y1 + rng.randint(10, 42)
        mix = rng.random()
        color = tuple(round(a * mix + b * (1 - mix)) for a, b in zip(*palette))
        draw.rectangle((x1, y1, x2, y2), fill=color)
    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.45, 1.25)))


def augment_symbol(semantic_id: str, symbol: str, variant_index: int) -> tuple[Image.Image, str]:
    rng = random.Random(f"{STAGE_ID}:{semantic_id}:{variant_index}")
    blue = (
        rng.randint(0, 15),
        rng.randint(78, 118),
        rng.randint(162, 214),
    )
    sign = draw_symbol(symbol, blue=blue)
    target_size = rng.randint(154, 204)
    sign = ImageOps.contain(sign, (target_size, target_size))
    angle = rng.uniform(-7.0, 7.0)
    sign = sign.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    brightness = rng.uniform(0.88, 1.12)
    contrast = rng.uniform(0.88, 1.14)
    rgb = sign.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    sign = Image.merge("RGBA", (*rgb.split(), sign.getchannel("A")))

    background = make_background(rng).convert("RGBA")
    x = (OUTPUT_SIZE - sign.width) // 2 + rng.randint(-9, 9)
    y = (OUTPUT_SIZE - sign.height) // 2 + rng.randint(-9, 9)
    shadow = Image.new("RGBA", sign.size, (0, 0, 0, 0))
    shadow_alpha = sign.getchannel("A").filter(ImageFilter.GaussianBlur(radius=3))
    shadow.putalpha(shadow_alpha.point(lambda value: int(value * 0.20)))
    background.alpha_composite(shadow, (x + 4, y + 5))
    background.alpha_composite(sign, (x, y))
    output = background.convert("RGB")
    if rng.random() < 0.35:
        output = output.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.08, 0.24)))

    summary = (
        f"symbol={symbol}; size={target_size}; angle={angle:.2f}; blue={blue}; "
        f"brightness={brightness:.2f}; contrast={contrast:.2f}; offset=({x},{y})"
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
    for semantic_id, (symbol, count) in CLASSES.items():
        for variant_index in range(1, count + 1):
            synthetic_id = f"S04-{counter:03d}_{semantic_id}"
            output_path = OUTPUT_ROOT / semantic_id / f"{synthetic_id}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image, transform_summary = augment_symbol(semantic_id, symbol, variant_index)
            image.save(output_path, quality=91)
            rows.append(
                {
                    "stage_id": STAGE_ID,
                    "synthetic_id": synthetic_id,
                    "semantic_sign_id": semantic_id,
                    "symbol_family": "compound_mandatory_direction",
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
    contact_sheets["_sample_all_candidates"] = render_sheet(rows, SHEET_ROOT / "_sample_all_candidates.jpg", max_rows=60)

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
            "These generated candidates close compound mandatory direction gaps at "
            "candidate-count level, but do not replace real local/demo photos."
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
