from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_sprint_06_warning_symbols"
OUTPUT_ROOT = PROJECT_ROOT / "data/generated/stage_c_sprint_06_warning_symbols"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_sprint_06_warning_symbols.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_sprint_06_warning_symbols.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_sprint_06_warning_symbols"

CLASSES = {
    "bicycle_crossing": "bicycle",
    "school_zone": "school",
    "steep_descent": "steep_descent",
    "uneven_road": "uneven_road",
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


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_warning_base(draw: ImageDraw.ImageDraw, *, yellow: tuple[int, int, int]) -> None:
    diamond = [(256, 44), (468, 256), (256, 468), (44, 256)]
    draw.polygon(diamond, fill=(*yellow, 255), outline=(20, 20, 20, 255))
    draw.line(diamond + [diamond[0]], fill=(20, 20, 20, 255), width=12, joint="curve")


def draw_bicycle(draw: ImageDraw.ImageDraw) -> None:
    black = (18, 18, 18, 255)
    draw.ellipse((120, 288, 220, 388), outline=black, width=18)
    draw.ellipse((292, 288, 392, 388), outline=black, width=18)
    draw.line((170, 338, 234, 238, 300, 338, 210, 338, 258, 338, 234, 238), fill=black, width=16)
    draw.line((234, 238, 244, 202), fill=black, width=16)
    draw.line((244, 202, 284, 202), fill=black, width=14)
    draw.line((300, 338, 336, 248), fill=black, width=16)
    draw.line((336, 248, 374, 232), fill=black, width=14)
    draw.ellipse((220, 134, 270, 184), fill=black)
    draw.line((244, 184, 228, 238), fill=black, width=16)


def draw_school(draw: ImageDraw.ImageDraw) -> None:
    black = (18, 18, 18, 255)
    # Adult/child crossing silhouettes.
    draw.ellipse((164, 134, 214, 184), fill=black)
    draw.line((188, 184, 170, 282), fill=black, width=20)
    draw.line((176, 224, 126, 244), fill=black, width=14)
    draw.line((176, 224, 228, 244), fill=black, width=14)
    draw.line((170, 282, 130, 352), fill=black, width=16)
    draw.line((170, 282, 214, 352), fill=black, width=16)
    draw.ellipse((294, 174, 334, 214), fill=black)
    draw.line((314, 214, 306, 292), fill=black, width=16)
    draw.line((306, 246, 264, 268), fill=black, width=12)
    draw.line((308, 292, 282, 352), fill=black, width=12)
    draw.line((308, 292, 348, 344), fill=black, width=12)
    text = "SCHOOL"
    text_font = font(48)
    bbox = draw.textbbox((0, 0), text, font=text_font)
    draw.text(((512 - (bbox[2] - bbox[0])) / 2, 374), text, fill=black, font=text_font)


def draw_steep_descent(draw: ImageDraw.ImageDraw) -> None:
    black = (18, 18, 18, 255)
    draw.line((116, 178, 394, 360), fill=black, width=28)
    draw.polygon([(116, 178), (394, 360), (116, 360)], fill=black)
    # Small car on slope.
    car = [(238, 222), (322, 274), (302, 308), (218, 256)]
    draw.polygon(car, fill=(250, 210, 50, 255), outline=black)
    draw.ellipse((224, 262, 254, 292), fill=black)
    draw.ellipse((286, 300, 316, 330), fill=black)
    text_font = font(54)
    draw.text((148, 114), "10%", fill=black, font=text_font)


def draw_uneven_road(draw: ImageDraw.ImageDraw) -> None:
    black = (18, 18, 18, 255)
    points = [(106, 316), (150, 316), (184, 248), (230, 248), (266, 316), (314, 316), (354, 254), (408, 254)]
    draw.line(points, fill=black, width=30, joint="curve")
    draw.line((112, 368, 400, 368), fill=black, width=18)


def draw_symbol(symbol: str, *, yellow: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGBA", (DRAW_SIZE, DRAW_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw_warning_base(draw, yellow=yellow)
    if symbol == "bicycle":
        draw_bicycle(draw)
    elif symbol == "school":
        draw_school(draw)
    elif symbol == "steep_descent":
        draw_steep_descent(draw)
    elif symbol == "uneven_road":
        draw_uneven_road(draw)
    else:
        raise ValueError(f"Unsupported symbol: {symbol}")
    return image


def make_background(rng: random.Random) -> Image.Image:
    palette = rng.choice(
        [
            ((140, 153, 145), (106, 118, 111)),
            ((182, 177, 164), (132, 126, 114)),
            ((122, 144, 155), (86, 102, 114)),
            ((136, 158, 132), (101, 118, 99)),
        ]
    )
    image = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE), palette[0])
    draw = ImageDraw.Draw(image)
    for _ in range(28):
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
    yellow = (
        rng.randint(222, 252),
        rng.randint(174, 218),
        rng.randint(24, 62),
    )
    sign = draw_symbol(symbol, yellow=yellow)
    target_size = rng.randint(156, 204)
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
        f"symbol={symbol}; size={target_size}; angle={angle:.2f}; yellow={yellow}; "
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
    for semantic_id, symbol in CLASSES.items():
        for variant_index in range(1, VARIANTS_PER_CLASS + 1):
            synthetic_id = f"S06-{counter:03d}_{semantic_id}"
            output_path = OUTPUT_ROOT / semantic_id / f"{synthetic_id}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image, transform_summary = augment_symbol(semantic_id, symbol, variant_index)
            image.save(output_path, quality=91)
            rows.append(
                {
                    "stage_id": STAGE_ID,
                    "synthetic_id": synthetic_id,
                    "semantic_sign_id": semantic_id,
                    "symbol_family": "warning_symbol",
                    "output_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                    "generation_status": "generated_candidate",
                    "review_status": "pending_stage_d_visual_qc",
                    "transform_summary": transform_summary,
                    "sha256": sha256(output_path),
                    "notes": "Project-owned generated warning-sign candidate; not real camera coverage.",
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
            "These generated candidates close warning-sign gaps at candidate-count "
            "level, but do not replace real local/demo photos."
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
