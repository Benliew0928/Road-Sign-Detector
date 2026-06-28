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
STAGE_ID = "stage_c_sprint_07_regulatory_text_symbols"
OUTPUT_ROOT = PROJECT_ROOT / "data/generated/stage_c_sprint_07_regulatory_text_symbols"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_sprint_07_regulatory_text_symbols.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_sprint_07_regulatory_text_symbols.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_sprint_07_regulatory_text_symbols"

CLASSES = {
    "motor_vehicles_only": ("mandatory_motor", 50),
    "no_lane_changing": ("no_lane_changing", 50),
    "no_motor_vehicles": ("no_motor", 50),
    "slow_text": ("slow_text", 50),
    "sound_horn": ("sound_horn", 50),
    "stop_for_checking": ("stop_check", 50),
    "width_restriction": ("width_limit", 49),
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
    "ocr_text",
    "parameter_value",
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


def text_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, *, fill, font_obj) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font_obj, spacing=6, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2
    draw.multiline_text((x, y), text, fill=fill, font=font_obj, spacing=6, align="center")


def arrow_head(tip: tuple[float, float], direction: float, *, length: float, width: float) -> list[tuple[float, float]]:
    tx, ty = tip
    bx = tx - math.cos(direction) * length
    by = ty - math.sin(direction) * length
    px = -math.sin(direction)
    py = math.cos(direction)
    return [(tx, ty), (bx + px * width / 2, by + py * width / 2), (bx - px * width / 2, by - py * width / 2)]


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, *, fill, width=34, head_length=48, head_width=70) -> None:
    sx, sy = start
    ex, ey = end
    direction = math.atan2(ey - sy, ex - sx)
    line_end = (ex - math.cos(direction) * head_length * 0.58, ey - math.sin(direction) * head_length * 0.58)
    draw.line((sx, sy, line_end[0], line_end[1]), fill=fill, width=width)
    draw.polygon(arrow_head(end, direction, length=head_length, width=head_width), fill=fill)


def draw_car(draw: ImageDraw.ImageDraw, *, fill, x: int, y: int, scale: float = 1.0) -> None:
    body = [x, y + int(36 * scale), x + int(150 * scale), y + int(82 * scale)]
    roof = [
        (x + int(34 * scale), y + int(36 * scale)),
        (x + int(58 * scale), y + int(8 * scale)),
        (x + int(112 * scale), y + int(8 * scale)),
        (x + int(136 * scale), y + int(36 * scale)),
    ]
    draw.rectangle(body, fill=fill)
    draw.polygon(roof, fill=fill)
    draw.ellipse((x + int(20 * scale), y + int(72 * scale), x + int(58 * scale), y + int(110 * scale)), fill=fill)
    draw.ellipse((x + int(104 * scale), y + int(72 * scale), x + int(142 * scale), y + int(110 * scale)), fill=fill)


def draw_motorcycle(draw: ImageDraw.ImageDraw, *, fill, x: int, y: int, scale: float = 1.0) -> None:
    draw.ellipse((x, y + int(58 * scale), x + int(52 * scale), y + int(110 * scale)), outline=fill, width=max(6, int(10 * scale)))
    draw.ellipse((x + int(96 * scale), y + int(58 * scale), x + int(148 * scale), y + int(110 * scale)), outline=fill, width=max(6, int(10 * scale)))
    draw.line((x + int(26 * scale), y + int(84 * scale), x + int(74 * scale), y + int(34 * scale), x + int(118 * scale), y + int(84 * scale)), fill=fill, width=max(6, int(10 * scale)))
    draw.line((x + int(74 * scale), y + int(34 * scale), x + int(102 * scale), y + int(34 * scale)), fill=fill, width=max(6, int(8 * scale)))
    draw.ellipse((x + int(60 * scale), y, x + int(94 * scale), y + int(34 * scale)), fill=fill)


def draw_blue_circle(draw: ImageDraw.ImageDraw, blue: tuple[int, int, int]) -> None:
    draw.ellipse((42, 42, 470, 470), fill=(*blue, 255))


def draw_red_prohibition_base(draw: ImageDraw.ImageDraw, red: tuple[int, int, int]) -> None:
    draw.ellipse((42, 42, 470, 470), fill=(*red, 255))
    draw.ellipse((82, 82, 430, 430), fill=(248, 248, 242, 255))


def draw_symbol(symbol: str, rng: random.Random) -> tuple[Image.Image, str, str]:
    image = Image.new("RGBA", (DRAW_SIZE, DRAW_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    black = (18, 18, 18, 255)
    white = (255, 255, 255, 255)
    red = (rng.randint(190, 225), rng.randint(18, 42), rng.randint(26, 50))
    blue = (rng.randint(0, 16), rng.randint(80, 120), rng.randint(166, 216))
    ocr_text = ""
    parameter_value = ""

    if symbol == "mandatory_motor":
        draw_blue_circle(draw, blue)
        draw_car(draw, fill=white, x=120, y=150, scale=1.05)
        draw_motorcycle(draw, fill=white, x=190, y=250, scale=0.92)
    elif symbol == "sound_horn":
        draw_blue_circle(draw, blue)
        draw.polygon([(150, 275), (230, 215), (230, 335)], fill=white)
        draw.rectangle((112, 250, 156, 300), fill=white)
        draw.arc((230, 205, 360, 345), start=-50, end=50, fill=white, width=26)
        draw.arc((260, 165, 430, 385), start=-45, end=45, fill=white, width=22)
    elif symbol == "no_motor":
        draw_red_prohibition_base(draw, red)
        draw_car(draw, fill=black, x=106, y=146, scale=0.94)
        draw_motorcycle(draw, fill=black, x=214, y=250, scale=0.82)
        draw.line((152, 106, 382, 406), fill=(*red, 255), width=50)
    elif symbol == "no_lane_changing":
        draw_red_prohibition_base(draw, red)
        draw.line((192, 350, 192, 150), fill=black, width=24)
        draw.line((320, 350, 320, 150), fill=black, width=24)
        draw_arrow(draw, (192, 310), (310, 190), fill=black, width=24, head_length=42, head_width=58)
        draw.line((152, 106, 382, 406), fill=(*red, 255), width=50)
    elif symbol == "width_limit":
        draw_red_prohibition_base(draw, red)
        value = rng.choice(["2.5m", "2.7m", "3.0m", "2.4m"])
        parameter_value = value
        ocr_text = value
        draw_arrow(draw, (168, 256), (106, 256), fill=black, width=20, head_length=34, head_width=46)
        draw_arrow(draw, (344, 256), (406, 256), fill=black, width=20, head_length=34, head_width=46)
        text_center(draw, (160, 178, 352, 322), value, fill=black, font_obj=font(72))
    elif symbol == "slow_text":
        text_options = ["SLOW", "PERLAHAN", "SLOW\\nDOWN", "\\u6162"]
        raw_text = rng.choice(text_options)
        text = raw_text.encode("utf-8").decode("unicode_escape") if raw_text.startswith("\\u") else raw_text
        ocr_text = text.replace("\n", " ")
        draw.rounded_rectangle((62, 142, 450, 370), radius=18, fill=(245, 198, 42, 255), outline=black, width=12)
        text_center(draw, (82, 158, 430, 352), text, fill=black, font_obj=font(76 if len(text) <= 5 else 52))
    elif symbol == "stop_check":
        text_options = ["STOP\\nCHECK", "STOP\\nPERIKSA", "BERHENTI\\nPERIKSA"]
        text = rng.choice(text_options)
        ocr_text = text.replace("\n", " ")
        octagon = [(210, 60), (302, 60), (452, 210), (452, 302), (302, 452), (210, 452), (60, 302), (60, 210)]
        draw.polygon(octagon, fill=(*red, 255), outline=white)
        draw.line(octagon + [octagon[0]], fill=white, width=10)
        text_center(draw, (80, 132, 432, 380), text, fill=white, font_obj=font(72 if "BERHENTI" not in text else 50))
    else:
        raise ValueError(f"Unsupported symbol: {symbol}")

    return image, ocr_text, parameter_value


def make_background(rng: random.Random) -> Image.Image:
    palette = rng.choice(
        [
            ((142, 153, 145), (106, 118, 111)),
            ((184, 178, 166), (132, 126, 114)),
            ((124, 144, 156), (86, 103, 114)),
            ((138, 158, 132), (101, 118, 99)),
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


def augment_symbol(semantic_id: str, symbol: str, variant_index: int) -> tuple[Image.Image, str, str, str]:
    rng = random.Random(f"{STAGE_ID}:{semantic_id}:{variant_index}")
    sign, ocr_text, parameter_value = draw_symbol(symbol, rng)
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
        f"symbol={symbol}; size={target_size}; angle={angle:.2f}; "
        f"brightness={brightness:.2f}; contrast={contrast:.2f}; offset=({x},{y})"
    )
    return output, ocr_text, parameter_value, summary


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
            synthetic_id = f"S07-{counter:03d}_{semantic_id}"
            output_path = OUTPUT_ROOT / semantic_id / f"{synthetic_id}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image, ocr_text, parameter_value, transform_summary = augment_symbol(semantic_id, symbol, variant_index)
            image.save(output_path, quality=91)
            rows.append(
                {
                    "stage_id": STAGE_ID,
                    "synthetic_id": synthetic_id,
                    "semantic_sign_id": semantic_id,
                    "symbol_family": "regulatory_text_symbol",
                    "output_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                    "generation_status": "generated_candidate",
                    "review_status": "pending_stage_d_visual_qc",
                    "ocr_text": ocr_text,
                    "parameter_value": parameter_value,
                    "transform_summary": transform_summary,
                    "sha256": sha256(output_path),
                    "notes": "Project-owned generated regulatory/text candidate; not real camera coverage.",
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
    contact_sheets["_sample_all_candidates"] = render_sheet(rows, SHEET_ROOT / "_sample_all_candidates.jpg", max_rows=70)

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
            "These generated candidates close remaining regulatory/text gaps at "
            "candidate-count level. OCR text rows still require Stage D transcript QC."
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
