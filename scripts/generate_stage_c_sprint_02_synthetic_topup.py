from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "stage_c_sprint_02_synthetic_near_minimum_topup"
OUTPUT_ROOT = PROJECT_ROOT / "data/generated/stage_c_sprint_02_synthetic_topup"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_sprint_02_synthetic_candidates.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_sprint_02_synthetic_candidates.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_sprint_02_synthetic_candidates"

IMAGE_SIZE = 224


@dataclass(frozen=True)
class Seed:
    semantic_sign_id: str
    seed_id: str
    seed_path: str
    seed_source: str
    visual_qc_basis: str


SEEDS = [
    Seed(
        "side_road_right",
        "emtd_1af03976d2f4cb04_001",
        "data/processed/emtd_classification/train/side_road_right/emtd_1af03976d2f4cb04_001.jpg",
        "p5_cleaned_emtd_crop",
        "Selected from P5 contact sheet as a clear side-road-right crop.",
    ),
    Seed(
        "side_road_right",
        "emtd_205cc2c8e60c9eaa_001",
        "data/processed/emtd_classification/train/side_road_right/emtd_205cc2c8e60c9eaa_001.jpg",
        "p5_cleaned_emtd_crop",
        "Selected from P5 contact sheet as a clear side-road-right crop.",
    ),
    Seed(
        "side_road_right",
        "emtd_2f159e69b7244de0_001",
        "data/processed/emtd_classification/train/side_road_right/emtd_2f159e69b7244de0_001.jpg",
        "p5_cleaned_emtd_crop",
        "Selected from P5 contact sheet as a clear side-road-right crop.",
    ),
    Seed(
        "side_road_right",
        "emtd_a8b9cba0d160dc08_001",
        "data/processed/emtd_classification/train/side_road_right/emtd_a8b9cba0d160dc08_001.jpg",
        "p5_cleaned_emtd_crop",
        "Selected from P5 contact sheet as a clear side-road-right crop.",
    ),
    Seed(
        "side_road_right",
        "emtd_df521cafd2e7f940_001",
        "data/processed/emtd_classification/train/side_road_right/emtd_df521cafd2e7f940_001.jpg",
        "p5_cleaned_emtd_crop",
        "Selected from P5 contact sheet as a clear side-road-right crop.",
    ),
    Seed(
        "no_heavy_vehicle",
        "emtd_f064bdcb26a6bd99_001",
        "data/processed/emtd_classification/test/no_heavy_vehicle/emtd_f064bdcb26a6bd99_001.jpg",
        "p5_cleaned_emtd_crop",
        "Selected from P5 contact sheet as a visible no-heavy-vehicle crop.",
    ),
    Seed(
        "no_heavy_vehicle",
        "emtd_0f05a1cb8eeac810_001",
        "data/processed/emtd_classification/train/no_heavy_vehicle/emtd_0f05a1cb8eeac810_001.jpg",
        "p5_cleaned_emtd_crop",
        "Selected from P5 contact sheet as a visible no-heavy-vehicle crop.",
    ),
    Seed(
        "no_heavy_vehicle",
        "emtd_1ce62fadeb705de6_001",
        "data/processed/emtd_classification/train/no_heavy_vehicle/emtd_1ce62fadeb705de6_001.jpg",
        "p5_cleaned_emtd_crop",
        "Selected from P5 contact sheet as a visible no-heavy-vehicle crop.",
    ),
    Seed(
        "no_heavy_vehicle",
        "emtd_576aa01f8ed15bc9_001",
        "data/processed/emtd_classification/train/no_heavy_vehicle/emtd_576aa01f8ed15bc9_001.jpg",
        "p5_cleaned_emtd_crop",
        "Selected from P5 contact sheet as a visible no-heavy-vehicle crop.",
    ),
    Seed(
        "no_heavy_vehicle",
        "emtd_9b609598d3644b81_001",
        "data/processed/emtd_classification/train/no_heavy_vehicle/emtd_9b609598d3644b81_001.jpg",
        "p5_cleaned_emtd_crop",
        "Selected from P5 contact sheet as a visible no-heavy-vehicle crop.",
    ),
    Seed(
        "no_heavy_vehicle",
        "emtd_a38b0961a70badc9_001",
        "data/processed/emtd_classification/train/no_heavy_vehicle/emtd_a38b0961a70badc9_001.jpg",
        "p5_cleaned_emtd_crop",
        "Selected from P5 contact sheet as a visible no-heavy-vehicle crop.",
    ),
    Seed(
        "no_heavy_vehicle",
        "S01-010_lori_dilarang",
        "data/raw/local_collection/stage_c_sprint_01/wikimedia_commons/no_heavy_vehicle/raster/Lori_dilarang.png",
        "stage_c_sprint_01_commons_candidate",
        "Selected from Sprint 01 visual sheet as a clean no-heavy-vehicle reference.",
    ),
]


FIELDNAMES = [
    "stage_id",
    "synthetic_id",
    "semantic_sign_id",
    "seed_id",
    "seed_source",
    "seed_path",
    "output_path",
    "generation_status",
    "review_status",
    "transform_summary",
    "sha256",
    "visual_qc_basis",
    "notes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_background(rng: random.Random) -> Image.Image:
    base = rng.randint(120, 205)
    tint = (
        min(255, max(0, base + rng.randint(-16, 22))),
        min(255, max(0, base + rng.randint(-18, 18))),
        min(255, max(0, base + rng.randint(-18, 18))),
    )
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), tint)
    draw = ImageDraw.Draw(image)
    for _ in range(18):
        x1 = rng.randint(-40, IMAGE_SIZE)
        y1 = rng.randint(-40, IMAGE_SIZE)
        x2 = x1 + rng.randint(50, 160)
        y2 = y1 + rng.randint(8, 34)
        color = tuple(min(255, max(0, channel + rng.randint(-30, 30))) for channel in tint)
        draw.rectangle((x1, y1, x2, y2), fill=color)
    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.2)))


def prepare_seed(seed_path: Path) -> Image.Image:
    with Image.open(seed_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


def augment(seed: Seed, index: int) -> tuple[Image.Image, str]:
    rng = random.Random(f"{STAGE_ID}:{seed.seed_id}:{index}")
    source = prepare_seed(PROJECT_ROOT / seed.seed_path)
    background = make_background(rng).convert("RGBA")

    max_side = rng.randint(162, 204)
    source = ImageOps.contain(source, (max_side, max_side))

    brightness = rng.uniform(0.86, 1.16)
    contrast = rng.uniform(0.86, 1.18)
    color = rng.uniform(0.90, 1.12)
    source_rgb = source.convert("RGB")
    source_rgb = ImageEnhance.Brightness(source_rgb).enhance(brightness)
    source_rgb = ImageEnhance.Contrast(source_rgb).enhance(contrast)
    source_rgb = ImageEnhance.Color(source_rgb).enhance(color)

    alpha = source.getchannel("A")
    source = Image.merge("RGBA", (*source_rgb.split(), alpha))

    angle = rng.uniform(-7.5, 7.5)
    source = source.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    if rng.random() < 0.35:
        source = source.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.45)))

    x = (IMAGE_SIZE - source.width) // 2 + rng.randint(-10, 10)
    y = (IMAGE_SIZE - source.height) // 2 + rng.randint(-10, 10)
    shadow = Image.new("RGBA", source.size, (0, 0, 0, 0))
    shadow_alpha = source.getchannel("A").filter(ImageFilter.GaussianBlur(radius=3))
    shadow.putalpha(shadow_alpha.point(lambda value: int(value * 0.22)))
    background.alpha_composite(shadow, (x + 4, y + 4))
    background.alpha_composite(source, (x, y))

    output = background.convert("RGB")
    if rng.random() < 0.5:
        output = output.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.08, 0.22)))

    summary = (
        f"size={max_side}; angle={angle:.2f}; brightness={brightness:.2f}; "
        f"contrast={contrast:.2f}; color={color:.2f}; offset=({x},{y})"
    )
    return output, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def render_sheet(rows: list[dict[str, str]], output_path: Path) -> str:
    columns = 4
    tile_width = 250
    tile_height = 245
    label_height = 54
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
        draw.text((x + 7, y + tile_height - label_height + 25), row["semantic_sign_id"], fill=(180, 198, 190))
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
    for index, seed in enumerate(SEEDS, start=1):
        source_path = PROJECT_ROOT / seed.seed_path
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        synthetic_id = f"S02-{index:03d}_{seed.semantic_sign_id}"
        output_path = OUTPUT_ROOT / seed.semantic_sign_id / f"{synthetic_id}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image, transform_summary = augment(seed, index)
        image.save(output_path, quality=91)
        rows.append(
            {
                "stage_id": STAGE_ID,
                "synthetic_id": synthetic_id,
                "semantic_sign_id": seed.semantic_sign_id,
                "seed_id": seed.seed_id,
                "seed_source": seed.seed_source,
                "seed_path": seed.seed_path,
                "output_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                "generation_status": "generated_candidate",
                "review_status": "pending_stage_d_visual_qc",
                "transform_summary": transform_summary,
                "sha256": sha256(output_path),
                "visual_qc_basis": seed.visual_qc_basis,
                "notes": "Synthetic top-up candidate; not real-world coverage and not final training data until Stage D/E.",
            }
        )

    write_csv(MANIFEST_PATH, rows)

    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["semantic_sign_id"], []).append(row)
    contact_sheets = {
        label: render_sheet(label_rows, SHEET_ROOT / f"{label}.jpg")
        for label, label_rows in sorted(by_class.items())
    }
    contact_sheets["_all_candidates"] = render_sheet(rows, SHEET_ROOT / "_all_candidates.jpg")

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
        "status": "generated_candidates_pending_stage_d_qc",
        "important_note": (
            "These are synthetic top-up candidates created from visually selected seeds. "
            "They can help classifier robustness but do not replace real local/demo photos."
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
