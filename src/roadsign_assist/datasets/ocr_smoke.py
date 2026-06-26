from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from roadsign_assist.paths import project_path

SAMPLES = (
    ("malay_speed", "HAD LAJU 50", "latin", Path("C:/Windows/Fonts/arialbd.ttf")),
    ("malay_stop", "BERHENTI", "latin", Path("C:/Windows/Fonts/arialbd.ttf")),
    ("english_speed", "SPEED LIMIT 80", "latin", Path("C:/Windows/Fonts/arialbd.ttf")),
    ("chinese_school", "学校区域", "chinese", Path("C:/Windows/Fonts/msyhbd.ttc")),
    ("chinese_ahead", "前方学校", "chinese", Path("C:/Windows/Fonts/msyhbd.ttc")),
)


def create_synthetic_ocr_smoke_set(
    output_root: str | Path = "data/processed/ocr_smoke",
) -> Path:
    root = project_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, str]] = []
    for sample_id, text, script, font_path in SAMPLES:
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
        font = ImageFont.truetype(str(font_path), 72)
        temporary = Image.new("RGB", (1, 1))
        bounds = ImageDraw.Draw(temporary).textbbox((0, 0), text, font=font)
        width = round(bounds[2] - bounds[0] + 80)
        height = round(bounds[3] - bounds[1] + 80)
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((40, 40 - bounds[1]), text, fill="black", font=font)
        path = root / f"{sample_id}.png"
        image.save(path)
        samples.append(
            {
                "sample_id": sample_id,
                "text": text,
                "script": script,
                "path": path.as_posix(),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "scope": "synthetic_multilingual_pipeline_smoke",
                "samples": samples,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
