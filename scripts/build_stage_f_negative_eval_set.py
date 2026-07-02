from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def project_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain an images directory: {image_path}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        coordinates = values[1:]
        if len(coordinates) < 6 or len(coordinates) % 2:
            continue
        xs = [coordinates[index] * width for index in range(0, len(coordinates), 2)]
        ys = [coordinates[index] * height for index in range(1, len(coordinates), 2)]
        boxes.append(Box(min(xs), min(ys), max(xs), max(ys)))
    return boxes


def intersection(first: Box, second: Box) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def has_sign_overlap(candidate: Box, signs: list[Box], *, max_overlap_ratio: float) -> bool:
    for sign in signs:
        cx, cy = sign.center
        if candidate.x1 <= cx <= candidate.x2 and candidate.y1 <= cy <= candidate.y2:
            return True
        if sign.area and intersection(candidate, sign) / sign.area > max_overlap_ratio:
            return True
    return False


def dataset_images(data_yaml: Path, splits: list[str]) -> list[Path]:
    payload: dict[str, Any] = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(payload["path"]))
    images: list[Path] = []
    for split in splits:
        split_key = "val" if split == "validation" else split
        relative = payload.get(split_key)
        if not relative:
            continue
        images.extend(
            sorted(
                path
                for path in (root / str(relative)).rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        )
    return images


def crop_candidates(
    *,
    width: int,
    height: int,
    rng: random.Random,
    attempts: int,
) -> list[Box]:
    candidates: list[Box] = []
    min_side = min(width, height)
    for _ in range(attempts):
        crop_width = int(rng.uniform(0.32, 0.72) * width)
        crop_height = int(crop_width * rng.uniform(0.65, 1.35))
        crop_width = max(96, min(crop_width, width))
        crop_height = max(96, min(crop_height, height))
        if crop_width > width or crop_height > height:
            crop_width = min(width, min_side)
            crop_height = min(height, min_side)
        x1 = rng.randint(0, max(0, width - crop_width))
        y1 = rng.randint(0, max(0, height - crop_height))
        candidates.append(Box(x1, y1, x1 + crop_width, y1 + crop_height))
    return candidates


def build_negative_set(args: argparse.Namespace) -> dict[str, Any]:
    data_yaml = project_path(args.data)
    output_root = project_path(args.output)
    image_output = output_root / "images"
    label_output = output_root / "labels"
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    source_images = dataset_images(data_yaml, args.splits)
    rng.shuffle(source_images)

    rows: list[dict[str, str]] = []
    per_source_counts: dict[Path, int] = {}
    for source_path in source_images:
        if len(rows) >= args.target:
            break
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        boxes = read_boxes(label_path_for(source_path), width, height)
        if not boxes:
            continue
        count_for_source = per_source_counts.get(source_path, 0)
        for candidate in crop_candidates(
            width=width,
            height=height,
            rng=rng,
            attempts=args.attempts_per_image,
        ):
            if count_for_source >= args.max_per_source or len(rows) >= args.target:
                break
            if has_sign_overlap(candidate, boxes, max_overlap_ratio=args.max_overlap_ratio):
                continue
            crop = image.crop(
                (
                    int(candidate.x1),
                    int(candidate.y1),
                    int(candidate.x2),
                    int(candidate.y2),
                )
            )
            sample_id = f"stage_f_neg_{len(rows) + 1:04d}"
            destination = image_output / f"{sample_id}.jpg"
            crop.save(destination, quality=92)
            (label_output / f"{sample_id}.txt").write_text("", encoding="utf-8")
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": project_rel(destination),
                    "label_path": project_rel(label_output / f"{sample_id}.txt"),
                    "source_image": project_rel(source_path),
                    "source_label": project_rel(label_path_for(source_path)),
                    "crop_x1": str(int(candidate.x1)),
                    "crop_y1": str(int(candidate.y1)),
                    "crop_x2": str(int(candidate.x2)),
                    "crop_y2": str(int(candidate.y2)),
                    "source_width": str(width),
                    "source_height": str(height),
                }
            )
            count_for_source += 1
            per_source_counts[source_path] = count_for_source

    manifest = output_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": "1.0",
        "source_dataset": project_rel(data_yaml),
        "source_splits": args.splits,
        "samples": len(rows),
        "target": args.target,
        "purpose": "Stage F no-sign detector false-positive evaluation only",
        "training_use": "not used for positive sign classifier training",
        "max_overlap_ratio": args.max_overlap_ratio,
        "seed": args.seed,
        "manifest": project_rel(manifest),
    }
    (output_root / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Stage F no-sign negative crops from full-frame detector data."
    )
    parser.add_argument("--data", default="data/processed/emtd_segmentation/data.yaml")
    parser.add_argument("--output", default="data/processed/stage_f_negative_eval")
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2513)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--attempts-per-image", type=int, default=80)
    parser.add_argument("--max-per-source", type=int, default=2)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.01)
    parser.add_argument("--clean", action="store_true")
    return parser


def main() -> int:
    metadata = build_negative_set(build_parser().parse_args())
    print(
        "Stage F negative set complete: "
        f"samples={metadata['samples']}, manifest={metadata['manifest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
