from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2

from roadsign_assist.baseline.features import extract_feature_sets, normalized_crop
from roadsign_assist.baseline.pipeline import process_image, read_bgr
from roadsign_assist.config import load_yaml
from roadsign_assist.paths import OFFICIAL_ROOT, project_path


def _read_inputs(path: Path) -> list[tuple[str, Path]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "relative_path" not in rows[0]:
            raise ValueError("CSV input must contain relative_path")
        root = OFFICIAL_ROOT / "assignment_images"
        return [
            (row.get("image_id", Path(row["relative_path"]).stem), root / row["relative_path"])
            for row in rows
        ]

    inputs: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip().strip('"')
        if not value:
            continue
        image_path = project_path(value)
        inputs.append((image_path.stem, image_path))
    return inputs


def _annotate(image: Any, candidates: tuple[Any, ...]) -> Any:
    annotated = image.copy()
    colors = {"red": (30, 50, 235), "blue": (235, 130, 30), "yellow": (30, 220, 230)}
    for index, candidate in enumerate(candidates, start=1):
        bbox = candidate.bbox
        color = colors.get(candidate.color, (50, 210, 80))
        cv2.rectangle(annotated, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 2)
        label = f"{index} {candidate.color} {candidate.shape_label} {candidate.score:.2f}"
        cv2.putText(
            annotated,
            label,
            (bbox.x, max(18, bbox.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_baseline_batch(input_path: str | Path, output_root: str | Path) -> None:
    config = load_yaml("configs/baseline/default.yaml")
    inputs = _read_inputs(project_path(input_path))
    if not inputs:
        raise ValueError("Input list is empty")

    output = project_path(output_root)
    for directory in ("masks", "crops", "annotated", "features"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    image_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for image_id, image_path in inputs:
        image = read_bgr(image_path)
        result = process_image(
            image,
            image_id=image_id,
            image_path=str(image_path),
            config=config,
        )
        for color, mask in result.masks.items():
            cv2.imwrite(str(output / "masks" / f"{image_id}__{color}.png"), mask)
        cv2.imwrite(
            str(output / "annotated" / f"{image_id}.jpg"), _annotate(image, result.candidates)
        )

        for index, candidate in enumerate(result.candidates, start=1):
            crop = normalized_crop(image, candidate)
            crop_path = output / "crops" / f"{image_id}__{index:02d}.png"
            cv2.imwrite(str(crop_path), crop)
            feature_sets = extract_feature_sets(image, candidate)
            feature_path = output / "features" / f"{image_id}__{index:02d}.npz"
            import numpy as np

            np.savez_compressed(
                str(feature_path),
                hog=feature_sets["hog"],
                hog_hsv=feature_sets["hog_hsv"],
                all_handcrafted=feature_sets["all_handcrafted"],
            )
            candidate_rows.append(
                {
                    "image_id": image_id,
                    "candidate_index": index,
                    **candidate.serializable(),
                    "crop_path": crop_path.relative_to(output).as_posix(),
                    "feature_path": feature_path.relative_to(output).as_posix(),
                }
            )
        image_rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "width": result.width,
                "height": result.height,
                "candidate_count": len(result.candidates),
                "runtime_ms": round(result.runtime_ms, 3),
            }
        )

    _write_csv(output / "images.csv", image_rows)
    _write_jsonl(output / "candidates.jsonl", candidate_rows)
    print(f"Processed {len(image_rows)} images into {output}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
