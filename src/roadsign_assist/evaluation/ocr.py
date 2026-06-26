from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

import cv2

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.ocr.engine import MultilingualOCREngine
from roadsign_assist.ocr.normalization import normalize_ocr_text
from roadsign_assist.paths import project_path


def edit_distance(reference: str, prediction: str) -> int:
    previous = list(range(len(prediction) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_character in enumerate(prediction, start=1):
            substitution_cost = int(reference_character != prediction_character)
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    normalized_reference = normalize_ocr_text(reference).casefold()
    normalized_prediction = normalize_ocr_text(prediction).casefold()
    return edit_distance(normalized_reference, normalized_prediction) / max(
        1, len(normalized_reference)
    )


def evaluate_ocr_manifest(
    manifest_path: str | Path,
    output_root: str | Path = "outputs/evaluation/ocr_smoke",
) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(project_path(manifest_path).read_text(encoding="utf-8"))
    output = project_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    engine = MultilingualOCREngine()
    if not engine.available:
        raise RuntimeError("Frozen local OCR assets are unavailable")

    rows: list[dict[str, Any]] = []
    for sample in manifest["samples"]:
        image_path = project_path(sample["path"])
        raw_image: Any = cv2.imread(str(image_path))
        if raw_image is None:
            raise ValueError(f"Unable to read OCR sample: {image_path}")
        image = cast(UInt8Image, raw_image)
        started = time.perf_counter()
        result = engine.recognize(image)
        latency_ms = (time.perf_counter() - started) * 1000
        expected = normalize_ocr_text(str(sample["text"]))
        predicted = normalize_ocr_text(result.text)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "script": sample["script"],
                "expected": expected,
                "predicted": predicted,
                "exact_match": expected.casefold() == predicted.casefold(),
                "cer": character_error_rate(expected, predicted),
                "confidence": result.confidence,
                "latency_ms": latency_ms,
                "numeric_value": result.numeric_value,
                "unit": result.unit,
            }
        )

    ordered_latencies = [float(row["latency_ms"]) for row in rows]
    latencies = sorted(ordered_latencies)
    warm_latencies = ordered_latencies[1:]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "scope": manifest["scope"],
        "samples": len(rows),
        "exact_match_rate": mean(float(bool(row["exact_match"])) for row in rows),
        "mean_cer": mean(float(row["cer"]) for row in rows),
        "cold_start_latency_ms": ordered_latencies[0],
        "mean_latency_including_cold_start_ms": mean(latencies),
        "warm_mean_latency_ms": mean(warm_latencies) if warm_latencies else None,
        "warm_median_latency_ms": median(warm_latencies) if warm_latencies else None,
        "maximum_latency_ms": max(latencies),
        "offline_assets": True,
        "claims_real_road_accuracy": False,
        "predictions": rows,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
