from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, cast

import cv2

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.inference.engine import InferenceEngine, annotate_frame
from roadsign_assist.paths import OFFICIAL_ROOT, project_path


def evaluate_coursework_images(
    config_path: str | Path,
    output_root: str | Path = "outputs/evaluation/coursework_experimental",
) -> dict[str, Any]:
    output = project_path(output_root)
    annotated_root = output / "annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)
    manifest_path = project_path("data/manifests/official_images.csv")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 84:
        raise ValueError(f"Expected 84 coursework images, found {len(rows)}")
    review_manifest = project_path("data/manifests/coursework_manifest.csv")
    with review_manifest.open(newline="", encoding="utf-8") as handle:
        reviewed_rows = list(csv.DictReader(handle))
    expected_by_id = {
        row["image_id"]: row["semantic_sign_id"].strip()
        for row in reviewed_rows
        if row["semantic_sign_id"].strip()
    }

    engine = InferenceEngine(config_path)
    warmup = engine.warmup()
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        path = OFFICIAL_ROOT / "assignment_images" / row["relative_path"]
        raw_image: Any = cv2.imread(str(path))
        if raw_image is None:
            raise ValueError(f"Unable to read coursework image: {path}")
        image = cast(UInt8Image, raw_image)
        started = time.perf_counter()
        result = engine.new_session().process_frame(image, assume_stable=True)
        elapsed_ms = (time.perf_counter() - started) * 1000
        events = result.events
        predictions = sorted({event.semantic_sign_id for event in events})
        ocr_values = [event.ocr.text for event in events if event.ocr.text]
        expected = expected_by_id.get(row["image_id"], "")
        results.append(
            {
                "image_id": row["image_id"],
                "relative_path": row["relative_path"],
                "mode": result.mode,
                "events": len(events),
                "semantic_predictions": " ".join(predictions),
                "expected_semantic_draft": expected,
                "semantic_match_draft": (expected in predictions if expected else ""),
                "ocr": " | ".join(ocr_values),
                "runtime_ms": elapsed_ms,
                "under_two_seconds": elapsed_ms < 2000,
            }
        )
        annotated = annotate_frame(image, result)
        cv2.imwrite(
            str(annotated_root / f"{row['image_id']}.jpg"),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        print(f"[{index}/{len(rows)}] {row['image_id']}: {len(events)} events, {elapsed_ms:.1f} ms")

    csv_path = output / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    runtimes = sorted(float(row["runtime_ms"]) for row in results)
    p95_index = min(len(runtimes) - 1, round(0.95 * (len(runtimes) - 1)))
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "scope": "84_official_coursework_images_external_acceptance",
        "config": str(config_path),
        "images": len(results),
        "completed": len(results),
        "mean_runtime_ms": mean(runtimes),
        "p95_runtime_ms": runtimes[p95_index],
        "maximum_runtime_ms": max(runtimes),
        "under_two_seconds": sum(bool(row["under_two_seconds"]) for row in results),
        "images_with_events": sum(int(row["events"]) > 0 for row in results),
        "draft_semantic_scored_images": sum(
            bool(row["expected_semantic_draft"]) for row in results
        ),
        "draft_semantic_correct_images": sum(
            row["semantic_match_draft"] is True for row in results
        ),
        "draft_semantic_exact_match_rate": (
            sum(row["semantic_match_draft"] is True for row in results)
            / max(
                1,
                sum(bool(row["expected_semantic_draft"]) for row in results),
            )
        ),
        "draft_semantic_warning": (
            "Coursework labels and EMTD mappings are single-review drafts; "
            "this is diagnostic evidence, not final ground truth."
        ),
        "warmup": warmup,
        "development_machine_only": True,
        "lab_machine_benchmark_pending": True,
        "experimental": bool(engine.config.get("experimental", False)),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
