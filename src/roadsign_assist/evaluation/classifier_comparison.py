from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from roadsign_assist.paths import project_path


def compare_classifier_runs(
    training_root: str | Path = "outputs/training",
    output_root: str | Path = "outputs/evaluation/classifier_comparison",
) -> dict[str, Any]:
    root = project_path(training_root)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("emtd_classifier_*/metrics.json")):
        metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": metrics_path.parent.name,
                "architecture": metrics["architecture"],
                "experimental": bool(metrics["experimental"]),
                "train_samples": int(metrics["train_samples"]),
                "validation_samples": int(metrics["validation_samples"]),
                "test_samples": int(metrics["test_samples"]),
                "test_accuracy": float(metrics["accuracy"]),
                "selective_coverage": (
                    float(metrics["selective_coverage"])
                    if "selective_coverage" in metrics
                    else None
                ),
                "selective_accuracy": (
                    float(metrics["selective_accuracy"])
                    if metrics.get("selective_accuracy") is not None
                    else None
                ),
                "accepted_correct_rate": (
                    float(metrics["accepted_correct_rate"])
                    if "accepted_correct_rate" in metrics
                    else None
                ),
                "test_macro_f1_observed": float(metrics["macro_f1_observed"]),
                "test_macro_f1_all_labels": float(metrics["macro_f1_all_labels"]),
                "test_label_coverage": float(metrics["evaluation_label_coverage"]),
                "ece": float(metrics["ece"]),
                "temperature": float(metrics["temperature"]),
                "onnx_parity": bool(metrics["onnx_parity"]["passed"]),
            }
        )
    if not rows:
        raise FileNotFoundError("No completed EMTD classifier metrics were found")
    rows.sort(
        key=lambda row: float(row["test_macro_f1_all_labels"]),
        reverse=True,
    )
    output = project_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "selection_metric": "test_macro_f1_all_labels",
        "best_run": rows[0]["run"],
        "runs": rows,
        "production_selection_allowed": False,
        "reason": "Source class mappings and annotations remain single-review drafts.",
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
