from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from roadsign_assist.paths import project_path


def compare_classifier_runs(
    training_root: str | Path = "outputs/training",
    output_root: str | Path = "outputs/evaluation/classifier_comparison",
) -> dict[str, Any]:
    root = project_path(training_root)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
        validation_value = metrics.get("validation")
        validation: dict[str, Any] = (
            cast(dict[str, Any], validation_value)
            if isinstance(validation_value, dict)
            else metrics
        )
        evaluation_split = str(metrics.get("evaluation_split", "legacy_unspecified"))
        has_locked_test = evaluation_split == "test"
        configuration_value = metrics.get("configuration")
        configuration: dict[str, Any] = (
            cast(dict[str, Any], configuration_value)
            if isinstance(configuration_value, dict)
            else {}
        )
        rows.append(
            {
                "run": metrics_path.parent.name,
                "dataset_id": metrics.get("dataset_id", "legacy_unspecified"),
                "architecture": metrics["architecture"],
                "image_size": int(configuration.get("image_size", 0)),
                "batch_size": int(configuration.get("batch_size", 0)),
                "learning_rate": float(configuration.get("learning_rate", 0.0)),
                "weight_decay": float(configuration.get("weight_decay", 0.0)),
                "label_smoothing": float(configuration.get("label_smoothing", 0.0)),
                "workers": int(configuration.get("workers", 0)),
                "seed": int(configuration.get("seed", 0)),
                "experimental": bool(metrics["experimental"]),
                "train_samples": int(metrics["train_samples"]),
                "validation_samples": int(metrics["validation_samples"]),
                "test_samples": int(metrics["test_samples"]),
                "validation_accuracy": float(validation["accuracy"]),
                "validation_macro_f1_all_labels": float(
                    validation["macro_f1_all_labels"]
                ),
                "selective_coverage": (
                    float(validation["selective_coverage"])
                    if "selective_coverage" in validation
                    else None
                ),
                "selective_accuracy": (
                    float(validation["selective_accuracy"])
                    if validation.get("selective_accuracy") is not None
                    else None
                ),
                "accepted_correct_rate": (
                    float(validation["accepted_correct_rate"])
                    if "accepted_correct_rate" in validation
                    else None
                ),
                "validation_ece": float(validation["ece"]),
                "locked_test_evaluated": has_locked_test,
                "test_accuracy": float(metrics["accuracy"]) if has_locked_test else None,
                "test_macro_f1_all_labels": (
                    float(metrics["macro_f1_all_labels"]) if has_locked_test else None
                ),
                "temperature": float(metrics["temperature"]),
                "onnx_parity": bool(metrics["onnx_parity"]["passed"]),
            }
        )
    if not rows:
        raise FileNotFoundError("No completed classifier metrics were found")
    rows.sort(
        key=lambda row: float(row["validation_macro_f1_all_labels"]),
        reverse=True,
    )
    grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    group_fields = (
        "dataset_id",
        "architecture",
        "image_size",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "label_smoothing",
        "workers",
    )
    for row in rows:
        grouped_rows[tuple(row[field] for field in group_fields)].append(row)
    groups: list[dict[str, Any]] = []
    for key, members in grouped_rows.items():
        scores = [float(row["validation_macro_f1_all_labels"]) for row in members]
        seeds = sorted({int(row["seed"]) for row in members})
        representative = next(
            (row for row in members if int(row["seed"]) == 2513),
            max(members, key=lambda row: float(row["validation_macro_f1_all_labels"])),
        )
        group: dict[str, Any] = {
            str(field): value for field, value in zip(group_fields, key, strict=True)
        }
        group.update(
            {
                "runs": [str(row["run"]) for row in members],
                "seeds": seeds,
                "repeat_count": len(seeds),
                "mean_validation_macro_f1": statistics.fmean(scores),
                "sample_sd_validation_macro_f1": (
                    statistics.stdev(scores) if len(scores) > 1 else None
                ),
                "representative_run": representative["run"],
                "stable": len(seeds) >= 3,
            }
        )
        groups.append(group)
    groups.sort(
        key=lambda group: float(group["mean_validation_macro_f1"]),
        reverse=True,
    )
    stable_groups = [group for group in groups if bool(group["stable"])]
    selected_group = stable_groups[0] if stable_groups else None
    recommended_run = (
        str(selected_group["representative_run"])
        if selected_group is not None
        else str(rows[0]["run"])
    )
    output = project_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report: dict[str, Any] = {
        "schema_version": "3.0",
        "selection_metric": (
            "mean_validation_macro_f1_across_at_least_three_seeds"
            if selected_group is not None
            else "validation_macro_f1_all_labels"
        ),
        "validation_leader": rows[0]["run"],
        "best_run": recommended_run,
        "selected_stable_group": selected_group,
        "groups": groups,
        "runs": rows,
        "production_selection_allowed": False,
        "reason": (
            "Comparison uses validation metrics only and prefers the strongest mean among "
            "configurations repeated with at least three seeds. Promotion additionally "
            "requires a locked test evaluation, parity, app checks, and release review."
        ),
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
