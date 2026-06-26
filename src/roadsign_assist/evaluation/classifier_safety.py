from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roadsign_assist.catalogue.models import Severity
from roadsign_assist.catalogue.repository import catalogue_by_id
from roadsign_assist.paths import project_path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(project_path(".")))
    except ValueError:
        return str(path)


def evaluate_critical_class_recall(
    metrics_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    metrics_file = project_path(metrics_path)
    labels_file = project_path(labels_path)
    metrics: dict[str, Any] = json.loads(metrics_file.read_text(encoding="utf-8"))
    labels: list[str] = json.loads(labels_file.read_text(encoding="utf-8"))
    matrix: list[list[int]] = metrics["confusion_matrix"]
    if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
        raise ValueError("Confusion matrix dimensions do not match classifier labels")

    catalogue = catalogue_by_id()
    rows: list[dict[str, Any]] = []
    observed_recalls: list[float] = []
    for index, label in enumerate(labels):
        definition = catalogue.get(label)
        if definition is None or definition.severity is not Severity.CRITICAL:
            continue
        support = sum(matrix[index])
        recall = matrix[index][index] / support if support else None
        if recall is not None:
            observed_recalls.append(recall)
        rows.append(
            {
                "semantic_sign_id": label,
                "support": support,
                "true_positive": matrix[index][index],
                "recall": recall,
                "meets_90_percent_target": bool(recall is not None and recall >= 0.90),
            }
        )

    micro_support = sum(int(row["support"]) for row in rows)
    micro_true_positive = sum(int(row["true_positive"]) for row in rows)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "source_metrics": _display_path(metrics_file),
        "source_labels": _display_path(labels_file),
        "target_recall": 0.90,
        "critical_classes": len(rows),
        "observed_critical_classes": len(observed_recalls),
        "macro_recall_observed": (
            sum(observed_recalls) / len(observed_recalls) if observed_recalls else None
        ),
        "micro_recall_observed": (micro_true_positive / micro_support if micro_support else None),
        "all_observed_classes_meet_target": all(
            bool(row["meets_90_percent_target"]) for row in rows if int(row["support"]) > 0
        ),
        "classes": rows,
    }
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
