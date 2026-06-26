import json
from pathlib import Path

from roadsign_assist.evaluation.classifier_safety import (
    evaluate_critical_class_recall,
)


def test_classifier_safety_report_uses_confusion_matrix_support(tmp_path: Path) -> None:
    labels = ["stop", "give_way"]
    metrics = {"confusion_matrix": [[3, 1], [1, 3]]}
    metrics_path = tmp_path / "metrics.json"
    labels_path = tmp_path / "labels.json"
    output_path = tmp_path / "report.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    labels_path.write_text(json.dumps(labels), encoding="utf-8")

    report = evaluate_critical_class_recall(
        metrics_path,
        labels_path,
        output_path,
    )
    assert report["critical_classes"] == 2
    assert report["macro_recall_observed"] == 0.75
    assert report["micro_recall_observed"] == 0.75
    assert report["all_observed_classes_meet_target"] is False
