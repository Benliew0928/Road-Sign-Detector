import json
from pathlib import Path

from roadsign_assist.evaluation.classifier_comparison import (
    compare_classifier_runs,
)


def test_classifier_comparison_selects_macro_f1(tmp_path: Path) -> None:
    training = tmp_path / "training"
    for name, score in (("emtd_classifier_a", 0.5), ("emtd_classifier_b", 0.7)):
        run = training / name
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(
            json.dumps(
                {
                    "architecture": "mobilenet_v3_large",
                    "experimental": True,
                    "train_samples": 10,
                    "validation_samples": 3,
                    "test_samples": 3,
                    "accuracy": score,
                    "selective_coverage": 0.8,
                    "selective_accuracy": score,
                    "accepted_correct_rate": score * 0.8,
                    "macro_f1_observed": score,
                    "macro_f1_all_labels": score,
                    "evaluation_label_coverage": 1.0,
                    "ece": 0.1,
                    "temperature": 1.0,
                    "onnx_parity": {"passed": True},
                }
            ),
            encoding="utf-8",
        )
    report = compare_classifier_runs(training, tmp_path / "comparison")
    assert report["best_run"] == "emtd_classifier_b"
    assert (tmp_path / "comparison/comparison.csv").exists()
