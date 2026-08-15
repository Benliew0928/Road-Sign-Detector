import json
from pathlib import Path

from roadsign_assist.evaluation.classifier_comparison import (
    compare_classifier_runs,
)


def test_classifier_comparison_selects_validation_macro_f1(tmp_path: Path) -> None:
    training = tmp_path / "training"
    for name, score in (("clean_a", 0.5), ("clean_b", 0.7)):
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
                    "evaluation_split": "validation",
                    "validation": {
                        "accuracy": score,
                        "selective_coverage": 0.8,
                        "selective_accuracy": score,
                        "accepted_correct_rate": score * 0.8,
                        "macro_f1_all_labels": score,
                        "ece": 0.1,
                    },
                    "temperature": 1.0,
                    "onnx_parity": {"passed": True},
                }
            ),
            encoding="utf-8",
        )
    report = compare_classifier_runs(training, tmp_path / "comparison")
    assert report["best_run"] == "clean_b"
    assert report["selection_metric"] == "validation_macro_f1_all_labels"
    assert (tmp_path / "comparison/comparison.csv").exists()


def test_classifier_comparison_prefers_stable_group_mean(tmp_path: Path) -> None:
    training = tmp_path / "training"
    runs = (
        ("single_peak", 0.99, 2513, 256),
        ("stable_a", 0.94, 2513, 224),
        ("stable_b", 0.96, 1337, 224),
        ("stable_c", 0.95, 2026, 224),
    )
    for name, score, seed, image_size in runs:
        run = training / name
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(
            json.dumps(
                {
                    "dataset_id": "clean",
                    "architecture": "efficientnet_v2_s",
                    "experimental": True,
                    "train_samples": 10,
                    "validation_samples": 3,
                    "test_samples": 3,
                    "evaluation_split": "validation",
                    "configuration": {
                        "image_size": image_size,
                        "batch_size": 32,
                        "learning_rate": 3e-4,
                        "weight_decay": 1e-4,
                        "label_smoothing": 0.05,
                        "workers": 0,
                        "seed": seed,
                    },
                    "validation": {
                        "accuracy": score,
                        "selective_coverage": 0.8,
                        "selective_accuracy": score,
                        "accepted_correct_rate": score * 0.8,
                        "macro_f1_all_labels": score,
                        "ece": 0.1,
                    },
                    "temperature": 1.0,
                    "onnx_parity": {"passed": True},
                }
            ),
            encoding="utf-8",
        )

    report = compare_classifier_runs(training, tmp_path / "comparison")

    assert report["validation_leader"] == "single_peak"
    assert report["best_run"] == "stable_a"
    assert report["selection_metric"] == (
        "mean_validation_macro_f1_across_at_least_three_seeds"
    )
