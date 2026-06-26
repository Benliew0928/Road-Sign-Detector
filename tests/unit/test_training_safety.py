import json
from pathlib import Path

import pytest

from roadsign_assist.detection.training import (
    compare_metric_parity,
    validate_training_data,
)


def test_training_refuses_unreviewed_data_by_default(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("{}\n", encoding="utf-8")
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "annotation_status": "source_boxes_unreviewed",
                "coursework_images_included": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="annotation status"):
        validate_training_data(data_yaml, allow_unreviewed_experiment=False)
    metadata = validate_training_data(data_yaml, allow_unreviewed_experiment=True)
    assert metadata["coursework_images_included"] == 0


def test_training_always_refuses_coursework_images(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("{}\n", encoding="utf-8")
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "annotation_status": "approved",
                "coursework_images_included": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Coursework acceptance images"):
        validate_training_data(data_yaml, allow_unreviewed_experiment=True)


def test_metric_parity_reports_differences() -> None:
    parity = compare_metric_parity(
        {"mAP50": 0.80, "label": "pytorch"},
        {"mAP50": 0.79, "label": "onnx"},
        tolerance=0.02,
    )
    assert parity["passed"] is True
    assert abs(parity["maximum_absolute_difference"] - 0.01) < 1e-9


def test_metric_parity_requires_shared_scalars() -> None:
    with pytest.raises(ValueError, match="No shared scalar metrics"):
        compare_metric_parity(
            {"backend": "pytorch"},
            {"backend": "onnx"},
            tolerance=0.02,
        )
