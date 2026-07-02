import json
from pathlib import Path

import numpy as np
import pytest

from roadsign_assist.classification.folder_training import (
    compare_classifier_outputs,
    should_export_classifier_experimentally,
    validate_classifier_dataset,
)


def test_classifier_training_requires_review_or_experimental_override(
    tmp_path: Path,
) -> None:
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "annotation_status": "source_boxes_unreviewed",
                "coursework_images_included": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="experimental override"):
        validate_classifier_dataset(tmp_path, allow_unreviewed_experiment=False)
    metadata = validate_classifier_dataset(
        tmp_path,
        allow_unreviewed_experiment=True,
    )
    assert metadata["annotation_status"] == "source_boxes_unreviewed"


def test_experimental_override_protects_approved_classifier_export() -> None:
    approved_metadata = {"annotation_status": "approved"}
    unreviewed_metadata = {"annotation_status": "source_boxes_unreviewed"}

    assert (
        should_export_classifier_experimentally(
            approved_metadata,
            allow_unreviewed_experiment=False,
        )
        is False
    )
    assert (
        should_export_classifier_experimentally(
            approved_metadata,
            allow_unreviewed_experiment=True,
        )
        is True
    )
    assert (
        should_export_classifier_experimentally(
            unreviewed_metadata,
            allow_unreviewed_experiment=False,
        )
        is True
    )


def test_classifier_output_parity() -> None:
    pytorch_logits = np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    onnx_logits = pytorch_logits + np.float32(1e-5)
    parity = compare_classifier_outputs(pytorch_logits, onnx_logits)
    assert parity["passed"] is True
    assert parity["top1_agreement"] == 1.0
    assert parity["acceptance_agreement"] == 1.0


def test_classifier_output_parity_allows_one_borderline_disagreement() -> None:
    pytorch_logits = np.tile(np.asarray([[4.0, 0.1]], dtype=np.float32), (1000, 1))
    onnx_logits = pytorch_logits.copy()
    pytorch_logits[0] = np.asarray([0.01, 0.011], dtype=np.float32)
    onnx_logits[0] = np.asarray([0.011, 0.01], dtype=np.float32)

    parity = compare_classifier_outputs(pytorch_logits, onnx_logits)

    assert parity["passed"] is True
    assert parity["top1_agreement"] == 0.999


def test_classifier_output_parity_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_classifier_outputs(
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
