import json
from pathlib import Path

import numpy as np
import pytest

from roadsign_assist.classification.folder_training import (
    compare_classifier_outputs,
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


def test_classifier_output_parity() -> None:
    pytorch_logits = np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    onnx_logits = pytorch_logits + np.float32(1e-5)
    parity = compare_classifier_outputs(pytorch_logits, onnx_logits)
    assert parity["passed"] is True
    assert parity["top1_agreement"] == 1.0
    assert parity["acceptance_agreement"] == 1.0


def test_classifier_output_parity_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_classifier_outputs(
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
