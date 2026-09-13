from pathlib import Path
from typing import cast

import cv2
import pytest

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.classification.onnx_backend import ONNXSignClassifier

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data/processed/classifier_production_78_v3_20260829/test"


def _first_image(label: str) -> UInt8Image:
    path = next(path for path in (DATA / label).iterdir() if path.is_file())
    image = cv2.imread(str(path))
    assert image is not None
    return cast(UInt8Image, image)


@pytest.mark.model
def test_embedding_safety_preview_rejects_reproduced_confident_errors() -> None:
    classifier = ONNXSignClassifier(
        PROJECT
        / "models/candidates/pb_v3_effnetv2m_320_s2513/sign_classifier.embedding.onnx",
        PROJECT
        / "models/candidates/pb_v3_effnetv2m_320_s2513/sign_classifier.labels.json",
        calibration_path=PROJECT
        / "models/candidates/pb_v3_effnetv2m_320_s2513/sign_classifier.embedding.calibration.json",
        image_size=320,
        providers=["CPUExecutionProvider"],
    )

    upright_turn_left = classifier.classify(_first_image("turn_left"))
    wrong_upright_turn_right = classifier.classify(_first_image("turn_right"))
    rotated_turn_left = classifier.classify(
        cast(UInt8Image, cv2.rotate(_first_image("turn_left"), cv2.ROTATE_90_CLOCKWISE))
    )
    rotated_keep_right = classifier.classify(
        cast(UInt8Image, cv2.rotate(_first_image("keep_right"), cv2.ROTATE_90_CLOCKWISE))
    )
    rotated_stop = classifier.classify(
        cast(UInt8Image, cv2.rotate(_first_image("stop"), cv2.ROTATE_90_CLOCKWISE))
    )

    assert upright_turn_left.semantic_sign_id == "turn_left"
    assert upright_turn_left.accepted is True
    for rejected in (
        wrong_upright_turn_right,
        rotated_turn_left,
        rotated_keep_right,
        rotated_stop,
    ):
        assert rejected.semantic_sign_id == "unknown_sign"
        assert rejected.accepted is False
        assert "embedding_distance" in rejected.rejection_reasons
