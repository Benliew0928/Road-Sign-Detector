import json
from pathlib import Path
from typing import Any

import numpy as np

from roadsign_assist.classification.embedding import EmbeddingGate
from roadsign_assist.classification.onnx_backend import ONNXSignClassifier


class _Input:
    name = "image"


class _Session:
    def __init__(self, outputs: list[np.ndarray[Any, Any]]) -> None:
        self.outputs = outputs

    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(
        self,
        _output_names: Any,
        _inputs: dict[str, np.ndarray[Any, Any]],
    ) -> list[np.ndarray[Any, Any]]:
        return self.outputs


def test_embedding_gate_rejects_distance_and_label_disagreement() -> None:
    gate = EmbeddingGate.from_payload(
        {
            "prototype_labels": ["stop", "give_way"],
            "prototypes": [[1.0, 0.0], [0.0, 1.0]],
            "distance_threshold": 0.2,
            "require_label_agreement": True,
        }
    )
    accepted = gate.decide(np.asarray([1.0, 0.0]), "stop")
    assert accepted.accepted is True
    assert accepted.nearest_label == "stop"

    disagreement = gate.decide(np.asarray([0.0, 1.0]), "stop")
    assert disagreement.accepted is False
    assert "prototype_label_disagreement" in disagreement.reasons

    distant = gate.decide(np.asarray([1.0, 1.0]), "stop")
    assert distant.accepted is False
    assert "embedding_distance" in distant.reasons


def _classifier(
    tmp_path: Path,
    outputs: list[np.ndarray[Any, Any]],
    *,
    embedding: bool,
) -> ONNXSignClassifier:
    model_path = tmp_path / "classifier.onnx"
    labels_path = tmp_path / "labels.json"
    calibration_path = tmp_path / "calibration.json"
    model_path.write_bytes(b"model")
    labels_path.write_text(json.dumps(["stop", "give_way"]), encoding="utf-8")
    payload: dict[str, Any] = {
        "temperature": 1.0,
        "confidence_threshold": 0.72,
    }
    if embedding:
        payload["embedding_gate"] = {
            "prototype_labels": ["stop", "give_way"],
            "prototypes": [[1.0, 0.0], [0.0, 1.0]],
            "distance_threshold": 0.2,
            "require_label_agreement": True,
        }
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")
    classifier = ONNXSignClassifier(
        model_path,
        labels_path,
        calibration_path=calibration_path,
        image_size=8,
    )
    classifier._session = _Session(outputs)  # pyright: ignore[reportPrivateUsage]
    classifier._labels = ["stop", "give_way"]  # pyright: ignore[reportPrivateUsage]
    classifier._temperature = 1.0  # pyright: ignore[reportPrivateUsage]
    if embedding:
        classifier._embedding_gate = EmbeddingGate.from_payload(  # pyright: ignore[reportPrivateUsage]
            payload["embedding_gate"]
        )
    return classifier


def test_onnx_classifier_combines_confidence_and_embedding_rejection(
    tmp_path: Path,
) -> None:
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    classifier = _classifier(
        tmp_path,
        [
            np.asarray([[6.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        ],
        embedding=True,
    )
    result = classifier.classify(crop)
    assert result.semantic_sign_id == "unknown_sign"
    assert result.accepted is False
    assert result.nearest_prototype == "give_way"
    assert "prototype_label_disagreement" in result.rejection_reasons


def test_onnx_classifier_remains_compatible_with_logits_only_model(
    tmp_path: Path,
) -> None:
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    classifier = _classifier(
        tmp_path,
        [np.asarray([[6.0, 0.0]], dtype=np.float32)],
        embedding=False,
    )
    result = classifier.classify(crop)
    assert result.semantic_sign_id == "stop"
    assert result.accepted is True
    assert result.embedding_distance is None
