from __future__ import annotations

# pyright: reportPrivateUsage=false
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.detection.ultralytics_backend import (
    UltralyticsDetector,
    UltralyticsSegmenter,
)
from roadsign_assist.inference.engine import InferenceEngine


class _Tensor:
    def __init__(self, value: np.ndarray[Any, Any]) -> None:
        self.value = value

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.value


class _FakeModel:
    def __init__(
        self,
        *,
        masks: object | None,
        xyxy: np.ndarray[Any, Any] | None = None,
        confidence: np.ndarray[Any, Any] | None = None,
    ) -> None:
        self.masks = masks
        self.xyxy = xyxy
        self.confidence = confidence
        self.predict_calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[object]:
        self.predict_calls.append(kwargs)
        boxes = SimpleNamespace(
            xyxy=_Tensor(
                self.xyxy
                if self.xyxy is not None
                else np.asarray([[1.0, 2.0, 11.0, 12.0]], dtype=np.float32)
            ),
            conf=_Tensor(
                self.confidence
                if self.confidence is not None
                else np.asarray([0.91], dtype=np.float32)
            ),
        )
        return [SimpleNamespace(boxes=boxes, masks=self.masks)]


def test_detection_task_parses_boxes_without_masks(tmp_path: Path) -> None:
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"test")
    detector = UltralyticsDetector(model_path, task="detect", profile_name="cpu")
    model = _FakeModel(masks=None)
    detector._model = model

    detections = detector.detect(cast(UInt8Image, np.zeros((20, 20, 3), dtype=np.uint8)))

    assert detector.task == "detect"
    assert detector.mask_capable is False
    assert detector.name == "ultralytics-detect:detector.onnx"
    assert len(detections) == 1
    assert detections[0].mask is None
    assert detections[0].bbox.x2 == 11.0
    assert model.predict_calls[0]["imgsz"] == 640


def test_warmup_passes_configured_image_size_explicitly(tmp_path: Path) -> None:
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"test")
    detector = UltralyticsDetector(model_path, image_size=960)
    model = _FakeModel(masks=None)
    detector._model = model

    assert detector.warmup() is True
    assert model.predict_calls[0]["imgsz"] == 960


def test_segmentation_alias_remains_mask_compatible(tmp_path: Path) -> None:
    model_path = tmp_path / "segmenter.pt"
    model_path.write_bytes(b"test")
    detector = UltralyticsSegmenter(model_path)
    detector._model = _FakeModel(
        masks=SimpleNamespace(
            xy=[np.asarray([[1.0, 2.0], [11.0, 2.0], [11.0, 12.0]], dtype=np.float32)]
        )
    )

    detections = detector.detect(cast(UInt8Image, np.zeros((20, 20, 3), dtype=np.uint8)))

    assert detector.task == "segment"
    assert detector.mask_capable is True
    assert detections[0].mask is not None


def test_invalid_ultralytics_task_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported Ultralytics task"):
        UltralyticsDetector(tmp_path / "model.pt", task=cast(Any, "classify"))


def test_degenerate_and_non_finite_native_boxes_are_skipped(tmp_path: Path) -> None:
    model_path = tmp_path / "detector.pt"
    model_path.write_bytes(b"test")
    detector = UltralyticsDetector(model_path, task="detect")
    detector._model = _FakeModel(
        masks=None,
        xyxy=np.asarray(
            [
                [0.0, 0.0, 0.0, 2.0],
                [1.0, 1.0, np.nan, 3.0],
                [2.0, 2.0, 9.0, 10.0],
            ],
            dtype=np.float32,
        ),
        confidence=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
    )

    detections = detector.detect(cast(UInt8Image, np.zeros((20, 20, 3), dtype=np.uint8)))

    assert len(detections) == 1
    assert detections[0].detection_id == "deep-2"
    assert detections[0].bbox.x1 == 2.0


def test_phase_e_cpu_profile_reports_detection_only_runtime() -> None:
    engine = InferenceEngine("configs/inference/phase_e_candidate.yaml", runtime_profile="cpu")
    profile = cast(dict[str, Any], engine.model_status["detector_profile"])

    assert profile["task"] == "detect"
    assert profile["selected_profile"] == "cpu"
    assert profile["artifact_sha256"].endswith("9703")
    assert profile["assignment_only"] is True
    assert profile["mask_capable"] is False
    assert profile["output_masks"] is False
    assert cast(Any, engine.classifier).providers == ["CPUExecutionProvider"]
