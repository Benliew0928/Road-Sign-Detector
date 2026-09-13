from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.detection.fusion_backend import FusionSignDetector
from roadsign_assist.inference.models import BoundingBoxModel, DetectionModel


@dataclass
class _FakeDetector:
    detector_name: str
    detections: list[DetectionModel]
    warmup_result: bool = True

    @property
    def name(self) -> str:
        return self.detector_name

    @property
    def available(self) -> bool:
        return True

    def warmup(self) -> bool:
        return self.warmup_result

    def detect(self, image: UInt8Image) -> list[DetectionModel]:
        return self.detections


def _detection(identifier: str, box: tuple[float, float, float, float], score: float) -> DetectionModel:
    return DetectionModel(
        detection_id=identifier,
        bbox=BoundingBoxModel(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        confidence=score,
        detector="fake",
    )


def test_fusion_runs_both_members_and_applies_deterministic_class_agnostic_nms() -> None:
    candidate01 = _FakeDetector(
        "candidate01",
        [_detection("c01-overlap", (0, 0, 10, 10), 0.80)],
    )
    v12 = _FakeDetector(
        "v12",
        [
            _detection("v12-overlap", (1, 1, 11, 11), 0.70),
            _detection("v12-unique", (20, 20, 30, 30), 0.60),
        ],
    )
    detector = FusionSignDetector([candidate01, v12], nms_iou_threshold=0.50)

    detections = detector.detect(cast(UInt8Image, np.zeros((32, 32, 3), dtype=np.uint8)))

    assert [d.confidence for d in detections] == [0.80, 0.60]
    assert [d.detection_id for d in detections] == ["fusion-0", "fusion-1"]
    assert detections[0].detector == "candidate01|fusion"
    assert detections[1].detector == "v12|fusion"


def test_fusion_requires_all_members_to_warm_up() -> None:
    first = _FakeDetector("first", [], warmup_result=True)
    second = _FakeDetector("second", [], warmup_result=False)

    assert FusionSignDetector([first, second]).warmup() is False
