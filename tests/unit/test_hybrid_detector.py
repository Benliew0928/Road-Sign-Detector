import numpy as np

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.detection.hybrid_backend import HybridSignDetector
from roadsign_assist.inference.models import BoundingBoxModel, DetectionModel


class FakeDetector:
    def __init__(self, name: str, detections: list[DetectionModel]) -> None:
        self.name = name
        self.detections = detections
        self.available = True
        self.calls = 0

    def warmup(self) -> bool:
        return True

    def detect(self, image: UInt8Image) -> list[DetectionModel]:
        del image
        self.calls += 1
        return self.detections


def _detection() -> DetectionModel:
    return DetectionModel(
        detection_id="test",
        bbox=BoundingBoxModel(x1=1, y1=1, x2=10, y2=10),
        confidence=0.8,
        detector="test",
    )


def test_hybrid_uses_fallback_only_when_primary_is_empty() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    primary = FakeDetector("primary", [])
    fallback = FakeDetector("fallback", [_detection()])
    hybrid = HybridSignDetector(primary, fallback)
    assert hybrid.detect(image) == fallback.detections
    assert primary.calls == fallback.calls == 1

    primary.detections = [_detection()]
    assert hybrid.detect(image) == primary.detections
    assert primary.calls == 2
    assert fallback.calls == 1


def test_hybrid_caps_fallback_candidates() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    primary = FakeDetector("primary", [])
    fallback = FakeDetector("fallback", [_detection() for _ in range(5)])
    hybrid = HybridSignDetector(primary, fallback, max_fallback_detections=2)
    assert len(hybrid.detect(image)) == 2
