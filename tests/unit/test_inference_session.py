import numpy as np

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.inference.engine import InferenceEngine
from roadsign_assist.inference.models import BoundingBoxModel, DetectionModel
from roadsign_assist.tracking.iou_tracker import IoUTracker


class OneFrameDetector:
    name = "one-frame-detector"
    available = True

    def __init__(self) -> None:
        self.calls = 0

    def warmup(self) -> bool:
        return True

    def detect(self, image: UInt8Image) -> list[DetectionModel]:
        del image
        self.calls += 1
        if self.calls > 1:
            return []
        return [
            DetectionModel(
                detection_id="held-sign",
                bbox=BoundingBoxModel(x1=20, y1=20, x2=60, y2=60),
                confidence=0.9,
                detector=self.name,
            )
        ]


def test_new_session_has_independent_tracking_state() -> None:
    shared = InferenceEngine()
    first = shared.new_session()
    second = shared.new_session()
    assert first.detector is second.detector
    assert first.classifier is second.classifier
    assert first.tracker is not second.tracker
    assert first.frame_id == second.frame_id == 0
    assert first.model_status["tracker"]


def test_short_detector_gaps_keep_a_non_announcing_tracking_box() -> None:
    engine = InferenceEngine()
    engine.detector = OneFrameDetector()
    engine.tracker = IoUTracker(min_stable_frames=1, max_missed_frames=5)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    detected = engine.process_frame(image)
    first_held = engine.process_frame(image)
    second_held = engine.process_frame(image)
    expired = engine.process_frame(image)

    assert len(detected.events) == 1
    assert len(first_held.events) == 1
    assert len(second_held.events) == 1
    assert first_held.events[0].should_announce is False
    assert first_held.events[0].mask is None
    assert first_held.events[0].evidence[-1] == "tracker_hold:1"
    assert second_held.events[0].evidence[-1] == "tracker_hold:2"
    assert expired.events == []
