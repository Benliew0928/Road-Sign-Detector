import cv2
import numpy as np

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.catalogue.models import ActionCode
from roadsign_assist.inference.engine import InferenceEngine
from roadsign_assist.inference.models import (
    BoundingBoxModel,
    ClassificationModel,
    DetectionModel,
    OCRModel,
)
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


class FixedClassifier:
    name = "fixed-classifier"
    available = True

    def warmup(self) -> bool:
        return True

    def classify(self, crop: UInt8Image) -> ClassificationModel:
        del crop
        return ClassificationModel(
            semantic_sign_id="stop",
            confidence=0.70,
            accepted=True,
            model_name=self.name,
            top_k=[("stop", 0.70)],
            unknown_score=0.30,
        )


class AlwaysDetector(OneFrameDetector):
    def detect(self, image: UInt8Image) -> list[DetectionModel]:
        self.calls = 0
        return super().detect(image)


class AcceptThenRejectClassifier(FixedClassifier):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, crop: UInt8Image) -> ClassificationModel:
        del crop
        self.calls += 1
        if self.calls == 1:
            return ClassificationModel(
                semantic_sign_id="stop",
                confidence=0.95,
                accepted=True,
                model_name=self.name,
                top_k=[("stop", 0.95)],
                unknown_score=0.05,
            )
        return ClassificationModel(
            semantic_sign_id="unknown_sign",
            confidence=0.99,
            accepted=False,
            model_name=self.name,
            top_k=[("stop", 0.99)],
            unknown_score=0.72,
            rejection_reasons=["embedding_distance"],
        )


class NumericRestrictionClassifier(FixedClassifier):
    def classify(self, crop: UInt8Image) -> ClassificationModel:
        del crop
        return ClassificationModel(
            semantic_sign_id="weight_restriction",
            confidence=0.95,
            accepted=True,
            model_name=self.name,
            top_k=[("weight_restriction", 0.95), ("maximum_speed", 0.04)],
            unknown_score=0.01,
        )


class SpeedOCR:
    available = True
    loaded = True
    load_error = None

    def warmup(self) -> bool:
        return True

    def recognize(self, crop: UInt8Image) -> OCRModel:
        del crop
        return OCRModel(
            text="60",
            confidence=0.93,
            script="latin",
            language="zh_or_latin",
            numeric_value=60.0,
        )


def test_new_session_has_independent_tracking_state() -> None:
    shared = InferenceEngine()
    first = shared.new_session()
    second = shared.new_session()
    assert first.detector is second.detector
    assert first.classifier is second.classifier
    assert first.tracker is not second.tracker
    assert first.frame_id == second.frame_id == 0
    assert first.model_status["tracker"]
    assert first.model_status["runtime_badge"] == "LEGACY"
    assert first.model_status["config_name"] == "legacy_baseline_fallback_v1"
    assert first.model_status["preprocessing_version"] == "roadsign_raw_bgr_v1"
    assert len(str(first.model_status["config_sha256"])) == 64


def test_single_frame_keeps_raw_confidence_and_blocks_unsafe_announcement() -> None:
    engine = InferenceEngine()
    engine.detector = OneFrameDetector()
    engine.classifier = FixedClassifier()
    engine.tracker = IoUTracker(min_stable_frames=1)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    event = engine.process_frame(image, assume_stable=True).events[0]

    assert event.semantic_sign_id == "stop"
    assert event.confidence == 0.70
    assert event.advisory.safe_to_announce is False
    assert event.should_announce is False
    assert "classifier_fused:stop:0.700" in event.evidence


def test_rejected_frame_cannot_inherit_a_previous_strong_action() -> None:
    engine = InferenceEngine()
    engine.detector = AlwaysDetector()
    engine.classifier = AcceptThenRejectClassifier()
    engine.tracker = IoUTracker(min_stable_frames=1)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    accepted = engine.process_frame(image, assume_stable=True).events[0]
    rejected = engine.process_frame(image, assume_stable=True).events[0]

    assert accepted.action.code is ActionCode.STOP_REQUEST
    assert rejected.semantic_sign_id == "unknown_sign"
    assert rejected.action.code is ActionCode.UNKNOWN_CAUTION
    assert rejected.advisory.safe_to_announce is False
    assert rejected.should_announce is False
    assert "classifier_rejection:embedding_distance" in rejected.evidence


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


def test_close_up_mode_bypasses_detector_and_uses_the_full_image() -> None:
    engine = InferenceEngine()
    detector = OneFrameDetector()
    engine.detector = detector
    engine.classifier = FixedClassifier()
    engine.ocr = SpeedOCR()
    image = np.zeros((80, 120, 3), dtype=np.uint8)

    event = engine.process_close_up(image).events[0]

    assert detector.calls == 0
    assert event.bbox == BoundingBoxModel(x1=0, y1=0, x2=120, y2=80)
    assert event.semantic_sign_id == "stop"
    assert "input_mode:close_up_full_image" in event.evidence
    assert "detector:bypassed:1.000" in event.evidence


def test_close_up_speed_override_requires_ocr_and_red_circle_evidence() -> None:
    engine = InferenceEngine()
    engine.classifier = NumericRestrictionClassifier()
    engine.ocr = SpeedOCR()
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.circle(image, (60, 60), 48, (0, 0, 220), 10)

    event = engine.process_close_up(image).events[0]

    assert event.semantic_sign_id == "maximum_speed"
    assert event.action.target_speed_kmh == 60.0
    assert any(item.startswith("close_up_numeric_speed_override:60.0") for item in event.evidence)


def test_close_up_numeric_text_does_not_override_without_a_red_circle() -> None:
    engine = InferenceEngine()
    engine.classifier = NumericRestrictionClassifier()
    engine.ocr = SpeedOCR()
    image = np.full((120, 120, 3), 255, dtype=np.uint8)

    event = engine.process_close_up(image).events[0]

    assert event.semantic_sign_id == "weight_restriction"
    assert not any(item.startswith("close_up_numeric_speed_override:") for item in event.evidence)
