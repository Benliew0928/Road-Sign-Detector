from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.classification.base import SignClassifier, UnknownClassifier
from roadsign_assist.classification.onnx_backend import ONNXSignClassifier
from roadsign_assist.config import load_yaml
from roadsign_assist.detection.base import SignDetector
from roadsign_assist.detection.baseline_backend import BaselineSignDetector
from roadsign_assist.detection.hybrid_backend import HybridSignDetector
from roadsign_assist.detection.ultralytics_backend import UltralyticsSegmenter
from roadsign_assist.inference.models import (
    BoundingBoxModel,
    ClassificationModel,
    FrameResultModel,
    InferenceMode,
    OCRModel,
    SignEventModel,
)
from roadsign_assist.ocr.engine import MultilingualOCREngine
from roadsign_assist.paths import project_path
from roadsign_assist.semantics.rules import SemanticRuleEngine
from roadsign_assist.tracking.factory import build_tracker
from roadsign_assist.tracking.fusion import update_semantic_scores
from roadsign_assist.tracking.iou_tracker import TrackState

LOGGER = logging.getLogger(__name__)


def decode_image(data: bytes) -> UInt8Image:
    import numpy as np

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cast(UInt8Image | None, cv2.imdecode(buffer, cv2.IMREAD_COLOR))
    if image is None:
        raise ValueError("The supplied bytes are not a supported image")
    return image


def crop_detection(image: UInt8Image, bbox: Any, padding: float = 0.06) -> UInt8Image:
    height, width = image.shape[:2]
    pad_x = bbox.width * padding
    pad_y = bbox.height * padding
    x1 = max(0, int(bbox.x1 - pad_x))
    y1 = max(0, int(bbox.y1 - pad_y))
    x2 = min(width, int(bbox.x2 + pad_x))
    y2 = min(height, int(bbox.y2 + pad_y))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Detection produced an empty crop")
    return crop


class InferenceEngine:
    def __init__(
        self,
        config_path: str | Path = "configs/inference/default.yaml",
        *,
        shared: InferenceEngine | None = None,
    ) -> None:
        if shared is None:
            self.config = load_yaml(config_path)
            self.detector, self.mode, self.warnings = self._build_detector()
            if bool(self.config.get("experimental", False)):
                self.warnings.append(
                    "Experimental unreviewed models are active; results are not production claims."
                )
            self.classifier = self._build_classifier()
            self.ocr = MultilingualOCREngine(enabled=bool(self.config["ocr"]["enabled"]))
            semantic_settings = self.config["semantics"]
            self.rules = SemanticRuleEngine(
                critical_confidence=float(semantic_settings["critical_confidence_threshold"]),
                normal_confidence=float(semantic_settings["normal_confidence_threshold"]),
            )
        else:
            self.config = shared.config
            self.detector = shared.detector
            self.mode = shared.mode
            self.warnings = list(shared.warnings)
            self.classifier = shared.classifier
            self.ocr = shared.ocr
            self.rules = shared.rules
        tracking = self.config["tracking"]
        self.tracker = build_tracker(tracking)
        self._display_hold_frames = max(0, int(tracking.get("display_hold_frames", 0)))
        self.frame_id = 0
        self._ocr_cache: dict[int, OCRModel] = {}
        self._event_cache: dict[int, SignEventModel] = {}

    def new_session(self) -> InferenceEngine:
        """Create independent tracking state while sharing loaded model backends."""
        return InferenceEngine(shared=self)

    def _build_detector(self) -> tuple[SignDetector, InferenceMode, list[str]]:
        settings = self.config["detector"]
        deep = UltralyticsSegmenter(
            settings["model_path"],
            confidence_threshold=float(settings["confidence_threshold"]),
            nms_iou_threshold=float(settings["nms_iou_threshold"]),
            image_size=int(settings.get("image_size", 640)),
        )
        backend = str(settings.get("backend", "auto"))
        if backend in {"auto", "deep"} and deep.available:
            detector: SignDetector = deep
            if bool(settings.get("fallback_to_baseline", False)):
                detector = HybridSignDetector(
                    deep,
                    BaselineSignDetector(),
                    max_fallback_detections=int(settings.get("fallback_max_detections", 3)),
                )
            return detector, InferenceMode.DEEP, []
        if backend == "deep":
            raise FileNotFoundError(project_path(settings["model_path"]))
        return (
            BaselineSignDetector(),
            InferenceMode.BASELINE,
            ["Deep segmentation weights are unavailable; using color/shape baseline."],
        )

    def _build_classifier(self) -> SignClassifier:
        settings = self.config["classifier"]
        classifier = ONNXSignClassifier(
            settings["model_path"],
            settings.get(
                "labels_path",
                "models/exported/runtime/sign_classifier.labels.json",
            ),
            calibration_path=settings.get(
                "calibration_path",
                "models/exported/runtime/sign_classifier.calibration.json",
            ),
            confidence_threshold=float(settings["confidence_threshold"]),
            image_size=int(settings.get("image_size", 224)),
        )
        if classifier.available:
            return classifier
        self.warnings.append(
            "Semantic classifier weights are unavailable; detections remain unknown."
        )
        return UnknownClassifier()

    @property
    def model_status(self) -> dict[str, object]:
        classifier_providers = tuple(getattr(self.classifier, "active_providers", ()))
        detector_settings = self.config["detector"]
        classifier_settings = self.config["classifier"]
        return {
            "mode": self.mode,
            "detector": self.detector.name,
            "detector_available": self.detector.available,
            "detector_loaded": bool(getattr(self.detector, "loaded", True)),
            "detector_device": getattr(self.detector, "active_device", None),
            "detector_profile": {
                "backend": detector_settings.get("backend"),
                "model_path": detector_settings.get("model_path"),
                "image_size": detector_settings.get("image_size"),
                "confidence_threshold": detector_settings.get("confidence_threshold"),
                "nms_iou_threshold": detector_settings.get("nms_iou_threshold"),
                "fallback_to_baseline": detector_settings.get("fallback_to_baseline"),
                "fallback_max_detections": detector_settings.get("fallback_max_detections"),
            },
            "classifier": self.classifier.name,
            "classifier_release_status": classifier_settings.get(
                "release_status", "unspecified"
            ),
            "classifier_available": self.classifier.available,
            "classifier_loaded": bool(getattr(self.classifier, "loaded", True)),
            "classifier_providers": classifier_providers,
            "classifier_profile": {
                "backend": classifier_settings.get("backend"),
                "release_status": classifier_settings.get(
                    "release_status", "unspecified"
                ),
                "model_path": classifier_settings.get("model_path"),
                "labels_path": classifier_settings.get("labels_path"),
                "calibration_path": classifier_settings.get("calibration_path"),
                "image_size": classifier_settings.get("image_size"),
                "confidence_threshold": classifier_settings.get("confidence_threshold"),
            },
            "tracker": self.tracker.name,
            "ocr_available": self.ocr.available,
            "ocr_loaded": self.ocr.loaded,
            "ocr_load_error": self.ocr.load_error,
            "warnings": list(self.warnings),
        }

    def _runtime_device(self) -> str:
        detector_device = str(getattr(self.detector, "active_device", "") or "")
        classifier_providers = tuple(getattr(self.classifier, "active_providers", ()))
        if "cuda" in detector_device.casefold() or any(
            "cuda" in str(provider).casefold() for provider in classifier_providers
        ):
            return "cuda"
        return "cpu"

    def warmup(self) -> dict[str, bool]:
        return {
            "detector": self.detector.warmup(),
            "classifier": self.classifier.warmup(),
            "ocr": self.ocr.warmup(),
        }

    def _classify(self, crop: UInt8Image) -> ClassificationModel:
        try:
            return self.classifier.classify(crop)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            LOGGER.warning("Classifier failed: %s", exc)
            return UnknownClassifier().classify(crop)

    def _recognize_text(self, crop: UInt8Image, track: TrackState, stable: bool) -> OCRModel:
        cached = self._ocr_cache.get(track.track_id)
        if cached is not None:
            return cached
        if not stable:
            return OCRModel()
        try:
            result = self.ocr.recognize(crop)
        except (RuntimeError, ValueError) as exc:
            LOGGER.warning("OCR failed: %s", exc)
            result = OCRModel()
        self._ocr_cache[track.track_id] = result
        return result

    def process_frame(
        self,
        image: UInt8Image,
        *,
        assume_stable: bool = False,
    ) -> FrameResultModel:
        started = time.perf_counter()
        frame_id = self.frame_id
        self.frame_id += 1
        detections = self.detector.detect(image)
        assignments = self.tracker.update(detections, image=image)
        events: list[SignEventModel] = []

        active_ids = {track.track_id for _, track in assignments}
        tracked_ids = {track.track_id for track in self.tracker.tracks}
        self._ocr_cache = {
            track_id: value for track_id, value in self._ocr_cache.items() if track_id in tracked_ids
        }
        self._event_cache = {
            track_id: value for track_id, value in self._event_cache.items() if track_id in tracked_ids
        }

        for candidate_index, (detection, track) in enumerate(assignments):
            crop = crop_detection(image, detection.bbox)
            prediction = self._classify(crop)
            fused_label, fused_confidence = update_semantic_scores(track, prediction)
            stable = assume_stable or self.tracker.is_stable(track)
            should_run_ocr = self.rules.requires_ocr(fused_label) or (
                fused_label == "unknown_sign" and candidate_index == 0
            )
            ocr = self._recognize_text(crop, track, stable) if should_run_ocr else OCRModel()
            semantic_id, confidence, evidence = self.rules.resolve_label(
                fused_label,
                fused_confidence,
                ocr,
            )
            meaning, severity, action = self.rules.action_for(semantic_id, confidence, ocr)
            advisory = self.rules.advisory_for(semantic_id, meaning, confidence, action)
            should_announce = self.rules.should_announce(track, stable=stable)
            event = SignEventModel(
                frame_id=frame_id,
                track_id=track.track_id,
                semantic_sign_id=semantic_id,
                meaning=meaning,
                ocr=ocr,
                confidence=confidence,
                bbox=detection.bbox,
                mask=detection.mask,
                action=action,
                advisory=advisory,
                severity=severity,
                latency_ms=(time.perf_counter() - started) * 1000,
                device=self._runtime_device(),
                stable=stable,
                should_announce=should_announce,
                evidence=[
                    f"detector:{detection.detector}:{detection.confidence:.3f}",
                    (
                        f"classifier_raw:{prediction.top_k[0][0]}:{prediction.confidence:.3f}"
                        if prediction.top_k
                        else f"classifier_raw:unknown_sign:{prediction.confidence:.3f}"
                    ),
                    *(
                        [
                            "embedding:"
                            f"{prediction.nearest_prototype}:"
                            f"{prediction.embedding_distance:.3f}"
                        ]
                        if prediction.embedding_distance is not None
                        else []
                    ),
                    *[
                        f"classifier_rejection:{reason}"
                        for reason in prediction.rejection_reasons
                    ],
                    *evidence,
                ],
            )
            events.append(event)
            self._event_cache[track.track_id] = event

        for track in self.tracker.tracks:
            if (
                track.track_id in active_ids
                or track.missed < 1
                or track.missed > self._display_hold_frames
            ):
                continue
            previous = self._event_cache.get(track.track_id)
            if previous is None:
                continue
            held_bbox = _project_track_bbox(track, image.shape[1], image.shape[0])
            events.append(
                previous.model_copy(
                    update={
                        "frame_id": frame_id,
                        "bbox": held_bbox,
                        "mask": None,
                        "confidence": previous.confidence * (0.9**track.missed),
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "stable": False,
                        "should_announce": False,
                        "evidence": [*previous.evidence, f"tracker_hold:{track.missed}"],
                    }
                )
            )

        return FrameResultModel(
            frame_id=frame_id,
            width=image.shape[1],
            height=image.shape[0],
            mode=self.mode,
            latency_ms=(time.perf_counter() - started) * 1000,
            events=events,
            warnings=list(self.warnings),
        )


def _project_track_bbox(track: TrackState, width: int, height: int) -> BoundingBoxModel:
    steps = max(1, track.missed)
    dx = track.velocity_x * steps
    dy = track.velocity_y * steps
    x1 = min(max(0.0, track.bbox.x1 + dx), max(0.0, width - 1.0))
    y1 = min(max(0.0, track.bbox.y1 + dy), max(0.0, height - 1.0))
    x2 = min(max(x1 + 1.0, track.bbox.x2 + dx), float(width))
    y2 = min(max(y1 + 1.0, track.bbox.y2 + dy), float(height))
    return BoundingBoxModel(x1=x1, y1=y1, x2=x2, y2=y2)


def annotate_frame(image: UInt8Image, result: FrameResultModel) -> UInt8Image:
    annotated = image.copy()
    for event in result.events:
        bbox = event.bbox
        color = (40, 196, 120) if event.semantic_sign_id != "unknown_sign" else (0, 180, 240)
        cv2.rectangle(
            annotated,
            (round(bbox.x1), round(bbox.y1)),
            (round(bbox.x2), round(bbox.y2)),
            color,
            2,
        )
        label = f"#{event.track_id} {event.advisory.headline.en} {event.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (round(bbox.x1), max(18, round(bbox.y1) - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def encode_jpeg(image: UInt8Image, quality: int = 88) -> bytes:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("Unable to encode annotated frame")
    return encoded.tobytes()


def process_images(
    engine: InferenceEngine,
    images: Sequence[UInt8Image],
    *,
    assume_stable: bool = False,
) -> list[FrameResultModel]:
    return [engine.process_frame(image, assume_stable=assume_stable) for image in images]
