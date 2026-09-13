from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.classification.base import SignClassifier, UnknownClassifier
from roadsign_assist.classification.onnx_backend import ONNXSignClassifier
from roadsign_assist.classification.routed_backend import RoutedSignClassifier
from roadsign_assist.classification.preprocessing import CLASSIFIER_PREPROCESSING_VERSION
from roadsign_assist.config import load_yaml
from roadsign_assist.detection.base import SignDetector
from roadsign_assist.detection.baseline_backend import BaselineSignDetector
from roadsign_assist.detection.fusion_backend import FusionSignDetector
from roadsign_assist.detection.hybrid_backend import HybridSignDetector
from roadsign_assist.detection.ultralytics_backend import UltralyticsDetector, UltralyticsTask
from roadsign_assist.inference.models import (
    BoundingBoxModel,
    ClassificationModel,
    FrameResultModel,
    InferenceMode,
    OCRModel,
    SignEventModel,
)
from roadsign_assist.inference.preprocessing import PREPROCESSING_VERSION, decode_image_bytes
from roadsign_assist.ocr.engine import MultilingualOCREngine
from roadsign_assist.paths import project_path
from roadsign_assist.semantics.rules import SemanticRuleEngine
from roadsign_assist.tracking.factory import build_tracker
from roadsign_assist.tracking.fusion import update_semantic_scores
from roadsign_assist.tracking.iou_tracker import TrackState

LOGGER = logging.getLogger(__name__)
RUNTIME_BADGES = frozenset({"LEGACY", "CANDIDATE", "SHADOW"})
_CLOSE_UP_NUMERIC_CONFUSION_FAMILY = frozenset(
    {
        "maximum_speed",
        "height_restriction",
        "weight_restriction",
        "width_restriction",
        "stop_for_checking",
    }
)


def decode_image(data: bytes) -> UInt8Image:
    return decode_image_bytes(data)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _looks_like_centered_red_circle(image: UInt8Image) -> bool:
    """Conservatively identify a close-up red circular regulatory sign."""
    height, width = image.shape[:2]
    if height < 24 or width < 24:
        return False
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.asarray((0, 70, 55)), np.asarray((12, 255, 255)))
    red |= cv2.inRange(hsv, np.asarray((165, 70, 55)), np.asarray((179, 255, 255)))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    image_area = float(height * width)
    if area < image_area * 0.10:
        return False
    x, y, box_width, box_height = cv2.boundingRect(contour)
    aspect = box_width / max(1.0, float(box_height))
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    perimeter = float(cv2.arcLength(contour, True))
    circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0
    return bool(
        0.72 <= aspect <= 1.38
        and abs(center_x - width / 2.0) <= width * 0.28
        and abs(center_y - height / 2.0) <= height * 0.28
        and circularity >= 0.48
    )


def _close_up_speed_override(
    image: UInt8Image,
    prediction: ClassificationModel,
    ocr: OCRModel,
) -> tuple[str, float] | None:
    """Recover a speed sign only when independent OCR and shape evidence agree."""
    raw_label = prediction.top_k[0][0] if prediction.top_k else prediction.semantic_sign_id
    value = ocr.numeric_value
    unit = ocr.unit.upper() if ocr.unit else None
    if (
        raw_label not in _CLOSE_UP_NUMERIC_CONFUSION_FAMILY
        or value is None
        or ocr.confidence < 0.65
        or unit not in {None, "KM/H"}
        or not 5.0 <= value <= 160.0
        or not _looks_like_centered_red_circle(image)
    ):
        return None
    confidence = min(1.0, max(prediction.confidence, ocr.confidence))
    return "maximum_speed", confidence


class InferenceEngine:
    def __init__(
        self,
        config_path: str | Path = "configs/inference/default.yaml",
        *,
        shared: InferenceEngine | None = None,
        runtime_profile: str | None = None,
    ) -> None:
        if shared is None:
            self.config_path = project_path(config_path)
            self.config = load_yaml(self.config_path)
            self.config_name = str(self.config.get("config_name", "")).strip()
            self.runtime_badge = str(self.config.get("runtime_badge", "")).strip().upper()
            self.preprocessing_version = str(
                self.config.get("preprocessing_version", "")
            ).strip()
            self._validate_runtime_identity()
            self.runtime_profile = runtime_profile
            self.selected_runtime_profile = "legacy"
            self.crop_padding = float(self.config.get("crop_padding", 0.06))
            self.detector, self.mode, self.warnings = self._build_detector()
            if bool(self.config.get("experimental", False)):
                self.warnings.append(
                    "Experimental unreviewed models are active; results are not production claims."
                )
            self.classifier = self._build_classifier()
            self.bundle_identity = self._build_bundle_identity()
            self.ocr = MultilingualOCREngine(
                enabled=bool(self.config["ocr"]["enabled"]),
                numeric_second_pass=bool(
                    self.config["ocr"].get("numeric_second_pass", False)
                ),
            )
            semantic_settings = self.config["semantics"]
            self.rules = SemanticRuleEngine(
                critical_confidence=float(semantic_settings["critical_confidence_threshold"]),
                normal_confidence=float(semantic_settings["normal_confidence_threshold"]),
                directional_strong_actions_enabled=bool(
                    semantic_settings.get("directional_strong_actions_enabled", True)
                ),
            )
        else:
            self.config = shared.config
            self.config_path = shared.config_path
            self.config_name = shared.config_name
            self.runtime_badge = shared.runtime_badge
            self.preprocessing_version = shared.preprocessing_version
            self.runtime_profile = shared.runtime_profile
            self.selected_runtime_profile = shared.selected_runtime_profile
            self.crop_padding = shared.crop_padding
            self.detector = shared.detector
            self.mode = shared.mode
            self.warnings = list(shared.warnings)
            self.classifier = shared.classifier
            self.bundle_identity = shared.bundle_identity
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

    def _validate_runtime_identity(self) -> None:
        if not self.config_name:
            raise ValueError(f"Inference config has no config_name: {self.config_path}")
        if self.runtime_badge not in RUNTIME_BADGES:
            raise ValueError(
                "Inference config runtime_badge must be LEGACY, CANDIDATE, or SHADOW: "
                f"{self.config_path}"
            )
        if self.preprocessing_version != PREPROCESSING_VERSION:
            raise ValueError(
                "Inference config preprocessing_version does not match the runtime: "
                f"configured={self.preprocessing_version!r}, runtime={PREPROCESSING_VERSION!r}"
            )

    @staticmethod
    def _artifact_identity(path: Path | None, expected_sha256: object = None) -> dict[str, object]:
        if path is None:
            return {"path": None, "available": False, "sha256": None}
        return {
            "path": str(path),
            "available": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256(path) if path.is_file() else None,
            "configured_sha256": str(expected_sha256) if expected_sha256 else None,
        }

    def _build_bundle_identity(self) -> dict[str, object]:
        detector_settings = self.config["detector"]
        classifier_settings = self.config["classifier"]
        detector = self.detector.primary if isinstance(self.detector, HybridSignDetector) else self.detector
        detector_path = getattr(detector, "model_path", None)
        classifier_model = getattr(self.classifier, "model_path", None)
        classifier_labels = getattr(self.classifier, "labels_path", None)
        classifier_calibration = getattr(self.classifier, "calibration_path", None)
        tracker_path = project_path(self.config["tracking"]["tracker_config_path"])
        catalogue_path = project_path(self.config["catalogue_path"])
        lock_path = project_path("uv.lock")
        configured_profiles = detector_settings.get("profiles", {})
        detector_profiles: dict[str, object] = {}
        if isinstance(configured_profiles, dict):
            for profile_name, raw_profile in cast(dict[str, Any], configured_profiles).items():
                if not isinstance(raw_profile, dict):
                    continue
                typed_profile = cast(dict[str, Any], raw_profile)
                if not typed_profile.get("model_path"):
                    continue
                profile_path = project_path(str(typed_profile["model_path"]))
                detector_profiles[profile_name] = self._artifact_identity(
                    profile_path,
                    typed_profile.get("sha256"),
                )
        configured_members = detector_settings.get("members", [])
        if isinstance(configured_members, list):
            for member_index, raw_member in enumerate(configured_members):
                if not isinstance(raw_member, dict) or not raw_member.get("model_path"):
                    continue
                member_name = str(raw_member.get("name", f"member-{member_index}"))
                detector_profiles[member_name] = self._artifact_identity(
                    project_path(str(raw_member["model_path"])),
                    raw_member.get("sha256"),
                )
        return {
            "config_name": self.config_name,
            "config_path": str(self.config_path),
            "config_sha256": _sha256(self.config_path),
            "runtime_badge": self.runtime_badge,
            "preprocessing_version": self.preprocessing_version,
            "classifier_preprocessing_version": CLASSIFIER_PREPROCESSING_VERSION,
            "detector": self._artifact_identity(
                detector_path if isinstance(detector_path, Path) else None,
                getattr(detector, "artifact_sha256", detector_settings.get("sha256")),
            ),
            "detector_profiles": detector_profiles,
            "classifier": self._artifact_identity(
                classifier_model if isinstance(classifier_model, Path) else None,
                classifier_settings.get("sha256"),
            ),
            "classifier_labels": self._artifact_identity(
                classifier_labels if isinstance(classifier_labels, Path) else None,
                classifier_settings.get("labels_sha256"),
            ),
            "classifier_calibration": self._artifact_identity(
                classifier_calibration if isinstance(classifier_calibration, Path) else None,
                classifier_settings.get("calibration_sha256"),
            ),
            "catalogue": self._artifact_identity(catalogue_path),
            "tracker_config": self._artifact_identity(tracker_path),
            "dependency_lock": self._artifact_identity(lock_path),
        }

    def _build_detector(self) -> tuple[SignDetector, InferenceMode, list[str]]:
        settings = self.config["detector"]
        backend = str(settings.get("backend", "auto"))
        if backend == "fusion":
            return self._build_fusion_detector(settings)
        profile_name, profile = self._select_detector_profile(settings)
        self.selected_runtime_profile = profile_name
        model_path = profile.get("model_path", settings.get("model_path"))
        if not model_path:
            raise ValueError("Detector configuration does not define a model path")
        task = str(settings.get("task", "segment"))
        if task not in {"detect", "segment"}:
            raise ValueError(f"Unsupported detector task: {task}")
        deep = UltralyticsDetector(
            model_path,
            confidence_threshold=float(settings["confidence_threshold"]),
            nms_iou_threshold=float(settings["nms_iou_threshold"]),
            image_size=int(settings.get("image_size", 640)),
            device=str(profile.get("device", settings.get("device", "auto"))),
            task=cast(UltralyticsTask, task),
            profile_name=profile_name,
            artifact_sha256=(str(profile["sha256"]) if profile.get("sha256") is not None else None),
            assignment_only=bool(
                profile.get("assignment_only", settings.get("assignment_only", False))
            ),
        )
        expected_sha256 = profile.get("sha256")
        if (
            deep.available
            and expected_sha256
            and bool(settings.get("verify_artifact_hash", False))
            and _sha256(deep.model_path) != str(expected_sha256).casefold()
        ):
            raise ValueError(
                f"Detector artifact hash mismatch for runtime profile {profile_name}"
            )
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
            raise FileNotFoundError(project_path(model_path))
        return (
            BaselineSignDetector(),
            InferenceMode.BASELINE,
            [f"Deep {task} weights are unavailable; using color/shape baseline."],
        )

    def _build_fusion_detector(
        self,
        settings: dict[str, Any],
    ) -> tuple[SignDetector, InferenceMode, list[str]]:
        raw_members = settings.get("members")
        if not isinstance(raw_members, list) or len(raw_members) < 2:
            raise ValueError("Fusion detector configuration requires at least two members")
        if bool(settings.get("requires_cuda", True)) and not self._cuda_available():
            raise RuntimeError("The native detector fusion profile requires CUDA")

        members: list[SignDetector] = []
        for member_index, raw_member in enumerate(raw_members):
            if not isinstance(raw_member, dict):
                raise ValueError(f"Fusion detector member {member_index} is invalid")
            member = cast(dict[str, Any], raw_member)
            model_path_value = member.get("model_path")
            if not model_path_value:
                raise ValueError(f"Fusion detector member {member_index} has no model path")
            member_name = str(member.get("name", f"member-{member_index}"))
            detector = UltralyticsDetector(
                str(model_path_value),
                confidence_threshold=float(member["confidence_threshold"]),
                nms_iou_threshold=float(
                    member.get("nms_iou_threshold", settings.get("nms_iou_threshold", 0.50))
                ),
                image_size=int(member.get("image_size", settings.get("image_size", 640))),
                device=str(member.get("device", "0")),
                task="detect",
                profile_name=member_name,
                artifact_sha256=(str(member["sha256"]) if member.get("sha256") else None),
                assignment_only=bool(member.get("assignment_only", True)),
            )
            if not detector.available:
                raise FileNotFoundError(detector.model_path)
            expected_sha256 = member.get("sha256")
            if (
                expected_sha256
                and bool(settings.get("verify_artifact_hash", False))
                and _sha256(detector.model_path) != str(expected_sha256).casefold()
            ):
                raise ValueError(f"Detector artifact hash mismatch for fusion member {member_name}")
            members.append(detector)

        self.selected_runtime_profile = "fusion-gpu"
        fusion = FusionSignDetector(
            members,
            nms_iou_threshold=float(settings.get("nms_iou_threshold", 0.50)),
            profile_name="fusion-gpu",
        )
        return (
            fusion,
            InferenceMode.DEEP,
            [
                "Best-effort detector fusion is active; it improves development recall but did "
                "not pass the formal Phase 6 release gates."
            ],
        )

    def _select_detector_profile(
        self,
        settings: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        profiles = settings.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            return "legacy", {}
        requested = self.runtime_profile or str(settings.get("preferred_profile", "auto"))
        if requested not in {"auto", "gpu", "cpu"}:
            raise ValueError(f"Unsupported detector runtime profile: {requested}")
        if requested == "auto":
            requested = "gpu" if self._cuda_available() and "gpu" in profiles else "cpu"
        typed_profiles = cast(dict[str, Any], profiles)
        profile = typed_profiles.get(requested)
        if not isinstance(profile, dict):
            raise ValueError(f"Detector runtime profile is not configured: {requested}")
        return requested, cast(dict[str, Any], profile)

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def _build_classifier(self) -> SignClassifier:
        settings = self.config["classifier"]
        if str(settings.get("backend", "deep")) == "routed":
            policy_path = project_path(settings["policy_path"])
            if not policy_path.is_file():
                raise FileNotFoundError(policy_path)
            if bool(settings.get("verify_artifact_hash", False)):
                expected_policy = str(settings.get("policy_sha256", "")).casefold()
                if not expected_policy or _sha256(policy_path) != expected_policy:
                    raise ValueError("Classifier routing policy hash mismatch")
            policy = load_yaml(policy_path) if policy_path.suffix.lower() in {".yaml", ".yml"} else json.loads(policy_path.read_text(encoding="utf-8"))

            def member(role: str) -> ONNXSignClassifier:
                entry = policy[role]
                model_path = project_path(entry["model"])
                labels_path = model_path.parent / "sign_classifier.labels.json"
                calibration_path = model_path.parent / "sign_classifier.calibration.json"
                if bool(settings.get("verify_artifact_hash", False)):
                    if _sha256(model_path) != str(entry["model_sha256"]).casefold():
                        raise ValueError(f"Routed classifier {role} model hash mismatch")
                return ONNXSignClassifier(
                    model_path,
                    labels_path,
                    calibration_path=calibration_path,
                    confidence_threshold=float(entry["confidence_threshold"]),
                    image_size=int(settings.get("image_size", 320)),
                    providers=(
                        ["CPUExecutionProvider"]
                        if self.selected_runtime_profile == "cpu"
                        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    ),
                )

            routed = RoutedSignClassifier(
                member("primary"), member("fallback"),
                repair_labels=set(policy["repair_labels"]),
                name=str(settings.get("name", "routed_classifier")),
            )
            if routed.available:
                return routed
            self.warnings.append("Routed semantic classifier weights are unavailable; detections remain unknown.")
            return UnknownClassifier()
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
            providers=(
                ["CPUExecutionProvider"]
                if self.selected_runtime_profile == "cpu"
                else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ),
        )
        if classifier.available:
            if bool(settings.get("verify_artifact_hash", False)):
                expected = {
                    classifier.model_path: settings.get("sha256"),
                    classifier.labels_path: settings.get("labels_sha256"),
                    classifier.calibration_path: settings.get("calibration_sha256"),
                }
                for path, expected_sha256 in expected.items():
                    if path is None or expected_sha256 is None:
                        continue
                    if not path.is_file() or _sha256(path) != str(expected_sha256).casefold():
                        raise ValueError(f"Classifier artifact hash mismatch: {path.name}")
            if settings.get("release_status") == "safety_preview_not_promoted":
                self.warnings.append(
                    "Safety-preview rejection policy is active; uncertain signs fail closed and "
                    "this pipeline is not Gate E promoted."
                )
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
        classifier_runtime = dict(getattr(self.classifier, "runtime_metadata", {}))
        configured_release_status = classifier_settings.get("release_status", "unspecified")
        classifier_release_status = (
            configured_release_status
            if configured_release_status in {"legacy_retained", "safety_preview_not_promoted"}
            else classifier_runtime.get("release_status", configured_release_status)
        )
        detector_members = [
            {
                "name": member.profile_name,
                "model_path": str(member.model_path),
                "artifact_sha256": member.artifact_sha256,
                "confidence_threshold": member.confidence_threshold,
                "nms_iou_threshold": member.nms_iou_threshold,
                "image_size": member.image_size,
                "loaded": member.loaded,
                "device": member.active_device,
            }
            for member in getattr(self.detector, "members", ())
        ]
        detector_model_path = getattr(self.detector, "model_path", None)
        return {
            "runtime_badge": self.runtime_badge,
            "config_name": self.config_name,
            "config_path": str(self.config_path),
            "config_sha256": self.bundle_identity["config_sha256"],
            "preprocessing_version": self.preprocessing_version,
            "bundle_identity": self.bundle_identity,
            "mode": self.mode,
            "detector": self.detector.name,
            "detector_available": self.detector.available,
            "detector_loaded": bool(getattr(self.detector, "loaded", True)),
            "detector_device": getattr(self.detector, "active_device", None),
            "detector_profile": {
                "backend": detector_settings.get("backend"),
                "task": getattr(self.detector, "task", detector_settings.get("task", "segment")),
                "selected_profile": getattr(self.detector, "profile_name", "legacy"),
                "model_path": (
                    str(detector_model_path)
                    if detector_model_path is not None
                    else detector_settings.get("model_path")
                ),
                "artifact_sha256": getattr(self.detector, "artifact_sha256", None),
                "members": detector_members,
                "image_size": detector_settings.get("image_size"),
                "confidence_threshold": detector_settings.get("confidence_threshold"),
                "nms_iou_threshold": detector_settings.get("nms_iou_threshold"),
                "fallback_to_baseline": detector_settings.get("fallback_to_baseline"),
                "fallback_max_detections": detector_settings.get("fallback_max_detections"),
                "assignment_only": bool(getattr(self.detector, "assignment_only", False)),
                "mask_capable": bool(getattr(self.detector, "mask_capable", False)),
                "output_masks": bool(getattr(self.detector, "mask_capable", False)),
                "crop_padding": self.crop_padding,
            },
            "classifier": self.classifier.name,
            "classifier_release_status": classifier_release_status,
            "classifier_available": self.classifier.available,
            "classifier_loaded": bool(getattr(self.classifier, "loaded", True)),
            "classifier_providers": classifier_providers,
            "classifier_profile": {
                "backend": classifier_settings.get("backend"),
                "release_status": classifier_release_status,
                "model_path": classifier_settings.get("model_path"),
                "labels_path": classifier_settings.get("labels_path"),
                "calibration_path": classifier_settings.get("calibration_path"),
                "image_size": getattr(
                    self.classifier, "image_size", classifier_settings.get("image_size")
                ),
                "confidence_threshold": getattr(
                    self.classifier,
                    "confidence_threshold",
                    classifier_settings.get("confidence_threshold"),
                ),
                "internal_academic_only": classifier_runtime.get(
                    "internal_academic_only",
                    classifier_settings.get("internal_academic_only", False),
                ),
                "artifact_sha256": classifier_settings.get("sha256"),
                "calibration_sha256": classifier_settings.get("calibration_sha256"),
                "embedding_gate_active": bool(
                    getattr(self.classifier, "embedding_gate_active", False)
                ),
                "assignment_only": bool(classifier_settings.get("assignment_only", False)),
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

    def _recognize_text(
        self,
        crop: UInt8Image,
        track: TrackState,
        stable: bool,
        semantic_hint: str | None = None,
    ) -> OCRModel:
        cached = self._ocr_cache.get(track.track_id)
        if cached is not None:
            return cached
        if not stable:
            return OCRModel()
        try:
            result = self.ocr.recognize(crop, semantic_hint=semantic_hint)
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
            track_id: value
            for track_id, value in self._ocr_cache.items()
            if track_id in tracked_ids
        }
        self._event_cache = {
            track_id: value
            for track_id, value in self._event_cache.items()
            if track_id in tracked_ids
        }

        for candidate_index, (detection, track) in enumerate(assignments):
            crop = crop_detection(image, detection.bbox, padding=self.crop_padding)
            prediction = self._classify(crop)
            fused_label, fused_confidence = update_semantic_scores(track, prediction)
            stable = assume_stable or self.tracker.is_stable(track)
            should_run_ocr = self.rules.requires_ocr(fused_label) or (
                fused_label == "unknown_sign" and candidate_index == 0
            )
            ocr = (
                self._recognize_text(crop, track, stable, semantic_hint=fused_label)
                if should_run_ocr
                else OCRModel()
            )
            semantic_id, confidence, evidence = self.rules.resolve_label(
                fused_label,
                fused_confidence,
                ocr,
            )
            meaning, severity, action = self.rules.action_for(semantic_id, confidence, ocr)
            advisory = self.rules.advisory_for(semantic_id, meaning, confidence, action)
            should_announce = advisory.safe_to_announce and self.rules.should_announce(
                track, stable=stable
            )
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
                    *[f"classifier_rejection:{reason}" for reason in prediction.rejection_reasons],
                    f"classifier_fused:{fused_label}:{fused_confidence:.3f}",
                    *(
                        ["safety:directional_action_blocked"]
                        if self.rules.directional_action_blocked(semantic_id)
                        else []
                    ),
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

    def process_close_up(self, image: UInt8Image) -> FrameResultModel:
        """Interpret one upright, close-up sign without invoking detection or tracking."""
        started = time.perf_counter()
        frame_id = self.frame_id
        self.frame_id += 1
        height, width = image.shape[:2]
        bbox = BoundingBoxModel(x1=0.0, y1=0.0, x2=float(width), y2=float(height))
        prediction = self._classify(image)
        track = TrackState(track_id=1, bbox=bbox)
        ocr = self._recognize_text(
            image,
            track,
            stable=True,
            semantic_hint=prediction.semantic_sign_id,
        )
        classifier_label = prediction.semantic_sign_id
        classifier_confidence = prediction.confidence
        close_up_evidence = ["input_mode:close_up_full_image", "detector:bypassed:1.000"]
        speed_override = _close_up_speed_override(image, prediction, ocr)
        if speed_override is not None:
            classifier_label, classifier_confidence = speed_override
            close_up_evidence.append(
                f"close_up_numeric_speed_override:{ocr.numeric_value}:{ocr.confidence:.3f}"
            )
        semantic_id, confidence, evidence = self.rules.resolve_label(
            classifier_label,
            classifier_confidence,
            ocr,
        )
        meaning, severity, action = self.rules.action_for(semantic_id, confidence, ocr)
        advisory = self.rules.advisory_for(semantic_id, meaning, confidence, action)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw_label = prediction.top_k[0][0] if prediction.top_k else "unknown_sign"
        event = SignEventModel(
            frame_id=frame_id,
            track_id=1,
            semantic_sign_id=semantic_id,
            meaning=meaning,
            ocr=ocr,
            confidence=confidence,
            bbox=bbox,
            mask=None,
            action=action,
            advisory=advisory,
            severity=severity,
            latency_ms=elapsed_ms,
            device=self._runtime_device(),
            stable=True,
            should_announce=advisory.safe_to_announce,
            evidence=[
                *close_up_evidence,
                f"classifier_raw:{raw_label}:{prediction.confidence:.3f}",
                *[f"classifier_rejection:{reason}" for reason in prediction.rejection_reasons],
                f"classifier_fused:{classifier_label}:{classifier_confidence:.3f}",
                *(
                    ["safety:directional_action_blocked"]
                    if self.rules.directional_action_blocked(semantic_id)
                    else []
                ),
                *evidence,
            ],
        )
        warning = (
            "Close-up mode bypasses sign detection and expects exactly one upright sign "
            "occupying most of the image."
        )
        return FrameResultModel(
            frame_id=frame_id,
            width=width,
            height=height,
            mode=self.mode,
            latency_ms=elapsed_ms,
            events=[event],
            warnings=[*self.warnings, warning],
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
        label = f"#{event.track_id} {event.meaning.en} {event.confidence:.2f}"
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
