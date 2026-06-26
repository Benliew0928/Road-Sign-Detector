from __future__ import annotations

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.detection.base import SignDetector
from roadsign_assist.inference.models import DetectionModel


class HybridSignDetector:
    """Use the deep segmenter first and the classical detector on empty frames."""

    def __init__(
        self,
        primary: SignDetector,
        fallback: SignDetector,
        *,
        max_fallback_detections: int = 3,
    ) -> None:
        if max_fallback_detections < 1:
            raise ValueError("max_fallback_detections must be positive")
        self.primary = primary
        self.fallback = fallback
        self.max_fallback_detections = max_fallback_detections

    @property
    def name(self) -> str:
        return f"hybrid:{self.primary.name}+{self.fallback.name}"

    @property
    def available(self) -> bool:
        return self.primary.available and self.fallback.available

    @property
    def loaded(self) -> bool:
        return bool(getattr(self.primary, "loaded", True))

    @property
    def active_device(self) -> str | None:
        value = getattr(self.primary, "active_device", None)
        return str(value) if value is not None else None

    def warmup(self) -> bool:
        return self.primary.warmup() and self.fallback.warmup()

    def detect(self, image: UInt8Image) -> list[DetectionModel]:
        detections = self.primary.detect(image)
        if detections:
            return detections
        fallback = self.fallback.detect(image)
        return sorted(
            fallback,
            key=lambda detection: detection.confidence,
            reverse=True,
        )[: self.max_fallback_detections]
