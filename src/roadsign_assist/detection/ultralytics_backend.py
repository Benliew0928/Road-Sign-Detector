from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from pathlib import Path
from typing import Any, Literal

import numpy as np

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.inference.models import (
    BoundingBoxModel,
    DetectionModel,
    MaskModel,
)
from roadsign_assist.paths import project_path

UltralyticsTask = Literal["detect", "segment"]


class UltralyticsDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.50,
        device: str = "auto",
        image_size: int = 640,
        task: UltralyticsTask = "detect",
        profile_name: str = "legacy",
        artifact_sha256: str | None = None,
        assignment_only: bool = False,
    ) -> None:
        if task not in {"detect", "segment"}:
            raise ValueError(f"Unsupported Ultralytics task: {task}")
        self.model_path = project_path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = None if device == "auto" else device
        self.image_size = image_size
        self.task: UltralyticsTask = task
        self.profile_name = profile_name
        self.artifact_sha256 = artifact_sha256
        self.assignment_only = assignment_only
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return f"ultralytics-{self.task}:{self.model_path.name}"

    @property
    def mask_capable(self) -> bool:
        return self.task == "segment"

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def active_device(self) -> str | None:
        if self._model is None:
            return None
        predictor = getattr(self._model, "predictor", None)
        value = getattr(predictor, "device", None)
        return str(value) if value is not None else None

    def _load(self) -> Any:
        if self._model is None:
            if not self.available:
                raise FileNotFoundError(f"Ultralytics model does not exist: {self.model_path}")
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path), task=self.task)
        return self._model

    def warmup(self) -> bool:
        if not self.available:
            return False
        model = self._load()
        image = np.zeros(
            (self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            device=self.device,
            verbose=False,
        )
        return True

    def detect(self, image: UInt8Image) -> list[DetectionModel]:
        model = self._load()
        results = model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confidence = boxes.conf.cpu().numpy()
        polygons = result.masks.xy if result.masks is not None else []

        detections: list[DetectionModel] = []
        for index, (box, score) in enumerate(zip(xyxy, confidence, strict=True)):
            x1, y1, x2, y2 = (float(value) for value in box)
            # Some native checkpoints can emit a degenerate boundary box at
            # larger inference sizes. It is not a usable detection and must not
            # crash the entire frame pipeline.
            if not np.isfinite((x1, y1, x2, y2, float(score))).all():
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            points: list[tuple[float, float]] = []
            if index < len(polygons):
                polygon = np.asarray(polygons[index])
                points = [(float(x), float(y)) for x, y in polygon]
            detections.append(
                DetectionModel(
                    detection_id=f"deep-{index}",
                    bbox=BoundingBoxModel(x1=x1, y1=y1, x2=x2, y2=y2),
                    mask=MaskModel(points=points) if points else None,
                    confidence=float(score),
                    detector=self.name,
                )
            )
        return detections


class UltralyticsSegmenter(UltralyticsDetector):
    """Backward-compatible segmentation-only constructor."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.50,
        device: str = "auto",
        image_size: int = 640,
    ) -> None:
        super().__init__(
            model_path,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=device,
            image_size=image_size,
            task="segment",
            profile_name="legacy-segmentation",
        )
