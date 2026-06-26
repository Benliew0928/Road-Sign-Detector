from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from pathlib import Path
from typing import Any

import numpy as np

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.inference.models import (
    BoundingBoxModel,
    DetectionModel,
    MaskModel,
)
from roadsign_assist.paths import project_path


class UltralyticsSegmenter:
    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.50,
        device: str = "auto",
        image_size: int = 640,
    ) -> None:
        self.model_path = project_path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = None if device == "auto" else device
        self.image_size = image_size
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return f"ultralytics:{self.model_path.name}"

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
                raise FileNotFoundError(f"Segmentation model does not exist: {self.model_path}")
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path), task="segment")
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
