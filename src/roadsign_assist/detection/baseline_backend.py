from __future__ import annotations

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.baseline.pipeline import process_image
from roadsign_assist.config import load_yaml
from roadsign_assist.inference.models import (
    BoundingBoxModel,
    DetectionModel,
    MaskModel,
)


class BaselineSignDetector:
    def __init__(self) -> None:
        self._config = load_yaml("configs/baseline/default.yaml")

    @property
    def name(self) -> str:
        return "color_shape_baseline"

    @property
    def available(self) -> bool:
        return True

    def warmup(self) -> bool:
        return True

    def detect(self, image: UInt8Image) -> list[DetectionModel]:
        result = process_image(
            image,
            image_id="frame",
            image_path="<memory>",
            config=self._config,
        )
        detections: list[DetectionModel] = []
        for index, candidate in enumerate(result.candidates):
            contour = candidate.contour.reshape(-1, 2)
            points = [(float(x), float(y)) for x, y in contour]
            bbox = candidate.bbox
            detections.append(
                DetectionModel(
                    detection_id=f"baseline-{index}",
                    bbox=BoundingBoxModel(
                        x1=float(bbox.x),
                        y1=float(bbox.y),
                        x2=float(bbox.x2),
                        y2=float(bbox.y2),
                    ),
                    mask=MaskModel(points=points),
                    confidence=candidate.score,
                    detector=self.name,
                    color_hint=candidate.color,
                    shape_hint=candidate.shape_label,
                )
            )
        return detections
