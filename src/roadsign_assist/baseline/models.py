from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

UInt8Image = NDArray[np.uint8]
Contour = NDArray[np.int32]


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def intersection_over_union(self, other: BoundingBox) -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0


@dataclass(frozen=True)
class Candidate:
    color: str
    bbox: BoundingBox
    contour: Contour
    area: float
    area_ratio: float
    aspect_ratio: float
    extent: float
    solidity: float
    circularity: float
    polygon_vertices: int
    shape_label: str
    score: float

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("contour")
        value["bbox"] = asdict(self.bbox)
        return value


@dataclass(frozen=True)
class BaselineResult:
    image_id: str
    image_path: str
    width: int
    height: int
    runtime_ms: float
    candidates: tuple[Candidate, ...]
    masks: dict[str, UInt8Image]
