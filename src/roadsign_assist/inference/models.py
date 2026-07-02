from __future__ import annotations

# pyright: reportUnknownVariableType=false
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from roadsign_assist.catalogue.models import ActionCode, LocalizedText, Severity


class InferenceMode(StrEnum):
    BASELINE = "baseline"
    DEEP = "deep"
    AUTO = "auto"


class BoundingBoxModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_extent(self) -> BoundingBoxModel:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: BoundingBoxModel) -> float:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0


class MaskModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    encoding: str = "polygon"
    points: list[tuple[float, float]] = Field(default_factory=list)


class DetectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_id: str
    bbox: BoundingBoxModel
    mask: MaskModel | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    detector: str
    color_hint: str | None = None
    shape_hint: str | None = None


class ClassificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_sign_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    accepted: bool
    model_name: str
    top_k: list[tuple[str, float]] = Field(default_factory=list)
    unknown_score: float | None = Field(default=None, ge=0.0)
    embedding_distance: float | None = Field(default=None, ge=0.0, le=2.0)
    nearest_prototype: str | None = None
    rejection_reasons: list[str] = Field(default_factory=list)


class OCRModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    script: str = "none"
    language: str = "unknown"
    numeric_value: float | None = None
    unit: str | None = None
    semantic_sign_id: str | None = None


class ADASActionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ActionCode
    target_speed_kmh: float | None = Field(default=None, gt=0, le=160)
    restriction_value: float | None = Field(default=None, gt=0)
    restriction_unit: str | None = None
    direction: str | None = None
    advisory_only: bool = True


class ADASAdvisoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    headline: LocalizedText
    instruction: LocalizedText
    safe_to_announce: bool = True


class SignEventModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    frame_id: int = Field(ge=0)
    track_id: int = Field(ge=0)
    coursework_id: str | None = None
    semantic_sign_id: str
    meaning: LocalizedText
    ocr: OCRModel
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBoxModel
    mask: MaskModel | None = None
    action: ADASActionModel
    advisory: ADASAdvisoryModel
    severity: Severity
    latency_ms: float = Field(ge=0.0)
    device: str
    stable: bool
    should_announce: bool
    evidence: list[str] = Field(default_factory=list)


class FrameResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: int
    width: int
    height: int
    mode: InferenceMode
    latency_ms: float
    events: list[SignEventModel]
    warnings: list[str] = Field(default_factory=list)
