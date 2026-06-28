from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

SEMANTIC_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


class SignCategory(StrEnum):
    REGULATORY = "regulatory"
    WARNING = "warning"
    MANDATORY = "mandatory"
    INFORMATION = "information"
    TEMPORARY = "temporary"
    TEXT = "text"


class Severity(StrEnum):
    INFORMATION = "information"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


class ParameterType(StrEnum):
    NONE = "none"
    SPEED = "speed"
    HEIGHT = "height"
    WIDTH = "width"
    WEIGHT = "weight"
    AXLE_WEIGHT = "axle_weight"
    TIME = "time"
    DISTANCE = "distance"
    DIRECTION = "direction"


class ActionCode(StrEnum):
    STOP_REQUEST = "STOP_REQUEST"
    YIELD = "YIELD"
    REDUCE_SPEED = "REDUCE_SPEED"
    SET_TARGET_SPEED = "SET_TARGET_SPEED"
    PROHIBIT_LEFT_TURN = "PROHIBIT_LEFT_TURN"
    PROHIBIT_RIGHT_TURN = "PROHIBIT_RIGHT_TURN"
    PROHIBIT_U_TURN = "PROHIBIT_U_TURN"
    PROHIBIT_DIRECTION = "PROHIBIT_DIRECTION"
    PROHIBIT_LANE_CHANGE = "PROHIBIT_LANE_CHANGE"
    PROHIBIT_ENTRY = "PROHIBIT_ENTRY"
    PROHIBIT_OVERTAKING = "PROHIBIT_OVERTAKING"
    PROHIBIT_VEHICLE = "PROHIBIT_VEHICLE"
    PROHIBIT_PARKING = "PROHIBIT_PARKING"
    PROHIBIT_STOPPING = "PROHIBIT_STOPPING"
    PROHIBIT_HORN = "PROHIBIT_HORN"
    KEEP_LEFT = "KEEP_LEFT"
    KEEP_RIGHT = "KEEP_RIGHT"
    FOLLOW_DIRECTION = "FOLLOW_DIRECTION"
    SOUND_HORN = "SOUND_HORN"
    WATCH_PEDESTRIANS = "WATCH_PEDESTRIANS"
    WATCH_CHILDREN = "WATCH_CHILDREN"
    WATCH_CYCLISTS = "WATCH_CYCLISTS"
    WATCH_ROAD_HAZARD = "WATCH_ROAD_HAZARD"
    WATCH_TRAFFIC_SIGNAL = "WATCH_TRAFFIC_SIGNAL"
    WATCH_RAILWAY = "WATCH_RAILWAY"
    WATCH_ANIMALS = "WATCH_ANIMALS"
    HEIGHT_RESTRICTION = "HEIGHT_RESTRICTION"
    WIDTH_RESTRICTION = "WIDTH_RESTRICTION"
    WEIGHT_RESTRICTION = "WEIGHT_RESTRICTION"
    INFORMATION_ONLY = "INFORMATION_ONLY"
    UNKNOWN_CAUTION = "UNKNOWN_CAUTION"


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"


class LocalizedText(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    en: str = Field(min_length=1, max_length=120)
    ms: str = Field(min_length=1, max_length=120)
    zh: str = Field(min_length=1, max_length=120)


class StandardReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    document: str = Field(min_length=1)
    section: str = Field(min_length=1)
    url: HttpUrl


class StandardDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    title: str = Field(min_length=5)
    source_url: HttpUrl
    scope: list[str] = Field(min_length=1)
    local_archive_status: str = Field(min_length=2)
    local_archive_path: str | None = None
    local_archive_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    local_archive_bytes: int | None = Field(default=None, gt=0)
    notes: str = ""


class StandardsManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    retrieved_on: str
    publisher: str
    documents: list[StandardDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reference_ids(self) -> StandardsManifest:
        values = [document.reference_id for document in self.documents]
        if len(values) != len(set(values)):
            raise ValueError("standard reference IDs must be unique")
        return self


class SignDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_sign_id: str
    category: SignCategory
    names: LocalizedText
    aliases: list[str] = Field(default_factory=list)
    visual_family: str = Field(min_length=1, max_length=40)
    base_action: ActionCode
    severity: Severity
    parameter_type: ParameterType = ParameterType.NONE
    default_parameter: float | str | None = None
    audio_key: str
    standard_reference: StandardReference
    review_status: ReviewStatus = ReviewStatus.DRAFT

    @field_validator("semantic_sign_id", "audio_key")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not SEMANTIC_ID_PATTERN.fullmatch(value):
            raise ValueError("must be a lowercase snake_case identifier")
        return value

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, value: list[str]) -> list[str]:
        cleaned = [alias.strip() for alias in value if alias.strip()]
        if len(cleaned) != len(set(alias.casefold() for alias in cleaned)):
            raise ValueError("aliases must be unique, ignoring case")
        return cleaned

    @model_validator(mode="after")
    def validate_parameter(self) -> SignDefinition:
        if self.parameter_type is ParameterType.NONE and self.default_parameter is not None:
            raise ValueError("default_parameter requires a non-none parameter_type")
        return self


class SignCatalogue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    catalogue_version: str
    entries: list[SignDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_uniqueness(self) -> SignCatalogue:
        ids = [entry.semantic_sign_id for entry in self.entries]
        audio_keys = [entry.audio_key for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("semantic_sign_id values must be unique")
        if len(audio_keys) != len(set(audio_keys)):
            raise ValueError("audio_key values must be unique")
        return self
