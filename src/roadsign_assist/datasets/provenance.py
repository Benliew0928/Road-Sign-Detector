from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class LicenceDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVIEW_REQUIRED = "review_required"


class DatasetSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    name: str = Field(min_length=2)
    owner: str = Field(min_length=2)
    publisher: str = Field(min_length=2)
    source_url: HttpUrl
    doi: str | None = None
    licence_name: str = Field(min_length=2)
    licence_url: HttpUrl | None = None
    decision: LicenceDecision
    metadata_retrieved_on: date
    downloaded_on: date | None = None
    archive_url: HttpUrl | None = None
    archive_size_bytes: int | None = Field(default=None, gt=0)
    expected_archive_md5: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{32}$")
    archive_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    full_archive_status: str
    local_metadata_files: list[str] = Field(default_factory=list)
    geographic_relevance: str
    class_mapping_status: str
    attribution: str
    restrictions: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def require_download_evidence(self) -> DatasetSource:
        if self.downloaded_on is not None and self.archive_sha256 is None:
            raise ValueError("downloaded sources require archive_sha256")
        if self.decision is LicenceDecision.ACCEPTED and self.licence_url is None:
            raise ValueError("accepted sources require a licence URL")
        return self


class ProvenanceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    sources: list[DatasetSource]
