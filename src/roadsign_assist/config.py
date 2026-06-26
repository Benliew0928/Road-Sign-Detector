from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from roadsign_assist.paths import project_path


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_device: str = "auto"
    max_frame_queue: int = Field(default=2, ge=1, le=10)
    max_image_dimension: int = Field(default=1280, ge=320, le=4096)


class ApplicationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 2513
    runtime: RuntimeSettings = RuntimeSettings()


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value: object = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {resolved}")
    return cast(dict[str, Any], value)


@lru_cache(maxsize=1)
def load_application_settings() -> ApplicationSettings:
    raw = load_yaml("params.yaml")
    return ApplicationSettings.model_validate(
        {
            "seed": raw.get("seed", 2513),
            "runtime": raw.get("runtime", {}),
        }
    )
