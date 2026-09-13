from __future__ import annotations

import os
from functools import lru_cache

from roadsign_assist.inference.engine import InferenceEngine


@lru_cache(maxsize=1)
def get_engine() -> InferenceEngine:
    return InferenceEngine(
        os.environ.get(
            "ROADSIGN_CONFIG",
            "configs/inference/default.yaml",
        )
    )


@lru_cache(maxsize=1)
def get_close_up_engine() -> InferenceEngine:
    """Load the isolated close-up classifier profile without changing road inference."""
    return InferenceEngine(
        os.environ.get(
            "ROADSIGN_CLOSE_UP_CONFIG",
            "configs/inference/local_recovery_week1/close-up-repaired-v1.yaml",
        )
    )
