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
    """Use a fresh deployed-model session, or an explicitly configured close-up profile."""
    config = os.environ.get("ROADSIGN_CLOSE_UP_CONFIG")
    if config:
        return InferenceEngine(config)
    # Whole-image inference already bypasses detection; share deployed model weights.
    return get_engine().new_session()
