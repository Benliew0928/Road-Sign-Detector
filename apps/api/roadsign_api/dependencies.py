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
