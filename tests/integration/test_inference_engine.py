from typing import Any

import cv2
import numpy as np

from roadsign_assist.inference.engine import InferenceEngine


def test_engine_uses_honest_baseline_fallback() -> None:
    image: np.ndarray[Any, np.dtype[np.uint8]] = np.full((240, 320, 3), 230, dtype=np.uint8)
    cv2.circle(image, (160, 120), 65, (0, 0, 230), thickness=15)
    engine = InferenceEngine()
    result = engine.process_frame(image)
    assert result.mode == "baseline"
    assert result.events
    assert result.events[0].semantic_sign_id == "unknown_sign"
    assert any(
        "classifier weights are unavailable" in warning.lower() for warning in result.warnings
    )
