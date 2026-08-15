from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from roadsign_assist.config import load_yaml
from roadsign_assist.inference.engine import InferenceEngine


def test_engine_uses_honest_baseline_fallback(tmp_path: Path) -> None:
    image: np.ndarray[Any, np.dtype[np.uint8]] = np.full((240, 320, 3), 230, dtype=np.uint8)
    cv2.circle(image, (160, 120), 65, (0, 0, 230), thickness=15)
    config = deepcopy(load_yaml("configs/inference/default.yaml"))
    config["classifier"]["model_path"] = str(tmp_path / "missing-classifier.onnx")
    config["classifier"]["labels_path"] = str(tmp_path / "missing-labels.json")
    config["classifier"]["calibration_path"] = str(tmp_path / "missing-calibration.json")
    config_path = tmp_path / "baseline-fallback.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    engine = InferenceEngine(config_path)
    result = engine.process_frame(image)
    assert result.mode == "baseline"
    assert result.events
    assert result.events[0].semantic_sign_id == "unknown_sign"
    assert any(
        "classifier weights are unavailable" in warning.lower() for warning in result.warnings
    )
