from pathlib import Path

import cv2
import numpy as np

from roadsign_assist.ocr.engine import MultilingualOCREngine
from roadsign_assist.inference.models import OCRModel


def test_ocr_rectification_warps_a_large_quadrilateral() -> None:
    image = np.zeros((220, 280, 3), dtype=np.uint8)
    polygon = np.asarray([[45, 35], [245, 55], [225, 190], [25, 170]], dtype=np.int32)
    cv2.fillConvexPoly(image, polygon, (245, 245, 245))
    cv2.putText(
        image,
        "50",
        (95, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (20, 20, 20),
        5,
        cv2.LINE_AA,
    )
    rectified = MultilingualOCREngine.rectify(image)
    assert rectified.dtype == np.uint8
    assert rectified.ndim == 3
    assert rectified.shape[0] >= 100
    assert rectified.shape[1] >= 150


def test_ocr_is_unavailable_without_local_assets(tmp_path: Path) -> None:
    engine = MultilingualOCREngine(model_root=tmp_path)
    assert engine.available is False


def test_numeric_second_pass_only_runs_for_numeric_hint() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, image: np.ndarray) -> list[object]:
            del image
            self.calls += 1
            text = "" if self.calls == 1 else "80"
            return [type("Result", (), {"json": {"res": {"rec_texts": [text], "rec_scores": [0.9]}}})()]

    engine = MultilingualOCREngine(numeric_second_pass=True)
    fake = FakeEngine()
    engine._engine = fake
    engine._load_attempted = True
    image = np.full((120, 120, 3), 255, dtype=np.uint8)

    result = engine.recognize(image, semantic_hint="maximum_speed")

    assert result.numeric_value == 80
    assert fake.calls == 2


def test_conflicting_numeric_passes_suppress_actionable_value(monkeypatch) -> None:
    engine = MultilingualOCREngine(numeric_second_pass=True)
    engine._engine = object()
    engine._load_attempted = True
    outputs = iter(
        [
            OCRModel(text="80", confidence=0.9, numeric_value=80),
            OCRModel(text="30", confidence=0.9, numeric_value=30),
        ]
    )
    monkeypatch.setattr(engine, "_recognize_view", lambda image: next(outputs))
    image = np.full((120, 120, 3), 255, dtype=np.uint8)

    result = engine.recognize(image, semantic_hint="maximum_speed")

    assert result.text == "80"
    assert result.numeric_value is None


def test_numeric_second_pass_is_not_used_for_non_numeric_sign(monkeypatch) -> None:
    engine = MultilingualOCREngine(numeric_second_pass=True)
    engine._engine = object()
    engine._load_attempted = True
    calls: list[int] = []
    monkeypatch.setattr(
        engine,
        "_recognize_view",
        lambda image: calls.append(1) or OCRModel(text="STOP", confidence=0.9),
    )
    image = np.full((120, 120, 3), 255, dtype=np.uint8)

    engine.recognize(image, semantic_hint="stop")

    assert len(calls) == 1


def test_dimension_hint_repairs_dropped_decimal() -> None:
    result = MultilingualOCREngine._normalize_numeric_for_hint(
        OCRModel(text="32 M", confidence=0.9, numeric_value=32, unit="M"),
        "width_restriction",
    )

    assert result.numeric_value == 3.2
    assert result.unit == "M"


def test_speed_hint_does_not_reinterpret_integer() -> None:
    result = MultilingualOCREngine._normalize_numeric_for_hint(
        OCRModel(text="80", confidence=0.9, numeric_value=80),
        "maximum_speed",
    )

    assert result.numeric_value == 80
