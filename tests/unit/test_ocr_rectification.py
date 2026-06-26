from pathlib import Path

import cv2
import numpy as np

from roadsign_assist.ocr.engine import MultilingualOCREngine


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
