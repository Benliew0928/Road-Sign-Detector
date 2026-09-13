from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import importlib.util
from pathlib import Path
from typing import Any, cast

import cv2

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.catalogue.repository import match_alias, match_alias_fuzzy
from roadsign_assist.inference.models import OCRModel
from roadsign_assist.ocr.normalization import (
    detect_script,
    extract_numeric_parameter,
    normalize_ocr_text,
)
from roadsign_assist.paths import project_path


NUMERIC_SIGN_RANGES: dict[str, tuple[float, float]] = {
    "maximum_speed": (5.0, 160.0),
    "height_restriction": (0.5, 10.0),
    "width_restriction": (0.5, 10.0),
    "weight_restriction": (0.5, 100.0),
}


class MultilingualOCREngine:
    def __init__(
        self,
        enabled: bool = True,
        model_root: str | Path = "models/ocr",
        numeric_second_pass: bool = False,
    ) -> None:
        self.enabled = enabled
        self.model_root = project_path(model_root)
        self.numeric_second_pass = numeric_second_pass
        self._engine: Any | None = None
        self._load_attempted = False
        self.load_error: str | None = None

    @property
    def available(self) -> bool:
        required = (
            self.model_root / "PP-OCRv6_small_det" / "inference.pdiparams",
            self.model_root / "PP-OCRv6_small_rec" / "inference.pdiparams",
            self.model_root / "manifest.json",
        )
        return (
            self.enabled
            and importlib.util.find_spec("paddleocr") is not None
            and all(path.is_file() for path in required)
        )

    @property
    def loaded(self) -> bool:
        return self._load_attempted and self._engine is not None

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from paddleocr import PaddleOCR

            from roadsign_assist.ocr.assets import verify_ocr_assets

            verify_ocr_assets(self.model_root)
            self._engine = PaddleOCR(
                text_detection_model_name="PP-OCRv6_small_det",
                text_detection_model_dir=str(self.model_root / "PP-OCRv6_small_det"),
                text_recognition_model_name="PP-OCRv6_small_rec",
                text_recognition_model_dir=str(self.model_root / "PP-OCRv6_small_rec"),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                enable_mkldnn=False,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self._engine = None
            self.load_error = str(exc)

    def warmup(self) -> bool:
        if not self.available:
            return False
        self._ensure_loaded()
        return self.loaded

    @staticmethod
    def rectify(crop: UInt8Image) -> UInt8Image:
        import numpy as np

        source = crop
        gray_source = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_source, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = crop.shape[0] * crop.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < image_area * 0.18:
                break
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            points = polygon.reshape(4, 2).astype(np.float32)
            sums = points.sum(axis=1)
            differences = np.diff(points, axis=1).reshape(-1)
            ordered = np.asarray(
                [
                    points[int(np.argmin(sums))],
                    points[int(np.argmin(differences))],
                    points[int(np.argmax(sums))],
                    points[int(np.argmax(differences))],
                ],
                dtype=np.float32,
            )
            top_left, top_right, bottom_right, bottom_left = ordered
            width = round(
                max(
                    np.linalg.norm(top_right - top_left),
                    np.linalg.norm(bottom_right - bottom_left),
                )
            )
            height = round(
                max(
                    np.linalg.norm(bottom_left - top_left),
                    np.linalg.norm(bottom_right - top_right),
                )
            )
            if width < 16 or height < 16:
                continue
            destination = np.asarray(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            )
            transform = cv2.getPerspectiveTransform(ordered, destination)
            source = cv2.warpPerspective(crop, transform, (width, height))
            break

        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        sharpened = cv2.addWeighted(
            enhanced,
            1.6,
            cv2.GaussianBlur(enhanced, (0, 0), 1.2),
            -0.6,
            0,
        )
        return cast(UInt8Image, cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR))

    @staticmethod
    def numeric_view(rectified: UInt8Image) -> UInt8Image:
        """Build one high-contrast view for numeric signs without changing source pixels."""
        import numpy as np

        height, width = rectified.shape[:2]
        scale = max(1.0, 224.0 / max(1, min(height, width)))
        enlarged = cv2.resize(
            rectified,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
        padded = cv2.copyMakeBorder(binary, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)
        return cast(UInt8Image, cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR))

    def _recognize_view(self, image: UInt8Image) -> OCRModel:
        if self._engine is None:
            return OCRModel()
        result = self._engine.predict(image)
        texts: list[str] = []
        scores: list[float] = []
        for item in result:
            data = item.json if hasattr(item, "json") else {}
            payload = data.get("res", data) if isinstance(data, dict) else {}
            texts.extend(str(value) for value in payload.get("rec_texts", []))
            scores.extend(float(value) for value in payload.get("rec_scores", []))
        text = normalize_ocr_text(" ".join(texts))
        confidence = sum(scores) / len(scores) if scores else 0.0
        numeric = extract_numeric_parameter(text)
        exact_match = match_alias(text)
        fuzzy_match = match_alias_fuzzy(text) if exact_match is None else None
        matched_sign = exact_match or (fuzzy_match.sign if fuzzy_match else None)
        return OCRModel(
            text=text,
            confidence=confidence,
            script=detect_script(text),
            language="zh_or_latin" if text else "unknown",
            numeric_value=numeric.value if numeric else None,
            unit=numeric.unit if numeric else None,
            semantic_sign_id=matched_sign.semantic_sign_id if matched_sign else None,
        )

    @staticmethod
    def _plausible_numeric(result: OCRModel, semantic_hint: str) -> bool:
        limits = NUMERIC_SIGN_RANGES.get(semantic_hint)
        return (
            limits is not None
            and result.numeric_value is not None
            and limits[0] <= result.numeric_value <= limits[1]
        )

    @staticmethod
    def _normalize_numeric_for_hint(result: OCRModel, semantic_hint: str) -> OCRModel:
        """Repair a dropped decimal only where an integer cannot be a plausible dimension."""
        value = result.numeric_value
        if (
            semantic_hint in {"height_restriction", "width_restriction"}
            and value is not None
            and 10.0 < value <= 99.0
            and "." not in result.text
            and "," not in result.text
        ):
            return result.model_copy(update={"numeric_value": value / 10.0})
        return result

    def recognize(self, crop: UInt8Image, *, semantic_hint: str | None = None) -> OCRModel:
        self._ensure_loaded()
        if self._engine is None:
            return OCRModel()
        rectified = self.rectify(crop)
        primary = self._recognize_view(rectified)
        if not self.numeric_second_pass or semantic_hint not in NUMERIC_SIGN_RANGES:
            return primary

        primary = self._normalize_numeric_for_hint(primary, semantic_hint)
        secondary = self._normalize_numeric_for_hint(
            self._recognize_view(self.numeric_view(rectified)), semantic_hint
        )
        primary_valid = self._plausible_numeric(primary, semantic_hint)
        secondary_valid = self._plausible_numeric(secondary, semantic_hint)
        if primary_valid and secondary_valid:
            if abs(float(primary.numeric_value) - float(secondary.numeric_value)) <= 0.05:
                return primary if primary.confidence >= secondary.confidence else secondary
            # Conflicting numeric readings are unsafe. Preserve the text for diagnosis,
            # but suppress the value so semantic rules cannot emit a restriction action.
            return primary.model_copy(update={"numeric_value": None, "unit": None})
        if secondary_valid:
            return secondary
        return primary
