from roadsign_assist.ocr.normalization import (
    detect_script,
    extract_numeric_parameter,
    normalize_ocr_text,
)


def test_ocr_normalization_and_numeric_extraction() -> None:
    assert normalize_ocr_text("  HAD   LAJU  50 KM/H ") == "HAD LAJU 50 KM/H"
    parameter = extract_numeric_parameter("HAD LAJU 50 KM/H")
    assert parameter is not None
    assert parameter.value == 50
    assert parameter.unit == "KM/H"


def test_script_detection_supports_mixed_text() -> None:
    assert detect_script("学校 SCHOOL") == "mixed"
