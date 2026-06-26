from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

SPACE_PATTERN = re.compile(r"\s+")
NUMBER_UNIT_PATTERN = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>KM/?H|KPH|KMJ|M|CM|T|TON|TAN)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumericParameter:
    value: float
    unit: str | None


def normalize_ocr_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\u2014", "-").replace("\u2013", "-")
    return SPACE_PATTERN.sub(" ", normalized).strip()


def extract_numeric_parameter(value: str) -> NumericParameter | None:
    normalized = normalize_ocr_text(value)
    match = NUMBER_UNIT_PATTERN.search(normalized)
    if not match:
        return None
    number = float(match.group("value").replace(",", "."))
    unit = match.group("unit")
    if unit:
        unit = unit.upper().replace("KPH", "KM/H").replace("KMJ", "KM/H")
        unit = unit.replace("TON", "T").replace("TAN", "T")
    return NumericParameter(value=number, unit=unit)


def detect_script(value: str) -> str:
    has_chinese = any("\u3400" <= character <= "\u9fff" for character in value)
    has_latin = any("A" <= character.upper() <= "Z" for character in value)
    has_digit = any(character.isdigit() for character in value)
    if has_chinese and has_latin:
        return "mixed"
    if has_chinese:
        return "chinese"
    if has_latin:
        return "latin"
    if has_digit:
        return "numeric"
    return "none"
