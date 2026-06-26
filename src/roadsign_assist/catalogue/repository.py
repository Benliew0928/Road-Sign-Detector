from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from roadsign_assist.catalogue.models import (
    SignCatalogue,
    SignDefinition,
    StandardsManifest,
)
from roadsign_assist.paths import project_path

DEFAULT_CATALOGUE_PATH = Path("configs/catalogue/malaysia_signs.v1.json")
DEFAULT_STANDARDS_PATH = Path("configs/catalogue/standards_manifest.json")


@dataclass(frozen=True)
class AliasMatch:
    sign: SignDefinition
    score: float
    exact: bool


@lru_cache(maxsize=4)
def load_catalogue(path: str | Path = DEFAULT_CATALOGUE_PATH) -> SignCatalogue:
    resolved = project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SignCatalogue.model_validate(payload)


def load_default_catalogue() -> tuple[SignDefinition, ...]:
    return tuple(load_catalogue().entries)


@lru_cache(maxsize=2)
def load_standards_manifest(
    path: str | Path = DEFAULT_STANDARDS_PATH,
) -> StandardsManifest:
    resolved = project_path(path)
    return StandardsManifest.model_validate_json(resolved.read_text(encoding="utf-8"))


def catalogue_by_id(path: str | Path = DEFAULT_CATALOGUE_PATH) -> dict[str, SignDefinition]:
    catalogue = load_catalogue(path)
    return {entry.semantic_sign_id: entry for entry in catalogue.entries}


def match_alias(text: str, path: str | Path = DEFAULT_CATALOGUE_PATH) -> SignDefinition | None:
    normalized = " ".join(text.casefold().split())
    if not normalized:
        return None
    for entry in load_catalogue(path).entries:
        values = [entry.names.en, entry.names.ms, entry.names.zh, *entry.aliases]
        if normalized in {" ".join(value.casefold().split()) for value in values}:
            return entry
    return None


def match_alias_fuzzy(
    text: str,
    path: str | Path = DEFAULT_CATALOGUE_PATH,
    *,
    minimum_score: float = 0.88,
    minimum_margin: float = 0.08,
) -> AliasMatch | None:
    normalized = " ".join(text.casefold().split())
    compact = "".join(character for character in normalized if character.isalnum())
    if len(compact) < 4:
        return None

    candidates: list[tuple[float, SignDefinition]] = []
    for entry in load_catalogue(path).entries:
        values = [entry.names.en, entry.names.ms, entry.names.zh, *entry.aliases]
        score = max(
            SequenceMatcher(
                None,
                compact,
                "".join(character for character in value.casefold() if character.isalnum()),
                autojunk=False,
            ).ratio()
            for value in values
        )
        candidates.append((score, entry))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_sign = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    if best_score < minimum_score or best_score - second_score < minimum_margin:
        return None
    return AliasMatch(sign=best_sign, score=best_score, exact=False)
