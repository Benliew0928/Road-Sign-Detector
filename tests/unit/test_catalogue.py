import json

from roadsign_assist.catalogue.repository import (
    catalogue_by_id,
    load_catalogue,
    load_standards_manifest,
    match_alias,
    match_alias_fuzzy,
)
from roadsign_assist.paths import project_path


def test_catalogue_is_valid_and_broad() -> None:
    catalogue = load_catalogue()
    assert len(catalogue.entries) >= 60
    assert all(
        any("\u3400" <= character <= "\u9fff" for character in entry.names.zh)
        for entry in catalogue.entries
    )


def test_alias_matching_is_case_insensitive() -> None:
    sign = match_alias("  zon sekolah ")
    assert sign is not None
    assert sign.semantic_sign_id == "school_zone"


def test_every_catalogue_reference_is_registered() -> None:
    catalogue = load_catalogue()
    standards = load_standards_manifest()
    known = {document.reference_id for document in standards.documents}
    assert all(entry.standard_reference.reference_id in known for entry in catalogue.entries)
    assert all(
        str(entry.standard_reference.url).startswith("https://") for entry in catalogue.entries
    )


def test_fuzzy_alias_matching_is_conservative() -> None:
    match = match_alias_fuzzy("BERHNTI")
    assert match is not None
    assert match.sign.semantic_sign_id == "stop"
    assert match.score >= 0.88
    assert match_alias_fuzzy("ROAD") is None


def test_emtd_draft_mapping_uses_known_semantic_ids() -> None:
    path = project_path("configs/catalogue/emtd_class_mapping.v0.1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    known = catalogue_by_id()
    mapped = {
        value["semantic_sign_id"]
        for value in payload["classes"].values()
        if value["semantic_sign_id"]
    }
    assert len(payload["classes"]) == 66
    assert mapped <= known.keys()
