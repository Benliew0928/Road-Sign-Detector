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
    assert len(catalogue.entries) >= 90
    assert all(
        any("\u3400" <= character <= "\u9fff" for character in entry.names.zh)
        for entry in catalogue.entries
    )


def test_p2_catalogue_entries_are_owner_approved() -> None:
    catalogue = load_catalogue()
    assert all(entry.review_status == "approved" for entry in catalogue.entries)


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


def test_p2_coursework_driven_classes_are_present() -> None:
    known = catalogue_by_id()
    expected = {
        "bicycle_crossing",
        "bicycles_only",
        "motor_vehicles_only",
        "no_horn",
        "no_lane_changing",
        "no_left_or_right_turn",
        "no_motor_vehicles",
        "no_straight_ahead",
        "no_straight_or_left",
        "no_straight_or_right",
        "permitted_u_turn",
        "residential_area_ahead",
        "roadway_diverges",
        "slow_text",
        "sound_horn",
        "stop_for_checking",
        "tractors_ahead",
        "turn_left_or_right",
    }
    assert expected <= known.keys()


def test_coursework_mapping_uses_known_semantic_ids() -> None:
    path = project_path("configs/catalogue/coursework_draft_mapping.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    known = catalogue_by_id()
    mapped = {
        value["semantic_sign_id"]
        for value in payload["mappings"].values()
        if value["semantic_sign_id"]
    }
    assert mapped <= known.keys()
    assert not {
        sign_id
        for sign_id, value in payload["mappings"].items()
        if not value["semantic_sign_id"]
    }
    assert payload["mappings"]["sign_008"]["semantic_sign_id"] == "no_straight_or_left"
    assert payload["mappings"]["sign_010"]["semantic_sign_id"] == "no_straight_ahead"
    assert payload["mappings"]["sign_014"]["semantic_sign_id"] == "no_overtaking"
    assert payload["mappings"]["sign_023"]["semantic_sign_id"] == "turn_left_or_right"
    assert payload["mappings"]["sign_032"]["semantic_sign_id"] == "roadway_diverges"
    assert payload["mappings"]["sign_036"]["semantic_sign_id"] == "bicycle_crossing"
    assert payload["mappings"]["sign_045"]["semantic_sign_id"] == "residential_area_ahead"
    assert payload["mappings"]["sign_051"]["semantic_sign_id"] == "tractors_ahead"
    assert payload["mappings"]["sign_057"]["semantic_sign_id"] == "stop_for_checking"


def test_unresolved_coursework_mappings_are_explicit_review_items() -> None:
    path = project_path("configs/catalogue/coursework_draft_mapping.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    unresolved = {
        sign_id
        for sign_id, value in payload["mappings"].items()
        if not value["semantic_sign_id"]
    }
    assert unresolved == set()
