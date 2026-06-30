from __future__ import annotations

import json

from roadsign_assist.audio.advisory import build_advisory_manifest
from roadsign_assist.catalogue.repository import load_catalogue
from roadsign_assist.paths import PROJECT_ROOT


def test_advisory_manifest_covers_every_catalogue_audio_key() -> None:
    catalogue = load_catalogue()
    manifest = build_advisory_manifest()
    phrases = manifest["phrases"]

    assert manifest["fallback_phrase_id"] in phrases
    for entry in catalogue.entries:
        phrase_id = manifest["audio_key_phrase_ids"][entry.audio_key]
        phrase = phrases[phrase_id]
        assert phrase["semantic_sign_id"] == entry.semantic_sign_id
        assert phrase["audio_key"] == entry.audio_key
        for language in ("en", "ms", "zh"):
            assert phrase["text"][language].strip()
            assert phrase["assets"][language]["src"].endswith(".wav")


def test_generated_audio_manifest_has_all_assets() -> None:
    manifest_path = PROJECT_ROOT / "apps/web/public/audio/p16/advisory_audio_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    public_root = PROJECT_ROOT / "apps/web/public"

    assert len(manifest["phrases"]) >= len(load_catalogue().entries)
    for phrase in manifest["phrases"].values():
        for language in manifest["languages"]:
            asset = phrase["assets"][language]
            path = public_root / str(asset["src"]).lstrip("/")
            assert path.exists(), path
            assert asset["generated"] is True
            assert asset["sha256"]
            assert asset["bytes"] == path.stat().st_size
            assert asset["duration_seconds"] > 0


def test_speed_and_restriction_variants_are_available() -> None:
    manifest = build_advisory_manifest()
    variants = manifest["variant_phrase_ids"]

    assert variants["speed_limit_kmh"]["50"] == "speed_limit_50_kmh"
    assert variants["height_limit_m"]["4.5"] == "height_limit_4_5_m"
    assert variants["width_limit_m"]["3.5"] == "width_limit_3_5_m"
    assert variants["weight_limit_t"]["10"] == "weight_limit_10_t"
