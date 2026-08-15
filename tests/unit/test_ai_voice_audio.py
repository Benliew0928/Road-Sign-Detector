from __future__ import annotations

from roadsign_assist.audio.advisory import LanguageCode, build_advisory_manifest
from roadsign_assist.audio.ai_voice import (
    STYLE_PROFILES,
    available_sample_phrase_ids,
    parse_voice_map,
    rewrite_manifest_for_ai_pack,
)


def test_ai_voice_manifest_rewrites_assets_with_fallback_sources() -> None:
    base = build_advisory_manifest()
    voices: dict[LanguageCode, str] = {
        "en": "marin",
        "ms": "marin",
        "zh": "cedar",
    }
    manifest = rewrite_manifest_for_ai_pack(
        base,
        public_audio_root="/audio/p16_ai",
        fallback_audio_root="/audio/p16",
        provider="openai",
        model="gpt-4o-mini-tts",
        voices=voices,
        style_profile=STYLE_PROFILES["calm_adas"],
        status="sample",
        selected_phrase_ids={"unknown_sign"},
    )

    asset = manifest["phrases"]["unknown_sign"]["assets"]["en"]
    assert asset["src"] == "/audio/p16_ai/en/unknown_sign.wav"
    assert asset["fallback_src"] == "/audio/p16/en/unknown_sign.wav"
    assert asset["provider"] == "openai"
    assert asset["model"] == "gpt-4o-mini-tts"
    assert asset["voice"] == "marin"
    assert asset["style_profile"] == "calm_adas"
    assert asset["selected_for_generation"] is True

    non_sample_asset = manifest["phrases"]["stop"]["assets"]["en"]
    assert non_sample_asset["selected_for_generation"] is False
    assert manifest["audio_pack"]["runtime_policy"] == "local_assets_only"


def test_sample_phrase_selection_only_returns_existing_manifest_phrases() -> None:
    manifest = build_advisory_manifest()
    selected = available_sample_phrase_ids(manifest)

    assert "unknown_sign" in selected
    assert "speed_limit_50_kmh" in selected
    assert selected
    assert set(selected) <= set(manifest["phrases"])


def test_parse_voice_map_allows_language_specific_voices() -> None:
    voices = parse_voice_map("en=Kore,ms=Puck,zh=Kore", default_voice="Kore")

    assert voices == {"en": "Kore", "ms": "Puck", "zh": "Kore"}
