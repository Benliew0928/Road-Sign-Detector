from __future__ import annotations

import base64
import wave

from scripts.generate_p16_ai_audio_assets import (
    _extract_gemini_audio_data,
    _gemini_prompt,
    _is_tts_prompt_error,
    _write_pcm_wav,
)

from roadsign_assist.audio.advisory import build_advisory_manifest
from roadsign_assist.audio.ai_voice import (
    STYLE_PROFILES,
    available_sample_phrase_ids,
    parse_voice_map,
    rewrite_manifest_for_ai_pack,
)


def test_ai_voice_manifest_rewrites_assets_with_fallback_sources() -> None:
    base = build_advisory_manifest()
    voices = {"en": "marin", "ms": "marin", "zh": "cedar"}
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


def test_extract_gemini_audio_data_from_interactions_response() -> None:
    encoded = base64.b64encode(b"\x00\x01\x02\x03").decode("ascii")

    assert _extract_gemini_audio_data({"output_audio": {"data": encoded}}) == encoded
    assert (
        _extract_gemini_audio_data(
            {"content": [{"inlineData": {"mimeType": "audio/L16", "data": encoded}}]}
        )
        == encoded
    )


def test_write_pcm_wav_wraps_gemini_pcm_audio(tmp_path) -> None:
    path = tmp_path / "gemini.wav"

    _write_pcm_wav(path, b"\x00\x00\x01\x00" * 20)

    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 40


def test_gemini_prompt_error_detection_and_strict_prompt() -> None:
    prompt = _gemini_prompt(
        text="Keep left. Follow the required direction.",
        language="en",
        instructions="Speak like a calm ADAS assistant.",
    )
    strict_prompt = _gemini_prompt(
        text="Keep left. Follow the required direction.",
        language="en",
        instructions="Speak like a calm ADAS assistant.",
        strict=True,
    )

    assert prompt.startswith("Say in ")
    assert "without adding or translating anything" in prompt
    assert strict_prompt.startswith("Say exactly this English transcript")
    assert _is_tts_prompt_error(
        RuntimeError("Model tried to generate text, but it should only be used for TTS.")
    )
