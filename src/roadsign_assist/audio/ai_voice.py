from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from roadsign_assist.audio.advisory import LANGUAGES, LanguageCode

StyleProfileName = Literal["calm_adas", "warm_assistant", "crisp_safety"]


@dataclass(frozen=True)
class AiVoiceStyleProfile:
    name: StyleProfileName
    label: str
    instructions: str


STYLE_PROFILES: dict[StyleProfileName, AiVoiceStyleProfile] = {
    "calm_adas": AiVoiceStyleProfile(
        name="calm_adas",
        label="Calm ADAS",
        instructions=(
            "Speak like a calm but alert Malaysian ADAS driving assistant. "
            "Use a natural human voice that is clear, warm, and not robotic. "
            "Use controlled urgency for warnings and critical phrases, but do not sound panicked. "
            "Keep the phrase concise, stable, and easy to understand inside a vehicle."
        ),
    ),
    "warm_assistant": AiVoiceStyleProfile(
        name="warm_assistant",
        label="Warm Assistant",
        instructions=(
            "Speak like a friendly but professional in-car safety assistant. "
            "Sound human, reassuring, and attentive. "
            "Give the warning with gentle urgency, then deliver the driving advice clearly. "
            "Avoid dramatic emotion, shouting, or a robotic navigation voice."
        ),
    ),
    "crisp_safety": AiVoiceStyleProfile(
        name="crisp_safety",
        label="Crisp Safety",
        instructions=(
            "Speak like a concise vehicle safety alert system with a natural human voice. "
            "Use firm, controlled urgency and short pauses between the hazard and advice. "
            "Prioritize clarity, seriousness, and fast comprehension. "
            "Do not sound cold, monotone, panicked, or overly cheerful."
        ),
    ),
}

DEFAULT_SAMPLE_PHRASE_IDS: tuple[str, ...] = (
    "unknown_sign",
    "stop",
    "give_way",
    "children_crossing",
    "pedestrian_crossing",
    "roadworks",
    "no_entry",
    "no_stopping",
    "maximum_speed",
    "speed_limit_50_kmh",
    "speed_limit_110_kmh",
    "height_limit_4_5_m",
    "width_limit_3_m",
    "weight_limit_10_t",
    "traffic_signal_ahead",
)


def available_sample_phrase_ids(manifest: dict[str, Any]) -> list[str]:
    phrases = manifest["phrases"]
    return [phrase_id for phrase_id in DEFAULT_SAMPLE_PHRASE_IDS if phrase_id in phrases]


def language_list(value: str) -> list[LanguageCode]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        return list(LANGUAGES)
    invalid = sorted(set(requested) - set(LANGUAGES))
    if invalid:
        raise ValueError(f"Unsupported language code(s): {', '.join(invalid)}")
    return [item for item in LANGUAGES if item in requested]


def parse_voice_map(value: str, *, default_voice: str) -> dict[LanguageCode, str]:
    voices = {language: default_voice for language in LANGUAGES}
    if not value.strip():
        return voices
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError("Voice map entries must look like en=Kore,ms=Kore,zh=Kore")
        language, voice = [part.strip() for part in item.split("=", 1)]
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language code in voice map: {language}")
        if not voice:
            raise ValueError(f"Voice value for {language} cannot be empty")
        voices[language] = voice
    return voices


def rewrite_manifest_for_ai_pack(
    manifest: dict[str, Any],
    *,
    public_audio_root: str,
    fallback_audio_root: str = "/audio/p16",
    provider: str,
    model: str,
    voices: dict[LanguageCode, str],
    style_profile: AiVoiceStyleProfile,
    status: str,
    selected_phrase_ids: set[str] | None = None,
) -> dict[str, Any]:
    next_manifest = copy.deepcopy(manifest)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    next_manifest["audio_pack"] = {
        "kind": "ai_tts",
        "status": status,
        "provider": provider,
        "model": model,
        "voices": voices,
        "style_profile": style_profile.name,
        "style_label": style_profile.label,
        "style_instructions": style_profile.instructions,
        "generated_at": generated_at,
        "runtime_policy": "local_assets_only",
        "fallback_audio_root": fallback_audio_root,
        "selected_phrase_ids": sorted(selected_phrase_ids) if selected_phrase_ids else None,
    }

    for phrase_id, phrase in next_manifest["phrases"].items():
        phrase_selected = selected_phrase_ids is None or phrase_id in selected_phrase_ids
        for language, asset in phrase["assets"].items():
            old_src = str(asset["src"])
            filename = Path(old_src).name
            asset["fallback_src"] = f"{fallback_audio_root}/{language}/{filename}"
            asset["src"] = f"{public_audio_root}/{language}/{filename}"
            asset["provider"] = provider
            asset["model"] = model
            asset["voice"] = voices[language]
            asset["style_profile"] = style_profile.name
            asset["generated"] = False
            asset["selected_for_generation"] = phrase_selected
            asset["sha256"] = None
            asset["bytes"] = None
            asset["duration_seconds"] = None
    return next_manifest


def phrase_audio_path(public_root: Path, asset_src: str) -> Path:
    return public_root / asset_src.lstrip("/")
