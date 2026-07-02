from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, cast

from roadsign_assist.audio.advisory import (
    build_advisory_manifest,
    update_asset_metadata,
    write_manifest,
)
from roadsign_assist.audio.ai_voice import (
    STYLE_PROFILES,
    StyleProfileName,
    available_sample_phrase_ids,
    language_list,
    parse_voice_map,
    phrase_audio_path,
    rewrite_manifest_for_ai_pack,
)
from roadsign_assist.paths import PROJECT_ROOT

GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
OPENAI_SPEECH_URL = "https://api.openai.com/v1/audio/speech"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-tts-preview"
DEFAULT_GEMINI_VOICE = "Kore"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini-tts"
DEFAULT_OPENAI_VOICE = "marin"
PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_base_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return build_advisory_manifest()
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_wav(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        if frames <= 0 or rate <= 0:
            raise ValueError(f"{path} has no playable frames")
        if channels != 1:
            raise ValueError(f"{path} is not mono")
        if sample_width != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        return round(frames / float(rate), 3)


def _write_pcm_wav(path: Path, pcm_data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH)
        wav_file.setframerate(PCM_SAMPLE_RATE)
        wav_file.writeframes(pcm_data)


def _provider_defaults(provider: str) -> tuple[str, str]:
    if provider == "gemini":
        return DEFAULT_GEMINI_MODEL, DEFAULT_GEMINI_VOICE
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL, DEFAULT_OPENAI_VOICE
    raise ValueError(f"Unsupported provider: {provider}")


def _is_quota_error(cause: Exception) -> bool:
    message = str(cause).lower()
    return "http 429" in message or "too_many_requests" in message or "quota" in message


def _is_timeout_error(cause: Exception) -> bool:
    message = str(cause).lower()
    return "timed out" in message or "timeout" in message or "read operation" in message


def _is_tts_prompt_error(cause: Exception) -> bool:
    message = str(cause).lower()
    return "tried to generate text" in message or "only be used for tts" in message


def _request_openai_speech(
    *,
    api_key: str,
    model: str,
    voice: str,
    text: str,
    language: str,
    instructions: str,
    timeout_seconds: int,
) -> bytes:
    # Keep multiple payload variants because the speech API has both legacy and
    # newer field names in circulation. The first successful variant wins.
    payloads: list[dict[str, Any]] = [
        {
            "model": model,
            "voice": voice,
            "input": text,
            "instructions": instructions,
            "response_format": "wav",
            "language": language,
        },
        {
            "model": model,
            "voice": voice,
            "input": text,
            "instructions": instructions,
            "response_format": "wav",
        },
        {
            "model": model,
            "voice": voice,
            "input": text,
            "instructions": instructions,
            "format": "wav",
            "language": language,
        },
        {
            "model": model,
            "voice": voice,
            "input": text,
            "instructions": instructions,
            "format": "wav",
        },
    ]
    last_error = ""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for payload in payloads:
        request = urllib.request.Request(
            OPENAI_SPEECH_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = response.read()
                if not data.startswith(b"RIFF"):
                    raise RuntimeError("Speech API did not return a WAV payload")
                return data
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code not in {400, 422}:
                break
        except urllib.error.URLError as exc:
            last_error = str(exc)
            break
    raise RuntimeError(f"OpenAI speech generation failed. Last error: {last_error}")


def _gemini_prompt(*, text: str, language: str, instructions: str, strict: bool = False) -> str:
    language_names = {
        "en": "English",
        "ms": "Bahasa Melayu",
        "zh": "Mandarin Chinese",
    }
    if strict:
        return f'Say exactly this {language_names.get(language, language)} transcript:\n"{text}"'
    style = (
        "a calm, alert Malaysian driver-assistance voice; natural, clear, and not robotic"
    )
    if "warm" in instructions.lower():
        style = "a warm, professional in-car safety assistant voice"
    if "crisp" in instructions.lower():
        style = "a crisp, serious vehicle safety alert voice"
    return (
        f"Say in {style}. "
        f"Read only this {language_names.get(language, language)} transcript, without adding or translating anything:\n"
        f'"{text}"'
    )


def _extract_gemini_audio_data(response: dict[str, Any]) -> str:
    for key in ("output_audio", "outputAudio"):
        output_audio = response.get(key)
        if isinstance(output_audio, dict) and isinstance(output_audio.get("data"), str):
            return output_audio["data"]

    stack: list[Any] = [response]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            data = item.get("data")
            mime_type = item.get("mime_type") or item.get("mimeType")
            if isinstance(data, str) and isinstance(mime_type, str) and "audio" in mime_type:
                return data
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    raise RuntimeError("Gemini response did not include output_audio.data")


def _request_gemini_speech(
    *,
    api_key: str,
    model: str,
    voice: str,
    text: str,
    language: str,
    instructions: str,
    timeout_seconds: int,
    strict_prompt: bool = False,
) -> bytes:
    payload = {
        "model": model,
        "input": _gemini_prompt(
            text=text,
            language=language,
            instructions=instructions,
            strict=strict_prompt,
        ),
        "response_format": {"type": "audio"},
        "generation_config": {
            "speech_config": [
                {"voice": voice},
            ]
        },
    }
    request = urllib.request.Request(
        GEMINI_INTERACTIONS_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini speech generation failed. HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini speech generation failed: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Gemini speech generation failed: read operation timed out") from exc

    encoded_audio = _extract_gemini_audio_data(response_json)
    try:
        return base64.b64decode(encoded_audio)
    except ValueError as exc:
        raise RuntimeError("Gemini output_audio.data was not valid base64") from exc


def _selected_phrase_ids(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    if args.phrase_ids:
        phrase_ids = _csv(args.phrase_ids)
    elif args.mode == "sample":
        phrase_ids = available_sample_phrase_ids(manifest)
    else:
        phrase_ids = sorted(manifest["phrases"])
    if args.max_phrases:
        phrase_ids = phrase_ids[: args.max_phrases]
    missing = [phrase_id for phrase_id in phrase_ids if phrase_id not in manifest["phrases"]]
    if missing:
        raise ValueError(f"Unknown phrase ID(s): {', '.join(missing)}")
    return phrase_ids


def _default_audio_root(mode: str, style_profile: str) -> str:
    if mode == "sample":
        return f"/audio/p16_ai_samples/{style_profile}"
    return "/audio/p16_ai"


def _default_output_manifest(mode: str, style_profile: str) -> Path:
    if mode == "sample":
        return (
            PROJECT_ROOT
            / "outputs"
            / "audio"
            / "p16_ai_samples"
            / style_profile
            / "advisory_audio_manifest.json"
        )
    return PROJECT_ROOT / "apps" / "web" / "public" / "audio" / "p16_ai" / "advisory_audio_manifest.json"


def _write_review_report(
    *,
    output_manifest: Path,
    manifest: dict[str, Any],
    selected_phrase_ids: list[str],
    languages: list[str],
    dry_run: bool,
) -> None:
    provider = str(manifest["audio_pack"]["provider"])
    key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    report_path = output_manifest.with_name("README_REVIEW.md")
    lines = [
        "# P16 AI Voice Sample Review",
        "",
        "Listen to these files before full generation. Approve the style only if the voice sounds human, calm, clear, and suitable for a driver-assistance warning.",
        "",
        f"- Manifest: `{output_manifest}`",
        f"- Style profile: `{manifest['audio_pack']['style_profile']}`",
        f"- Provider/model: `{manifest['audio_pack']['provider']}` / `{manifest['audio_pack']['model']}`",
        f"- Languages: `{', '.join(languages)}`",
        f"- Dry run: `{'yes' if dry_run else 'no'}`",
        "",
    ]
    if dry_run:
        lines.extend(
            [
                f"This was a dry run. The paths below show where files will be written after running with `{key_name}`; the WAV files do not exist yet.",
                "",
            ]
        )
    lines.append("## Sample Files")
    public_root = PROJECT_ROOT / "apps" / "web" / "public"
    for phrase_id in selected_phrase_ids:
        phrase = manifest["phrases"][phrase_id]
        lines.append(f"### `{phrase_id}`")
        for language in languages:
            asset = phrase["assets"][language]
            path = phrase_audio_path(public_root, str(asset["src"]))
            lines.append(f"- `{language}`: `{path}`")
        lines.append("")
    lines.extend(
        [
            "## Approval Decision",
            "",
            "- [ ] Approved for full pack generation.",
            "- [ ] Reject and try another style/voice.",
            "",
            "Notes:",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _generate_assets(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    phrase_ids: list[str],
    languages: list[str],
    voices: dict[str, str],
    instructions: str,
) -> None:
    key_env_name = "GEMINI_API_KEY" if args.provider == "gemini" else "OPENAI_API_KEY"
    api_key = os.environ.get(key_env_name, "").strip()
    if args.provider == "gemini" and not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        key_env_name = "GEMINI_API_KEY or GOOGLE_API_KEY"
    if not api_key and not args.dry_run:
        raise RuntimeError(
            f"{key_env_name} is not set. Set it for this terminal session before generating audio."
        )

    public_root = PROJECT_ROOT / "apps" / "web" / "public"
    model_candidates = [args.model, *_csv(args.fallback_models)]
    total = len(phrase_ids) * len(languages)
    done = 0
    for phrase_id in phrase_ids:
        phrase = manifest["phrases"][phrase_id]
        for language in languages:
            asset = phrase["assets"][language]
            output_path = phrase_audio_path(public_root, str(asset["src"]))
            done += 1
            if output_path.exists() and not args.force:
                asset["provider"] = args.existing_provider or args.provider
                asset["model"] = args.existing_model or "existing_resume_unknown"
                asset["voice"] = args.existing_voice or voices[language]
                print(f"[{done}/{total}] exists: {output_path}")
                continue
            print(f"[{done}/{total}] generate {language}/{phrase_id} -> {output_path}")
            if args.dry_run:
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            last_error: Exception | None = None
            for model in model_candidates:
                strict_prompt = False
                for attempt in range(1, args.retries + 2):
                    try:
                        if args.provider == "gemini":
                            pcm_bytes = _request_gemini_speech(
                                api_key=api_key,
                                model=model,
                                voice=voices[language],
                                text=str(phrase["text"][language]),
                                language=language,
                                instructions=instructions,
                                timeout_seconds=args.timeout_seconds,
                                strict_prompt=strict_prompt,
                            )
                            _write_pcm_wav(output_path, pcm_bytes)
                        else:
                            audio_bytes = _request_openai_speech(
                                api_key=api_key,
                                model=model,
                                voice=voices[language],
                                text=str(phrase["text"][language]),
                                language=language,
                                instructions=instructions,
                                timeout_seconds=args.timeout_seconds,
                            )
                            output_path.write_bytes(audio_bytes)
                        asset["provider"] = args.provider
                        asset["model"] = model
                        asset["voice"] = voices[language]
                        break
                    except Exception as exc:
                        last_error = exc
                        output_path.unlink(missing_ok=True)
                        if attempt <= args.retries and _is_timeout_error(exc):
                            print(
                                f"  timeout on {model}; retry {attempt}/{args.retries} after {args.retry_sleep_seconds}s"
                            )
                            time.sleep(args.retry_sleep_seconds)
                            continue
                        if attempt <= args.retries and _is_tts_prompt_error(exc):
                            strict_prompt = True
                            print(
                                f"  Gemini requested clearer TTS prompt on {model}; retry {attempt}/{args.retries} with strict transcript prompt"
                            )
                            time.sleep(args.retry_sleep_seconds)
                            continue
                        remaining = model_candidates[model_candidates.index(model) + 1 :]
                        if remaining and _is_quota_error(exc):
                            print(f"  quota hit on {model}; trying fallback model {remaining[0]}")
                            break
                        raise
                if output_path.exists():
                    break
            if last_error is not None and not output_path.exists():
                raise last_error
            _validate_wav(output_path)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)


def _validate_manifest_assets(
    *,
    manifest: dict[str, Any],
    phrase_ids: list[str],
    languages: list[str],
) -> list[str]:
    public_root = PROJECT_ROOT / "apps" / "web" / "public"
    failures: list[str] = []
    for phrase_id in phrase_ids:
        phrase = manifest["phrases"][phrase_id]
        for language in languages:
            path = phrase_audio_path(public_root, str(phrase["assets"][language]["src"]))
            if not path.exists():
                failures.append(f"missing {language}/{phrase_id}: {path}")
                continue
            try:
                _validate_wav(path)
            except (OSError, wave.Error, ValueError) as exc:
                failures.append(f"invalid {language}/{phrase_id}: {exc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate P16 human-like AI advisory audio assets.")
    parser.add_argument("--mode", choices=("sample", "full", "manifest-only", "validate"), default="sample")
    parser.add_argument("--provider", choices=("gemini", "openai"), default="gemini")
    parser.add_argument("--base-manifest", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--public-audio-root", default="")
    parser.add_argument("--fallback-audio-root", default="/audio/p16")
    parser.add_argument("--style-profile", choices=sorted(STYLE_PROFILES), default="calm_adas")
    parser.add_argument("--model", default="")
    parser.add_argument("--fallback-models", default="")
    parser.add_argument("--voice", default="")
    parser.add_argument("--voice-map", default="")
    parser.add_argument("--existing-provider", default="")
    parser.add_argument("--existing-model", default="")
    parser.add_argument("--existing-voice", default="")
    parser.add_argument("--languages", default="en,ms,zh")
    parser.add_argument("--phrase-ids", default="")
    parser.add_argument("--max-phrases", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    args = parser.parse_args()

    default_model, default_voice = _provider_defaults(args.provider)
    if not args.model:
        args.model = default_model
    if not args.voice:
        args.voice = default_voice

    style_name = cast(StyleProfileName, args.style_profile)
    style_profile = STYLE_PROFILES[style_name]
    languages = language_list(args.languages)
    voices = parse_voice_map(args.voice_map, default_voice=args.voice)
    base_manifest = _load_base_manifest(args.base_manifest)
    phrase_ids = _selected_phrase_ids(args, base_manifest)
    selected_set = set(phrase_ids) if args.mode == "sample" else None
    public_audio_root = args.public_audio_root or _default_audio_root(args.mode, args.style_profile)
    output_manifest = args.output_manifest or _default_output_manifest(args.mode, args.style_profile)

    manifest = rewrite_manifest_for_ai_pack(
        base_manifest,
        public_audio_root=public_audio_root,
        fallback_audio_root=args.fallback_audio_root,
        provider=args.provider,
        model=args.model,
        voices=voices,
        style_profile=style_profile,
        status=args.mode,
        selected_phrase_ids=selected_set,
    )
    manifest["audio_pack"]["dry_run"] = bool(args.dry_run)
    manifest["audio_pack"]["fallback_models"] = _csv(args.fallback_models)
    if args.existing_model:
        manifest["audio_pack"]["resume_existing_model"] = args.existing_model

    if args.mode in {"sample", "full"}:
        _generate_assets(
            args=args,
            manifest=manifest,
            phrase_ids=phrase_ids,
            languages=languages,
            voices=voices,
            instructions=style_profile.instructions,
        )

    if args.mode in {"sample", "full", "manifest-only"}:
        public_root = PROJECT_ROOT / "apps" / "web" / "public"
        manifest = update_asset_metadata(manifest, public_root, voices)
        write_manifest(output_manifest, manifest)
        if args.mode == "sample":
            _write_review_report(
                output_manifest=output_manifest,
                manifest=manifest,
                selected_phrase_ids=phrase_ids,
                languages=languages,
                dry_run=args.dry_run,
            )
        print(f"Wrote AI audio manifest: {output_manifest}")

    if args.mode == "validate":
        failures = _validate_manifest_assets(manifest=manifest, phrase_ids=phrase_ids, languages=languages)
        if failures:
            print("Validation failed:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print(f"Validated {len(phrase_ids) * len(languages)} AI audio assets.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
