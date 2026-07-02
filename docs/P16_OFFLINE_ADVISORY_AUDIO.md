# P16 Offline Advisory Audio

P16 provides offline spoken driver advice in English, Bahasa Melayu, and
Mandarin. The audio should not be a robotic reading of the detected sign name.
Each phrase warns the driver and suggests a careful action.

Current status: the app has a working offline Windows SAPI voice pack. This is
the baseline/fallback pack, not the final presentation-grade voice. The next
upgrade is to generate a human-like AI voice pack once, bundle it locally, and
continue playing local audio files during live detection.

Example speed warning:

- EN: This road has a speed limit of 50 kilometers per hour. Please keep your
  speed below the limit and leave enough space ahead.
- MS: Jalan ini mempunyai had laju 50 kilometer sejam. Sila pastikan kelajuan
  tidak melebihi had dan kekalkan jarak selamat.
- ZH: 这条路的限速是每小时50公里。请不要超速，并与前车保持安全距离。

## Assets

Generated source:

- `src/roadsign_assist/audio/advisory.py`
- `scripts/build_p16_audio_manifest.py`
- `scripts/generate_p16_audio_assets.ps1`

Runtime assets:

- `apps/web/public/audio/p16/advisory_audio_manifest.json`
- `apps/web/public/audio/p16/en/*.wav`
- `apps/web/public/audio/p16/ms/*.wav`
- `apps/web/public/audio/p16/zh/*.wav`

Current baseline generated pack:

- 182 advisory phrase IDs.
- 546 WAV files, one file per phrase per language.
- Parameter variants for speed, height, width, and weight restrictions.
- WAV is used because it is browser-compatible and fully offline without adding
  an external encoder dependency.

Voice note:

- EN uses the installed Windows `Microsoft Zira Desktop` voice.
- ZH uses the installed Windows `Microsoft Huihui Desktop` voice.
- MS currently uses `Microsoft Zira Desktop` because this laptop does not have a
  Malay SAPI voice installed. The phrase text and manifest are ready for later
  replacement with a proper Malay voice or human recordings without changing app
  logic.

## AI Voice Upgrade Plan

The efficient final approach is to generate audio online only during asset
creation, then run the presentation fully offline.

1. Keep `src/roadsign_assist/audio/advisory.py` as the source of truth for all
   phrase IDs and text.
2. Add an AI TTS batch generator that reads the current manifest and writes to
   `apps/web/public/audio/p16_ai`.
3. Generate a small sample pack first: around 10-15 phrases across English,
   Bahasa Melayu, Mandarin, speed limits, warning signs, prohibitory signs, and
   `unknown_sign`.
4. Review the samples by listening, choose one voice style, and reject weak
   styles before full generation.
5. Batch-generate the complete pack only after approval.
6. Validate asset count, SHA-256, duration, WAV format, loudness, and missing
   files.
7. Switch the runtime manifest to prefer `/audio/p16_ai/...` while keeping the
   current `/audio/p16/...` Windows pack as fallback.
8. Disconnect internet and run a final EN/MS/ZH listening test.

Suggested style brief:

```text
Speak like a calm but alert Malaysian ADAS driving assistant.
Natural human voice, clear and warm, not robotic.
Use controlled urgency for warning and critical signs.
Do not sound panicked. Keep the phrase concise and easy to understand in a car.
```

Runtime rule: live camera/image/video detection must never call online TTS. The
browser should only play bundled local audio files.

Implemented support:

- `src/roadsign_assist/audio/ai_voice.py` defines reusable AI voice style
  profiles and manifest rewrite helpers.
- `scripts/generate_p16_ai_audio_assets.py` can generate a sample pack, a full
  pack, a manifest-only dry run, and validation checks using Gemini by default.
- The React dashboard now tries `/audio/p16_ai/advisory_audio_manifest.json`
  first, then falls back to `/audio/p16/advisory_audio_manifest.json`.
- AI manifest assets can carry `fallback_src` so missing AI WAV files can fall
  back to the current Windows SAPI WAV files.

Generate a small Gemini sample pack after setting an API key:

```powershell
$env:GEMINI_API_KEY = "your_gemini_api_key_here"
.\.venv\Scripts\python.exe scripts\generate_p16_ai_audio_assets.py --mode sample
```

`sample` mode writes to `apps/web/public/audio/p16_ai_samples/...`. It is only
for checking voice style. After you approve the style, use `full` mode to
continue the real runtime pack in `apps/web/public/audio/p16_ai`.

Generate only English first if you want a cheaper style check:

```powershell
$env:GEMINI_API_KEY = "your_gemini_api_key_here"
.\.venv\Scripts\python.exe scripts\generate_p16_ai_audio_assets.py --mode sample --languages en
```

Try another style:

```powershell
.\.venv\Scripts\python.exe scripts\generate_p16_ai_audio_assets.py --mode sample --style-profile warm_assistant --languages en
.\.venv\Scripts\python.exe scripts\generate_p16_ai_audio_assets.py --mode sample --style-profile crisp_safety --languages en
```

Try different Gemini voices:

```powershell
.\.venv\Scripts\python.exe scripts\generate_p16_ai_audio_assets.py --mode sample --voice Puck --languages en
.\.venv\Scripts\python.exe scripts\generate_p16_ai_audio_assets.py --mode sample --voice-map en=Kore,ms=Puck,zh=Kore
```

After listening approval, generate the full offline AI pack:

```powershell
$env:GEMINI_API_KEY = "your_gemini_api_key_here"
.\.venv\Scripts\python.exe scripts\generate_p16_ai_audio_assets.py --mode full
```

Resume with a cheaper/lower-quota Gemini Flash TTS model if the default model
hits quota:

```powershell
$env:GEMINI_API_KEY = "your_gemini_api_key_here"
.\.venv\Scripts\python.exe scripts\generate_p16_ai_audio_assets.py --mode full --model gemini-2.5-flash-preview-tts --existing-model gemini-3.1-flash-tts-preview --sleep-seconds 2
```

If Gemini is slow or returns `read operation timed out`, resume with a longer
timeout and automatic retries:

```powershell
$env:GEMINI_API_KEY = "your_gemini_api_key_here"
.\.venv\Scripts\python.exe scripts\generate_p16_ai_audio_assets.py --mode full --model gemini-2.5-flash-preview-tts --existing-model gemini-3.1-flash-tts-preview --timeout-seconds 240 --retries 3 --retry-sleep-seconds 15 --sleep-seconds 2
```

If Gemini returns `Model tried to generate text, but it should only be used for
TTS`, keep the same resume command. The generator now retries that file with a
stricter transcript-only TTS prompt.

You can also let the script try a fallback model automatically when a quota
error happens:

```powershell
$env:GEMINI_API_KEY = "your_gemini_api_key_here"
.\.venv\Scripts\python.exe scripts\generate_p16_ai_audio_assets.py --mode full --fallback-models gemini-2.5-flash-preview-tts --existing-model gemini-3.1-flash-tts-preview --sleep-seconds 2
```

Do not add `--force` when resuming. Existing WAV files are skipped.

The full mode writes:

```text
apps/web/public/audio/p16_ai/advisory_audio_manifest.json
apps/web/public/audio/p16_ai/en/*.wav
apps/web/public/audio/p16_ai/ms/*.wav
apps/web/public/audio/p16_ai/zh/*.wav
```

OpenAI is still supported as a fallback provider if it becomes available later:

```powershell
$env:OPENAI_API_KEY = "your_openai_api_key_here"
.\.venv\Scripts\python.exe scripts\generate_p16_ai_audio_assets.py --provider openai --mode sample
```

## Playback Behavior

The React dashboard loads the manifest from:

```text
/audio/p16/advisory_audio_manifest.json
```

Audio plays only when an event has `should_announce = true`.

The playback policy:

- uses the selected dashboard language: English, Bahasa Melayu, or Mandarin;
- respects the mute button;
- applies phrase cooldown so the same warning does not repeat continuously;
- lets warning/critical phrases interrupt lower-priority audio;
- falls back to `unknown_sign` if a phrase is missing;
- never blocks visual inference when audio fails or the browser blocks autoplay.

## Regenerate Audio

From `C:\MiniProject`:

```powershell
.\scripts\generate_p16_audio_assets.ps1 -Force
```

To refresh only hashes/durations when files already exist:

```powershell
.\scripts\generate_p16_audio_assets.ps1
```

The script rebuilds the manifest, generates local WAV files using installed
Windows voices, then writes asset SHA-256, byte size, duration, and voice
metadata.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_advisory_audio.py -q
npm run test --prefix apps\web
npm run build --prefix apps\web
```

Offline manual test:

1. Disconnect internet.
2. Start the local app.
3. Use image/video/camera detection with the warning language set to each
   language.
4. Confirm audio still plays and mute stops warnings.
5. Confirm visual warnings continue even if audio playback is blocked.
