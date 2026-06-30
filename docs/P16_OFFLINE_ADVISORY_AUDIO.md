# P16 Offline Advisory Audio

P16 provides offline spoken driver advice in English, Bahasa Melayu, and
Mandarin. The audio is intentionally not a robotic reading of the detected sign
name. Each phrase warns the driver and suggests a careful action.

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

Current generated pack:

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
