# Encounter-based guidance implementation

The raw detector boxes and API responses remain frame-based. Shared frontend encounter state now drives live findings, history, video moments and offline audio.

## Behavior

- Confirmation: 1.5-second evidence window, at most one observation per 100 ms, at least three agreeing observations spanning 300 ms, and 80% agreement. Existing confidence and safety checks remain active.
- Track changes bridge only unambiguous same-class overlaps (IoU >= 0.30) within one second. A finding survives short gaps for two seconds, then leaves the active panel.
- Unknown detections use a quiet checking status. Detailed OCR, confidence and tracking evidence are expandable. Confirmed content stays steady through transient uncertainty.
- Voice uses confirmed encounter/action signatures, not frame IDs or backend processing-time cooldown pulses. Every recognized safe class is eligible. Identical signs coalesce; five seconds of absence rearms a new encounter.
- Speech queues at most three candidates, prioritizes severity and confirmation time, and uses a one-second wall-clock gap. Critical guidance can interrupt lower-priority speech. Uncertainty blocks new speech but does not chop an already confirmed sentence; confirmed revisions and expired findings invalidate old speech.
- Unknown/blocked advisories never speak. Numeric recordings require exact values; unavailable values use the recognized sign's generic recording.
- Seeking is silent. Replay guidance explicitly starts a new video announcement pass. Phone and desktop have separate mute controls; desktop speaks only the selected feed.

## Implementation map

- `apps/web/src/encounters.ts`: policy constants, encounter reducer and indexed video timeline.
- `apps/web/src/audio/encounterAudio.ts`: deterministic scheduler, playback ownership, suppression and fallback handling.
- `apps/web/src/hooks/useEncounters.ts` and `useAdvisoryAudio.ts`: live expiry, source lifecycle, shared manifest loading and cached local media.
- Webcam, video, phone sender and desktop monitor use the same encounter presentation. Still images keep one-result announcements.

## Audio audit and repairs

Both packs map all 78 runtime classifier labels. Each has 182 phrases and 546 assets across English, Malay and Chinese. Forty-three AI-pack WAV headers were repaired without changing PCM samples. Durations, checksums and pack status were refreshed, and `apps/web/public/audio.dvc` was updated in the local DVC cache. No recordings were regenerated and no remote assets were pushed.

Run `python scripts/audit_advisory_audio.py` to check both packs. `--repair` performs deterministic local header/metadata repair; `--report PATH` writes JSON. The original repair inventory and PCM hashes are in `outputs/audit/advisory_audio_repair_20260912.json`.

## Verification

- Frontend reducer, resolver, scheduler and video component tests cover repeat suppression, unknowns, dropouts, track changes, revisions, priority, autoplay failure, fallback, pause/seek/replay and source time.
- Python audio integrity, semantic safety and fusion tests pass, including PCM-preserving repair and repair idempotence.
- Desktop/mobile browser scenarios cover the panel, video counts, selected-feed audio, controls, layout and existing dashboard workflows. All 43 repaired files decode in Chromium; English/Malay/Chinese sample recordings reach `ended`.
- Build, ESLint and focused Ruff checks pass. DVC reports the audio pack up to date.
- Visual checks are saved in `outputs/encounter-ux/`.

Camera/video verification uses deterministic fixtures. The screenshot supplied for planning is not a source video, so that exact road sequence was not replayed. Existing pronunciation and wording were retained; browser decoding/playback verification is not a linguistic review. No model retraining, detector smoothing or runtime TTS requests were introduced.
