# Stage C Generated Reference Policy

Created: 2026-06-28

## Decision

Generated road-sign images from Stage C Sprints 02-07 are now treated as
`reference_only`.

They must not be counted as final realistic dataset coverage for the Malaysian
RoadSign Assist model.

## Reason

Visual review of Sprint 06 and Sprint 07 showed that the generated signs look
too artificial compared with real road-scene signs. Training the final model on
them as if they were real samples would make the coverage numbers look better
without proving the app can recognize real Malaysian signs.

This is especially risky for:

- Live camera demo reliability
- Assignment sign recognition
- OCR/text signs such as `slow_text`, `stop_for_checking`, and restrictions
- Warning signs where symbol shape and road-context variation matter

## Allowed Uses

Generated signs may still be used for:

- UI smoke tests
- App layout and overlay testing
- ADAS rule prototyping
- Audio warning prototyping
- OCR pipeline experiments where the limitation is documented
- Future augmentation experiments after real validation data exists

## Not Allowed For Final Claims

Generated signs must not be used to claim:

- A class has enough realistic training data
- A class is ready for final model training
- The app can recognize a real Malaysian sign
- Assignment recognition is solved
- Live-camera demo reliability is proven

## Authoritative Coverage Files

Use these files for realistic Stage C decisions:

- `outputs/audit/post_stage_c_realistic_gap_report.csv`
- `outputs/audit/post_stage_c_realistic_gap_report.json`

Current corrected summary:

- Real/public sprint candidates counted: 8
- Generated reference-only candidates excluded: 1,051
- `must` classes meeting minimum realistic count: 28
- `must` classes still below minimum realistic count: 24

## Reliable Replacement Flow

1. Collect real local camera photos/videos for demo-priority signs.
2. Use public licensed photos where local collection is hard.
3. Use official/reference sign material only as supporting reference, not as the
   only proof of real-world performance.
4. Keep assignment images as external test unless training use is explicitly
   approved.
5. Freeze a clean split only after Stage D QC.
6. Retrain the classifier and evaluate on real assignment/common/demo images.

## Affected Generated Batches

| Batch | Status | Notes |
|---|---|---|
| Sprint 02 synthetic top-up | `reference_only` | Seed-derived synthetic variants; not real coverage. |
| Sprint 03 mandatory direction symbols | `reference_only` | Useful icon references, not road-scene data. |
| Sprint 04 compound mandatory symbols | `reference_only` | Useful icon references, not road-scene data. |
| Sprint 05 prohibitory direction symbols | `reference_only` | Useful icon references, not road-scene data. |
| Sprint 06 warning symbols | `reference_only` | Judged too fake/unrealistic for final coverage. |
| Sprint 07 regulatory/text symbols | `reference_only` | Judged too fake/unrealistic for final OCR/classification coverage. |
