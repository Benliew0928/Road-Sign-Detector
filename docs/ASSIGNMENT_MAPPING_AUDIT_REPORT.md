# Assignment Mapping Audit Report

Generated: 2026-06-28

## Purpose

This audit checks whether the official assignment images are mapped to sensible
project `semantic_sign_id` classes. It is not a training result; it is a label
evidence check.

## Current Result

- Assignment images checked: 84
- Unique assignment sign IDs checked: 48
- Known wrong mappings fixed so far: `sign_010`, `sign_014`, `sign_032`, `sign_045`, `sign_051`
- Current unresolved / needs-second-review rows: 0

Evidence status summary:

- 25 signs confirmed by clear visual standard shape/text
- 14 signs confirmed by online reference evidence
- 6 signs confirmed by owner review
- 2 signs accepted with medium visual confidence
- 1 sign accepted with reference caution

## Important Correction

`sign_045` was previously mapped to `road_narrows_right`, then briefly to
`lane_ends`. After checking the exact China/GB-style reference and the user's
Google Lens screenshot, it is now mapped to:

- `residential_area_ahead`

Reference:

- China GB-style warning sign 23 reference, residential/village area warning

## Attention Cases

These are not currently wrong, but they have weaker evidence quality than the
obvious signs:

- `sign_008`: `no_straight_or_left`, owner-confirmed
- `sign_010`: corrected to `no_straight_ahead`
- `sign_014`: corrected to `no_overtaking`
- `sign_028`: `motor_vehicles_only`, visual assignment image is clear, but the
  downloaded China reference is a rectangular automobiles-lane sign, so source
  reference remains caution-marked
- `sign_032`: corrected to `roadway_diverges`
- `sign_038`: `double_curve`, accepted visual medium confidence
- `sign_050`: `railway_crossing`, accepted visual medium confidence
- `sign_051`: corrected to `tractors_ahead`
- `sign_057`: `stop_for_checking`, owner-confirmed

## Artifacts

- `data/manifests/assignment_mapping_audit_latest.csv`
- `outputs/audit/assignment_mapping_audit.json`
- `outputs/review/assignment_mapping_audit_unique_signs.jpg`
- `outputs/review/assignment_mapping_audit_low_confidence.jpg`

Note: `data/manifests/assignment_mapping_audit.csv` is currently locked by
another Windows process, so the corrected CSV was written to
`assignment_mapping_audit_latest.csv`.

## Honest Limit

There is no known wrong assignment mapping after this audit. However, the
highest authority would still be a lecturer-provided answer key. Until then,
the audit separates high-confidence mappings from owner-confirmed and
evidence-cautious mappings instead of pretending every label has identical
certainty.
