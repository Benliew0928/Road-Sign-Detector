# Stage C Sprint 05 Prohibitory Direction Symbols Report

Generated: 2026-06-28

Policy update on 2026-06-28: this batch is `reference_only`. It is excluded
from final realistic dataset coverage. Use
`outputs/audit/post_stage_c_realistic_gap_report.*` for current Stage C
decisions.

This sprint creates project-owned generated candidates for zero-data prohibitory
direction signs.

---

## Scope

Targeted classes:

- `no_straight_or_left`
- `no_straight_or_right`
- `no_left_or_right_turn`

---

## Source

These images are generated locally by:

- `scripts/generate_stage_c_sprint_05_prohibitory_direction_symbols.py`

Generated files:

- `data/generated/stage_c_sprint_05_prohibitory_direction_symbols/`

Manifest:

- `data/manifests/stage_c_sprint_05_prohibitory_direction_symbols.csv`

---

## Historical Generated-Count Result

| Class | Before sprint | Generated candidates | With sprint | Minimum | Result |
|---|---:|---:|---:|---:|---|
| `no_straight_or_left` | 0 | 50 | 50 | 50 | Meets minimum pending Stage D QC |
| `no_straight_or_right` | 0 | 50 | 50 | 50 | Meets minimum pending Stage D QC |
| `no_left_or_right_turn` | 0 | 50 | 50 | 50 | Meets minimum pending Stage D QC |

Historical result after Sprint 01-05:

- Historical generated/reference candidates counted: 510
- Classes with sprint gain: 14
- `must` classes meeting minimum with sprint candidates: 40
- `must` classes still below minimum with sprint candidates: 11

Audit files:

- `outputs/audit/stage_c_sprint_05_prohibitory_direction_symbols.json`
- `outputs/audit/post_stage_c_sprint_05_gap_report.csv`
- `outputs/audit/post_stage_c_sprint_05_gap_report.json`

Review sheets:

- `outputs/review/stage_c_sprint_05_prohibitory_direction_symbols/no_straight_or_left.jpg`
- `outputs/review/stage_c_sprint_05_prohibitory_direction_symbols/no_straight_or_right.jpg`
- `outputs/review/stage_c_sprint_05_prohibitory_direction_symbols/no_left_or_right_turn.jpg`
- `outputs/review/stage_c_sprint_05_prohibitory_direction_symbols/_sample_all_candidates.jpg`

---

## QC Notes

- Arrow orientation and prohibition slash were visually checked from contact
  sheets.
- Some slash placement partially covers the internal arrows, which is normal
  for this sign family but should remain flagged for Stage D review.
- These candidates are generated reference data, not real road-scene coverage.
