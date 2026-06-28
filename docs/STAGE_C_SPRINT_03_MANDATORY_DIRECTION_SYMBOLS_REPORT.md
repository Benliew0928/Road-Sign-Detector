# Stage C Sprint 03 Mandatory Direction Symbols Report

Generated: 2026-06-28

Policy update on 2026-06-28: this batch is `reference_only`. It is excluded
from final realistic dataset coverage. Use
`outputs/audit/post_stage_c_realistic_gap_report.*` for current Stage C
decisions.

This sprint creates project-owned reference-symbol candidates for zero-data
mandatory direction classes. It targets a small group first so visual QC stays
manageable.

---

## Scope

Targeted classes:

- `straight_ahead`
- `turn_left`
- `turn_right`

---

## Source

These images are generated locally by:

- `scripts/generate_stage_c_sprint_03_mandatory_direction_symbols.py`

They are not downloaded from an external dataset. They are simplified
project-owned reference-symbol candidates.

Generated files:

- `data/generated/stage_c_sprint_03_mandatory_direction_symbols/`

Manifest:

- `data/manifests/stage_c_sprint_03_mandatory_direction_symbols.csv`

---

## Historical Generated-Count Result

| Class | Before sprint | Generated candidates | With sprint | Minimum | Result |
|---|---:|---:|---:|---:|---|
| `straight_ahead` | 0 | 50 | 50 | 50 | Meets minimum pending Stage D QC |
| `turn_left` | 0 | 50 | 50 | 50 | Meets minimum pending Stage D QC |
| `turn_right` | 0 | 50 | 50 | 50 | Meets minimum pending Stage D QC |

Historical result after Sprint 01 + Sprint 02 + Sprint 03:

- Historical generated/reference candidates counted: 170
- Classes with sprint gain: 7
- `must` classes meeting minimum with sprint candidates: 33
- `must` classes still below minimum with sprint candidates: 18

Audit files:

- `outputs/audit/stage_c_sprint_03_mandatory_direction_symbols.json`
- `outputs/audit/post_stage_c_sprint_03_gap_report.csv`
- `outputs/audit/post_stage_c_sprint_03_gap_report.json`

Review sheets:

- `outputs/review/stage_c_sprint_03_mandatory_direction_symbols/straight_ahead.jpg`
- `outputs/review/stage_c_sprint_03_mandatory_direction_symbols/turn_left.jpg`
- `outputs/review/stage_c_sprint_03_mandatory_direction_symbols/turn_right.jpg`
- `outputs/review/stage_c_sprint_03_mandatory_direction_symbols/_sample_all_candidates.jpg`

---

## QC Notes

- Arrow orientation was visually checked from the generated sheets.
- These images are clean reference-symbol candidates, not real road-scene
  coverage.
- Final dataset freeze must keep these marked as generated/reference data and
  should supplement them with local camera images where possible.
