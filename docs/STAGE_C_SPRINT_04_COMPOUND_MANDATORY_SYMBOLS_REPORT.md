# Stage C Sprint 04 Compound Mandatory Symbols Report

Generated: 2026-06-28

Policy update on 2026-06-28: this batch is `reference_only`. It is excluded
from final realistic dataset coverage. Use
`outputs/audit/post_stage_c_realistic_gap_report.*` for current Stage C
decisions.

This sprint creates project-owned reference-symbol candidates for compound
mandatory direction signs and pass-either-side signs.

---

## Scope

Targeted classes:

- `straight_or_right`
- `turn_left_or_right`
- `roundabout_mandatory`
- `pass_either_side`

---

## Source

These images are generated locally by:

- `scripts/generate_stage_c_sprint_04_compound_mandatory_symbols.py`

Generated files:

- `data/generated/stage_c_sprint_04_compound_mandatory_symbols/`

Manifest:

- `data/manifests/stage_c_sprint_04_compound_mandatory_symbols.csv`

---

## Historical Generated-Count Result

| Class | Before sprint | Generated candidates | With sprint | Minimum | Result |
|---|---:|---:|---:|---:|---|
| `straight_or_right` | 0 | 50 | 50 | 50 | Meets minimum pending Stage D QC |
| `turn_left_or_right` | 0 | 50 | 50 | 50 | Meets minimum pending Stage D QC |
| `roundabout_mandatory` | 0 | 50 | 50 | 50 | Meets minimum pending Stage D QC |
| `pass_either_side` | 10 | 40 | 50 | 50 | Meets minimum pending Stage D QC |

Historical result after Sprint 01-04:

- Historical generated/reference candidates counted: 360
- Classes with sprint gain: 11
- `must` classes meeting minimum with sprint candidates: 37
- `must` classes still below minimum with sprint candidates: 14

Audit files:

- `outputs/audit/stage_c_sprint_04_compound_mandatory_symbols.json`
- `outputs/audit/post_stage_c_sprint_04_gap_report.csv`
- `outputs/audit/post_stage_c_sprint_04_gap_report.json`

Review sheets:

- `outputs/review/stage_c_sprint_04_compound_mandatory_symbols/straight_or_right.jpg`
- `outputs/review/stage_c_sprint_04_compound_mandatory_symbols/turn_left_or_right.jpg`
- `outputs/review/stage_c_sprint_04_compound_mandatory_symbols/roundabout_mandatory.jpg`
- `outputs/review/stage_c_sprint_04_compound_mandatory_symbols/pass_either_side.jpg`
- `outputs/review/stage_c_sprint_04_compound_mandatory_symbols/_sample_all_candidates.jpg`

---

## QC Notes

- `pass_either_side` was generated after checking the existing P5 contact sheet.
- Roundabout and compound arrow symbols are generated approximations and should
  be supplemented with real/reference official samples when possible.
- These candidates remain pending Stage D visual QC and Stage E split freeze.
