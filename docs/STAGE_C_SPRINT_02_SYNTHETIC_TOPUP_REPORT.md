# Stage C Sprint 02 Synthetic Top-Up Report

Generated: 2026-06-28

Policy update on 2026-06-28: this batch is `reference_only`. It is excluded
from final realistic dataset coverage. Use
`outputs/audit/post_stage_c_realistic_gap_report.*` for current Stage C
decisions.

This sprint originally attempted to close the remaining near-minimum
candidate-count gaps after Sprint 01. It creates synthetic candidates from
visually selected clean seeds. These images are useful as reference/support
material, but they do not replace real camera/demo examples and no longer count
toward final realistic coverage.

---

## Scope

Targeted classes:

- `side_road_right`, short by 5 candidates after Sprint 01
- `no_heavy_vehicle`, short by 7 candidates after Sprint 01

---

## Source Seeds

Seed types:

- P5-cleaned EMTD crops selected from class contact sheets
- One Sprint 01 Commons `no_heavy_vehicle` reference

Seed and output details are recorded in:

- `data/manifests/stage_c_sprint_02_synthetic_candidates.csv`

Generated files are stored under:

- `data/generated/stage_c_sprint_02_synthetic_topup/`

---

## Historical Generated-Count Result

| Class | Before sprint | Generated candidates | With sprint | Minimum | Result |
|---|---:|---:|---:|---:|---|
| `side_road_right` | 45 | 5 | 50 | 50 | Meets minimum pending Stage D QC |
| `no_heavy_vehicle` | 43 | 7 | 50 | 50 | Meets minimum pending Stage D QC |

Historical result after Sprint 01 + Sprint 02:

- Historical generated/reference candidates counted: 20
- Classes with sprint gain: 4
- `must` classes meeting minimum with sprint candidates: 30
- `must` classes still below minimum with sprint candidates: 21

Audit files:

- `outputs/audit/stage_c_sprint_02_synthetic_candidates.json`
- `outputs/audit/post_stage_c_sprint_02_gap_report.csv`
- `outputs/audit/post_stage_c_sprint_02_gap_report.json`

Review sheets:

- `outputs/review/stage_c_sprint_02_synthetic_candidates/side_road_right.jpg`
- `outputs/review/stage_c_sprint_02_synthetic_candidates/no_heavy_vehicle.jpg`
- `outputs/review/stage_c_sprint_02_synthetic_candidates/_all_candidates.jpg`

---

## QC Notes

- The generated samples are label-correct by visual inspection, but they are
  still synthetic and must remain pending Stage D review.
- Synthetic candidates should help minimum classifier coverage, but final model
  claims still need real local/demo samples for presentation reliability.
- These generated files should not be mixed into final splits until Stage E
  records them as synthetic and keeps seed-derived images leakage-safe.

---

## Next Step

The generated near-minimum class gaps are closed only as reference material.
Continue Stage C by collecting real/public/official/local data for the remaining
realistic gaps.
because they can be handled as a clean official/reference-symbol batch.
