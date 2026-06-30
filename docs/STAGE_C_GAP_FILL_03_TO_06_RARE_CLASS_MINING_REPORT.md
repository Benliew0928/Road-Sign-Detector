# Stage C Gap Fill 03-06 Rare Class Mining Report

Generated: 2026-06-29

## Summary

This sprint continued non-AI data mining for rare assignment and Malaysian road-sign classes after Stage C Gap Fill 02.

No AI-generated images were used. Two tempting mappings were explicitly rejected after visual probes:

- Mapillary `regulatory--no-straight-through` was rejected for `no_straight_ahead` because it showed yellow no-through-road/T-junction signs, not the red prohibition sign.
- Prashant TT100K `pb`/`pb5` was rejected for `width_restriction` because the crops did not preserve a clear width-limit sign/value.

## Accepted Candidate Data

| Stage | Source | Manifest | Accepted candidates | Notes |
|---|---|---|---:|---|
| Stage 03 | TT100K via `Genius-Society/tt100k` | `data/manifests/stage_c_gap_fill_03_remaining_tt100k_candidates.csv` | 242 | Full train/validation/test mirror scanned after completing all arrow shards. Exact TT100K labels only. |
| Stage 05 | TT100K alternate mirror via `PrashantDixit0/TT-100K` | `data/manifests/stage_c_gap_fill_05_prashant_tt100k_remote_candidates.csv` | 8 | Full 171-file remote metadata scan; only selected remote shards were read for accepted rows. |
| Stage 06 | Roboflow Malaysia Road Sign Dataset v1 | `data/manifests/stage_c_gap_fill_06_roboflow_roadway_diverges_candidates.csv` | 50 | Exact `Roadway diverges` class only; 120px minimum bbox side; one-per-base-image spread sampling. |

## Rejected Or Exhausted Sources

| Source | Result | Evidence |
|---|---|---|
| Mapillary Traffic Sign Dataset via `ThankGod/mapillary_traffic_sign_dataset` | 0 accepted candidates | `outputs/audit/stage_c_gap_fill_04_mapillary_exact_candidates.json`; rejected probes in `outputs/review/`. |
| Prashant TT100K `pb`/`pb5` as width restriction | Rejected | `outputs/review/stage_c_gap_fill_05_rejected_prashant_pb_width_probe.jpg`. |
| Mapillary `warning--pass-left-or-right` as roadway diverges | Rejected | `outputs/review/stage_c_gap_fill_04_mapillary_probe_pass_left_or_right.jpg`. |

## Current Remaining Must-Have Gaps

After Stage 06, 42 of 52 must-have classes meet minimum candidate count. The remaining below-minimum classes are:

| Class | Current | Minimum | Gap |
|---|---:|---:|---:|
| `sound_horn` | 0 | 50 | 50 |
| `steep_descent` | 0 | 50 | 50 |
| `no_left_or_right_turn` | 2 | 50 | 48 |
| `turn_left_or_right` | 3 | 50 | 47 |
| `no_straight_or_left` | 3 | 50 | 47 |
| `tractors_ahead` | 14 | 50 | 36 |
| `residential_area_ahead` | 18 | 50 | 32 |
| `width_restriction` | 28 | 50 | 22 |
| `no_straight_ahead` | 40 | 50 | 10 |
| `stop_for_checking` | 48 | 50 | 2 |

## Review Folders

- Stage 03 QA: `outputs/review/stage_c_gap_fill_03_remaining_tt100k/`
- Stage 04 rejection probes: `outputs/review/stage_c_gap_fill_04_mapillary_probe_pass_left_or_right.jpg`, `outputs/review/stage_c_gap_fill_04_mapillary_rejected_no_straight_through_probe.jpg`
- Stage 05 QA and rejection probe: `outputs/review/stage_c_gap_fill_05_prashant_tt100k_remote/`, `outputs/review/stage_c_gap_fill_05_rejected_prashant_pb_width_probe.jpg`
- Stage 06 QA: `outputs/review/stage_c_gap_fill_06_roboflow_roadway_diverges/`

## Next Recommendation

Do not keep mining broad public datasets blindly for the remaining zero/low classes. The reliable labelled datasets we checked are now exhausted for those classes.

Next best options:

1. Use official/reference sign artwork for the rare China/assignment-only classes, clearly separated from real-road photos.
2. Reduce the minimum requirement for classes that are rare by source availability, and document the evidence.
3. Continue with Stage D QC for all candidate data that now meets count, then freeze Stage E splits.
