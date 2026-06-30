# Data Progress Tracker

Updated: 2026-07-01

This is the clean entry point for tracking data collection and cleaning.
Open this file first before checking any older CSV or sprint report.

## Current Truth

- Active tracker CSV: `data/manifests/CURRENT_DATA_PROGRESS.csv`
- Latest backup tracker from rare-class gap-fill sprint:
  `data/manifests/CURRENT_DATA_PROGRESS.pending_stage_c_gap_fill_06.csv`
- Current detailed gap report: `outputs/audit/post_stage_c_realistic_gap_report.csv`
- Source/citation ledger: `docs/DATA_SOURCE_CITATION_LEDGER.md`
- Full technical plan: `DATA_TO_FINAL_MODEL_FLOW.md`

## Current Status

- Target classes: 103
- Must-have classes: 52
- Must-have classes with enough realistic candidate count: 42
- Must-have classes still below minimum realistic coverage: 10
- Must-have classes with Stage D minimum QC complete: 42
- Final dataset status: not frozen
- Next project stage: Collect the 10 remaining below-minimum must classes, then
  continue Stage D for those and freeze Stage E splits for the completed classes

Latest sprint note:

- `docs/STAGE_C_GAP_FILL_03_TO_06_RARE_CLASS_MINING_REPORT.md` records the
  latest rare-class mining sprint: 242 accepted Stage 03 TT100K candidates, 8
  accepted Stage 05 alternate-TT100K candidates, and 50 accepted Stage 06
  Roboflow `roadway_diverges` candidates.
- Stage 04 Mapillary and part of Stage 05 were useful as rejection audits:
  Mapillary `no-straight-through` and Prashant TT100K `pb`/`pb5` must not be
  reused as `no_straight_ahead` or `width_restriction`.

Stage D zero-gap minimum-QC batch on 2026-07-01:

- Output manifest: `data/manifests/stage_d_zero_gap_qc_manifest.csv`
- Review sheet: `outputs/review/00_CURRENT_REVIEW/stage_d_zero_gap_qc_contact_sheet.jpg`
- Completed every `must` class whose `gap_to_minimum` is `0`: 42 classes,
  50 accepted crops each, 2,100 accepted crops total.
- `keep_right`, `no_overtaking`, and `side_road_right` include a small number
  of `low_resolution_readable` fallback crops so they can meet the minimum
  honestly while preserving the quality caveat in the manifest.
- Roboflow classification-folder samples are accepted as classifier candidates
  first; detector boxes/masks still need separate Stage E/F handling if those
  sources are used for detector training.

## How To Use The Active CSV

Open `data/manifests/CURRENT_DATA_PROGRESS.csv`.

Important columns:

- `semantic_sign_id`: project class name.
- `priority`: `must`, `should`, or `optional`.
- `realistic_candidate_total`: current usable real/public candidate count.
- `minimum_clean_crops`: minimum target before final model training.
- `gap_to_minimum`: how many more samples are needed.
- `collection_status`: whether count is enough.
- `cleaning_status`: whether the class still needs QC.
- `next_action`: what to do next.

## Rules

- Do not train final models from candidate folders directly.
- Do not recreate or use AI-generated/fake sign images for this dataset.
- Coursework images are external acceptance tests unless the lecturer allows
  training on them.
- A class is only final-ready after Stage D QC and Stage E split freeze.

## Useful Review Folders

- `outputs/review/00_CURRENT_REVIEW/stage_d_zero_gap_qc_contact_sheet.jpg`
- `outputs/review/stage_c_roboflow_import_sheets/`
- `outputs/review/stage_c_mined_roboflow_gap_candidates/`
- `outputs/review/stage_b_weak_class_contact_sheets/`
- `outputs/review/stage_c_gap_fill_01_tt100k/`
- `outputs/review/stage_c_gap_fill_02_public_real_sources/`
- `outputs/review/stage_c_gap_fill_03_remaining_tt100k/`
- `outputs/review/stage_c_gap_fill_05_prashant_tt100k_remote/`
- `outputs/review/stage_c_gap_fill_06_roboflow_roadway_diverges/`
- `outputs/review/stage_c_gap_fill_04_mapillary_rejected_no_straight_through_probe.jpg`
- `outputs/review/stage_c_gap_fill_05_rejected_prashant_pb_width_probe.jpg`

## Cleanup Note

Historical one-off CSVs, sprint outputs, and helper scripts are kept for
traceability but should not be used as the daily tracker. Use the active CSV
above unless we deliberately regenerate a new current tracker.

Archived historical files:

- `_archive/2026-06-29-data-cleanup/manifests/`
- `_archive/2026-06-29-data-cleanup/scripts/`
- `_archive/2026-06-29-data-cleanup/audit_outputs/`

Active `data/manifests/` files are now intentionally limited to:

- `CURRENT_DATA_PROGRESS.csv`
- `stage_d_zero_gap_qc_manifest.csv`
- `target_sign_classes.csv`
- `coursework_manifest.csv`
- `dataset.csv`
- `dataset_sources.json`
- `official_images.csv`
- `official_checksums.csv`
- `stage_c_gap_fill_01_tt100k_candidates.csv`
- `stage_c_gap_fill_02_public_real_sources_candidates.csv`
- `stage_c_gap_fill_03_remaining_tt100k_candidates.csv`
- `stage_c_gap_fill_04_mapillary_exact_candidates.csv`
- `stage_c_gap_fill_05_prashant_tt100k_remote_candidates.csv`
- `stage_c_gap_fill_06_roboflow_roadway_diverges_candidates.csv`
