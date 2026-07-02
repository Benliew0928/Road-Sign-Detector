# Data Progress Tracker

Updated: 2026-07-02

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
- Must-have classes with enough realistic candidate count: 52
- Must-have classes still below minimum realistic coverage: 0
- Must-have classes with Stage D minimum QC complete: 52
- Non-must zero-gap classes with Stage D minimum QC complete: 26
- Total zero-gap classes with Stage D minimum QC complete: 78
- Target classes with enough candidate count before Stage D: 78
- Final dataset status: current Stage E classifier dataset frozen
- Next project stage: Retrain the semantic classifier from
  `data/processed/stage_e_classifier_current`. All `must` classes now meet
  their minimum, have Stage D minimum QC complete, and appear in the frozen
  train/validation/test classifier splits. The remaining below-minimum classes
  are non-must only and are ignored for this current training dataset if they
  have no Stage D samples.

Latest sprint note:

- `docs/STAGE_C_GAP_FILL_03_TO_06_RARE_CLASS_MINING_REPORT.md` records the
  latest rare-class mining sprint: 242 accepted Stage 03 TT100K candidates, 8
  accepted Stage 05 alternate-TT100K candidates, and 50 accepted Stage 06
  Roboflow `roadway_diverges` candidates.
- Stage 04 Mapillary and part of Stage 05 were useful as rejection audits:
  Mapillary `no-straight-through` and Prashant TT100K `pb`/`pb5` must not be
  reused as `no_straight_ahead` or `width_restriction`.
- Stage C manual visual search 01 filled 10 must classes to 50/50 candidate
  coverage: `no_straight_ahead`, `width_restriction`, `stop_for_checking`,
  `residential_area_ahead`, `tractors_ahead`, `turn_left_or_right`,
  `no_left_or_right_turn`, `no_straight_or_left`, `sound_horn`, and
  `steep_descent`. These are
  tracked in `data/manifests/stage_c_manual_01_candidates.csv`, stored under
  `data/raw/manual_collection/stage_c_manual_01/`, and completed Stage D in
  `data/manifests/stage_d_manual_pending_qc_manifest.csv` plus
  `data/manifests/stage_d_steep_descent_qc_manifest.csv`. The rarest active
  classes include controlled visual variants from exact sign references or
  real-photo bases; those rows are flagged in `source_modality`.

Stage D zero-gap minimum-QC batches on 2026-07-01 and 2026-07-02:

- Must output manifest: `data/manifests/stage_d_zero_gap_qc_manifest.csv`
- Must review sheet: `outputs/review/00_CURRENT_REVIEW/stage_d_zero_gap_qc_contact_sheet.jpg`
- Other zero-gap output manifest: `data/manifests/stage_d_other_zero_gap_qc_manifest.csv`
- Other zero-gap review sheet:
  `outputs/review/00_CURRENT_REVIEW/stage_d_other_zero_gap_qc_contact_sheet.jpg`
- Manual pending output manifest:
  `data/manifests/stage_d_manual_pending_qc_manifest.csv`
- Manual pending review sheet:
  `outputs/review/00_CURRENT_REVIEW/stage_d_manual_pending_qc_contact_sheet.jpg`
- Steep descent output manifest:
  `data/manifests/stage_d_steep_descent_qc_manifest.csv`
- Steep descent review sheet:
  `outputs/review/00_CURRENT_REVIEW/stage_d_steep_descent_qc_contact_sheet.jpg`
- Completed every class whose `gap_to_minimum` was `0` at the start of the
  2026-07-01 Stage D batch: 68 classes total.
- Completed the 9 later manual zero-gap classes on 2026-07-02, then completed
  the reopened `steep_descent` must class on 2026-07-02.
- Completed classes by priority: 42 `must`, 16 `should`, 10 `optional`.
- Completed classes by priority after the manual pending batch: 51 `must`, 16
  `should`, 10 `optional`.
- Completed classes by priority after `steep_descent`: 52 `must`, 16
  `should`, 10 `optional`.
- Accepted crops: 3,180 total. Minimums are 50 for `must`, 30 for `should`,
  and 10 for `optional`.
- `keep_right`, `no_overtaking`, and `side_road_right` include a small number
  of `low_resolution_readable` fallback crops so they can meet the minimum
  honestly while preserving the quality caveat in the manifest.
- Roboflow classification-folder samples are accepted as classifier candidates
  first; detector boxes/masks still need separate Stage E/F handling if those
  sources are used for detector training.

Stage E current classifier freeze on 2026-07-02:

- Freeze script: `scripts/freeze_stage_e_classifier_dataset.py`
- Frozen all-sample manifest: `data/manifests/final_dataset.csv`
- Frozen train manifest: `data/manifests/final_train.csv`
- Frozen validation manifest: `data/manifests/final_validation.csv`
- Frozen test manifest: `data/manifests/final_test.csv`
- Assignment external-test manifest:
  `data/manifests/assignment_external_test.csv`
- Split audit: `outputs/audit/final_split_audit.json`
- Trainable classifier folder dataset:
  `data/processed/stage_e_classifier_current/`
- Frozen unique crops: 3,178 across 78 labels.
- Split counts: 2,187 train, 497 validation, 494 test.
- Every included label appears in train, validation, and test; all 52 `must`
  labels appear in train and validation.
- Coursework assignment images included in training: 0. The 84 coursework
  images are external acceptance tests only.
- Two duplicate Stage D crop rows were skipped by identical label+crop SHA-256.
- Exact Stage D crop-level dedupe groups do not cross splits. Eight controlled
  visual-variant base groups cross splits in this current training freeze so
  every included class can be trained and evaluated; this is documented in the
  audit and should not be oversold as a final no-near-duplicate academic split.

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
- Do not use AI model-generated images as final training data. Controlled
  variants from exact references must stay flagged and be accepted or rejected
  during Stage D before training.
- Coursework images are external acceptance tests unless the lecturer allows
  training on them.
- A class is only final-ready after Stage D QC and Stage E split freeze.

## Useful Review Folders

- `outputs/review/00_CURRENT_REVIEW/stage_d_zero_gap_qc_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_d_other_zero_gap_qc_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_d_manual_pending_qc_contact_sheet.jpg`
- `outputs/review/stage_c_roboflow_import_sheets/`
- `outputs/review/stage_c_mined_roboflow_gap_candidates/`
- `outputs/review/stage_b_weak_class_contact_sheets/`
- `outputs/review/stage_c_gap_fill_01_tt100k/`
- `outputs/review/stage_c_gap_fill_02_public_real_sources/`
- `outputs/review/stage_c_gap_fill_03_remaining_tt100k/`
- `outputs/review/stage_c_gap_fill_05_prashant_tt100k_remote/`
- `outputs/review/stage_c_gap_fill_06_roboflow_roadway_diverges/`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_no_straight_ahead_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_width_restriction_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_stop_for_checking_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_residential_area_ahead_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_tractors_ahead_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_turn_left_or_right_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_no_left_or_right_turn_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_no_straight_or_left_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_sound_horn_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_steep_descent_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_d_steep_descent_qc_contact_sheet.jpg`
- `outputs/audit/final_split_audit.json`
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
- `stage_d_other_zero_gap_qc_manifest.csv`
- `stage_d_manual_pending_qc_manifest.csv`
- `stage_d_steep_descent_qc_manifest.csv`
- `final_dataset.csv`
- `final_train.csv`
- `final_validation.csv`
- `final_test.csv`
- `assignment_external_test.csv`
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
- `stage_c_manual_01_candidates.csv`
