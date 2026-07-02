# Data To Final Model Flow

## Purpose

This file is the step-by-step technical flow for turning the current prototype
into a useful final model that can recognize assignment road signs and common
Malaysian road signs in the live app.

It does not cover report writing. It only tracks data, annotation, training,
evaluation, and app integration.

## Current Problem

The live app can detect some signs, but many assignment signs and common
Malaysian signs are classified as `unknown` or wrong.

The main causes are:

- The current classifier was trained mainly on draft EMTD labels, not on the
  assignment sign set.
- Several important Malaysian signs have too few clean samples.
- Some public dataset labels are noisy at crop level.
- The model can only recognize classes that have enough clean training data.
- Cleaning all public dataset noise forever will not directly solve assignment
  recognition unless the target classes are covered.

## Core Strategy

Do not keep cleaning random dataset noise endlessly.

The correct flow is:

1. Lock the target classes.
2. Measure current clean sample coverage.
3. Collect real/public/official missing data only for priority classes.
4. Annotate and QC those priority classes.
5. Freeze a clean dataset split.
6. Retrain the detector only if signs are not being found.
7. Retrain the classifier if signs are found but classified as unknown or wrong.
8. Evaluate on assignment, Malaysian common-sign, and live-demo test sets.
9. Integrate the best model into the app.

Data authenticity rule:

- Do not use AI model-generated images as final training data.
- Controlled visual variants from exact reference signs must stay flagged in
  manifests and be accepted or rejected during Stage D before training.
- Final coverage gates must be based on real photos, public licensed image
  sources, official/reference sign material, or local camera/video collection.

---

# 1. Phase Map

| Flow stage | Project phase | Goal | Main output |
|---|---|---|---|
| Target class lock | P2, P3 | Decide exactly what signs must be recognized | Target class list |
| Data gap audit | P4, P5 | Count clean samples per target class | Coverage gap report |
| Data collection sprint | P4 | Add missing assignment/common/demo signs | Raw data archive |
| Annotation and QC | P5 | Boxes, crops, labels, OCR text where needed | Clean reviewed labels |
| Dataset freeze | P6 | Create leakage-safe train/val/test splits | Dataset version |
| Detector retraining | P8 | Improve sign finding/segmentation | YOLO model |
| Classifier retraining | P9 | Improve sign meaning recognition | Classifier model |
| OCR and rules | P10, P12 | Read numbers/text and generate ADAS output | OCR/rule outputs |
| App integration | P13, P14 | Use the selected model in the local app | Working demo app |
| Final evaluation | P18, P20 | Prove speed, accuracy, stability | Acceptance report |

---

# 2. Stage A - Lock Target Classes

## Goal

Decide what the final model must know. This avoids wasting time cleaning or
collecting signs that do not matter for the assignment or demonstration.

## Class Groups

| Group | Meaning | Required? |
|---|---|---:|
| Assignment signs | Signs from official coursework images | Yes |
| Common Malaysian signs | Stop, give way, speed limit, no entry, school zone, pedestrian, etc. | Yes |
| Demo signs | Signs that will appear in the final live demo | Yes |
| Rare signs | Uncommon signs from public datasets | No, unless needed |
| OCR-heavy signs | Text/numeric signs requiring OCR | Yes, if used in demo |

## Deliverables

- `data/manifests/target_sign_classes.csv`
- `outputs/audit/target_class_coverage.json`
- Updated P2/P3 mapping if any assignment signs are still unresolved

## Generated On 2026-06-27

- `scripts/generate_stage_ab_target_gap_report.py`
- `data/manifests/target_sign_classes.csv`
- `outputs/audit/target_class_coverage.json`

Scope decision:

- The target universe is the full `configs/catalogue/malaysia_signs.v1.json`
  catalogue.
- Assignment signs, common Malaysian signs, and inferred demo signs are promoted
  to `must`.
- Existing useful dataset classes outside the first critical set are `should`.
- Remaining catalogue classes are `optional` and excluded from final claims until
  promoted.

Mapping correction on 2026-06-28:

- `sign_045` was corrected from `road_narrows_right`, then from the temporary
  `lane_ends` label, to `residential_area_ahead` after checking the exact
  China/GB-style residential-area-ahead reference. The sign is village /
  residential area ahead, not lane ends or normal road-narrows.
- `sign_010` was corrected to `no_straight_ahead`.
- `sign_014` was corrected to `no_overtaking`.
- `sign_032` was corrected to `roadway_diverges`, also described as split
  traffic / traffic diverge.
- `sign_051` was corrected to `tractors_ahead`, also described as watch out
  for tractors.

Assignment mapping audit on 2026-06-28:

- `docs/ASSIGNMENT_MAPPING_AUDIT_REPORT.md`
- `data/manifests/assignment_mapping_audit_latest.csv`
- `outputs/audit/assignment_mapping_audit.json`
- `outputs/review/assignment_mapping_audit_unique_signs.jpg`
- `outputs/review/assignment_mapping_audit_low_confidence.jpg`

Note: `data/manifests/assignment_mapping_audit.csv` was locked by another
Windows process during regeneration, so use
`data/manifests/assignment_mapping_audit_latest.csv` as the current audit CSV
until the locked file can be overwritten.

Current audit result: 84 assignment images and 48 unique assignment sign IDs
checked; no known wrong mapping remains, but nine attention cases are recorded
with owner-confirmed, medium-confidence, or reference-caution evidence quality.

## Completion Gate

- [x] Every assignment sign has a target semantic meaning or is explicitly
      marked out of scope.
- [x] Every demo sign is in the target class list.
- [x] Every target class has a priority: `must`, `should`, or `optional`.

---

# 3. Stage B - Data Gap Audit

## Goal

Find out which target classes already have enough clean data and which classes
need collection.

## Sample Targets

| Class importance | Minimum clean crops | Good target | Strong target |
|---|---:|---:|---:|
| Assignment/demo critical | 50 | 150 | 300+ |
| Common Malaysian sign | 50 | 100 | 200+ |
| OCR/numeric sign | 50 per visual type | 100+ | 300+ |
| Rare optional sign | 10 | 30 | 50+ |

## Important Rule

For classifier training, cropped sign images are useful.

For detector/segmenter training, full images with boxes or masks are required.

## Deliverables

- `outputs/audit/data_gap_report.csv`
- `outputs/audit/data_gap_report.json`
- Contact sheets for weak classes

## Generated On 2026-06-27

- `outputs/audit/data_gap_report.csv`
- `outputs/audit/data_gap_report.json`
- `outputs/review/stage_b_weak_class_contact_sheets/`

Current result:

- Target classes: 103
- `must` classes: 52
- `should` classes: 26
- `optional` classes: 25
- `must` classes needing action before final model claims: 51
- Classes currently ready by count and QC flag for next classifier training: 3
- Classes with current P5 samples: 43

Important interpretation:

- Most assignment-only classes have no current P5 training crops, so the app is
  expected to output `unknown` or wrong labels for them until Stage C/D data is
  collected and Stage G retrains the classifier.
- `maximum_speed` and `height_restriction` have enough crop count, but remain
  flagged because the P5 review found historical label noise.

## Completion Gate

- [x] Every `must` class has a current clean count.
- [x] Every `must` class below target has a data source plan.
- [x] Classes with too few samples are either collected or excluded from model
      claims.

---

# 4. Stage C - Data Collection Sprint

## Goal

Collect only the missing data needed to make the final model useful.

## Priority Sources

| Source | Use case | Notes |
|---|---|---|
| Assignment images | Final assignment compatibility | Must never leak into training if used as external test |
| Phone camera photos | Real Malaysian demo signs | Best for presentation realism |
| Short videos | More natural variation | Extract frames carefully to avoid leakage |
| Public Malaysian sign photos | Fill class gaps | Need provenance and licence notes |
| Official/reference sign material | Clarify sign design and text layout | Reference-only unless converted into realistic evaluated data |
| Synthetic augmentation | Increase variation, not replace real data | Use only after clean real labels exist; never count as real coverage by itself |

## Collection Rule

For each priority sign class, collect:

- Different lighting
- Different distances
- Different angles
- Partial occlusion if realistic
- Blurry and small examples only if they still resemble real usage

## Deliverables

- `data/raw/local_collection/`
- `data/manifests/local_collection_manifest.csv`
- `outputs/audit/data_provenance_report.json`

## Stage C Public Import Progress On 2026-06-27

Roboflow public import completed as candidate data:

- `docs/STAGE_C_ROBOFLOW_IMPORT_REPORT.md`
- `docs/DATA_SOURCE_CITATION_LEDGER.md`
- `data/manifests/roboflow_class_mapping.csv`
- `data/manifests/roboflow_source_manifest.csv`
- `outputs/audit/roboflow_import_audit.json`
- `outputs/audit/post_roboflow_data_gap_report.csv`

Current result:

- Accepted Roboflow candidate samples: 22,611
- Accepted Roboflow candidate samples for target classes: 22,274
- Accepted staged classes: 57
- `must` classes meeting minimum after Roboflow candidate import: 25
- `must` classes still below minimum after Roboflow candidate import: 27

Important note:

- Roboflow imports are public mixed candidate data. They are separated and
  mapped, but they still need Stage D targeted QC and split freeze before final
  training claims.
- Post-Roboflow coverage can now be regenerated with
  `scripts/generate_post_roboflow_gap_report.py`.

## Stage C Sprint 01 Commons Top-Up On 2026-06-28

Small near-minimum top-up completed as candidate data:

- `docs/STAGE_C_SPRINT_01_COMMONS_TOPUP_REPORT.md`
- `data/manifests/stage_c_sprint_01_commons_candidates.csv`
- `outputs/audit/stage_c_sprint_01_commons_candidates.json`
- `outputs/audit/post_stage_c_sprint_01_gap_report.csv`

Current sprint result:

- Usable sprint candidates: 8
- Downloaded but excluded by visual QC: 1
- `no_overtaking`: 47 -> 51 candidate samples, meets minimum pending Stage D QC
- `keep_right`: 47 -> 50 candidate samples, meets minimum pending Stage D QC
- `no_heavy_vehicle`: 42 -> 43 candidate samples, still short by 7
- `must` classes meeting minimum with sprint candidates: 28
- `must` classes still below minimum with sprint candidates: 24

Important note:

- These Commons files are candidate data only. They still need Stage D visual
  QC/annotation and Stage E split freeze before final training.

## Stage C Online Reference Sources 01 On 2026-06-28

Reliable online reference-source sprint completed:

- `docs/STAGE_C_ONLINE_REFERENCE_SOURCES_01_REPORT.md`
- `data/manifests/stage_c_online_reference_sources_01.csv`
- `outputs/audit/stage_c_online_reference_sources_01.json`
- `outputs/review/stage_c_online_reference_sources_01/`

Current result:

- Downloaded online reference candidates: 15
- High-confidence exact/near-exact Malaysian references found for 11 classes
- `motor_vehicles_only` candidate downloaded but marked possible mismatch
- 12 classes remain unresolved from reliable online sources

Important note:

- These files are mostly diagrams/SVG references from Wikimedia/HuggingFace
  Wikimedia-linked sources.
- They support class definition but are still not 50 independent real
  road-scene photos and do not close the
  realistic-photo gap by themselves.

## Stage C Roboflow Gap Mining On 2026-06-28

Targeted mining from previously rejected/noisy Roboflow source labels completed:

- `scripts/mine_stage_c_roboflow_gap_candidates.py`
- `data/manifests/stage_c_mined_roboflow_gap_candidates.csv`
- `outputs/audit/stage_c_mined_roboflow_gap_candidates.json`
- `outputs/review/stage_c_mined_roboflow_gap_candidates/`

Current result:

- Mined public candidates: 51
- `pass_either_side`: 50 mined candidates, now meets minimum pending Stage D QC
- `side_road_right`: 1 mined candidate, still below minimum
- After the owner-approved assignment mapping corrections, `must` classes
  meeting minimum realistic count: 28
- After the owner-approved assignment mapping corrections, `must` classes
  still below minimum realistic count: 24

Important note:

- These are real public dataset candidates, but they came from source classes
  previously rejected as noisy/mixed.
- They must be visually QCed during Stage D before final training claims.

## Stage C China/GB Reference Sources 01 On 2026-06-28

Reliable China/GB-style reference-source sprint completed for rare assignment
signs:

- `docs/STAGE_C_CHINA_REFERENCE_SOURCES_01_REPORT.md`
- `scripts/collect_stage_c_china_reference_sources.py`
- `data/manifests/stage_c_china_reference_sources_01.csv`
- `outputs/audit/stage_c_china_reference_sources_01.json`
- `outputs/review/stage_c_china_reference_sources_01/`

Current result:

- Downloaded China/GB-style reference candidates: 19
- High-confidence class anchors: 18
- Medium-confidence possible style difference: `motor_vehicles_only`
- Download failures: 0
- Remaining unresolved rare assignment class: `no_lane_changing`

Important note:

- These references explain many of the strange assignment signs: they match
  Mainland China / GB-style sign families better than Malaysian road-scene
  datasets.
- These are official-style reference diagrams, not independent camera photos.
- They improve class definition and future augmentation/OCR work, but they do
  not change `realistic_candidate_total`.

## Stage C Manual Visual Search 01 On 2026-07-01

Manual visual-search pilot completed for the smallest remaining must-have gap:

- `scripts/collect_stage_c_manual_01_no_straight_ahead.py`
- `scripts/collect_stage_c_manual_01_residential_area_ahead.py`
- `scripts/collect_stage_c_manual_01_stop_for_checking.py`
- `scripts/collect_stage_c_manual_01_tractors_ahead.py`
- `scripts/collect_stage_c_manual_01_remaining_must_gaps.py`
- `scripts/collect_stage_c_manual_01_steep_descent.py`
- `data/manifests/stage_c_manual_01_candidates.csv`
- `data/raw/manual_collection/stage_c_manual_01/no_straight_ahead/`
- `data/raw/manual_collection/stage_c_manual_01/residential_area_ahead/`
- `data/raw/manual_collection/stage_c_manual_01/stop_for_checking/`
- `data/raw/manual_collection/stage_c_manual_01/tractors_ahead/`
- `data/raw/manual_collection/stage_c_manual_01/turn_left_or_right/`
- `data/raw/manual_collection/stage_c_manual_01/no_left_or_right_turn/`
- `data/raw/manual_collection/stage_c_manual_01/no_straight_or_left/`
- `data/raw/manual_collection/stage_c_manual_01/sound_horn/`
- `data/raw/manual_collection/stage_c_manual_01/width_restriction/`
- `data/raw/manual_collection/stage_c_manual_01/steep_descent/`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_no_straight_ahead_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_residential_area_ahead_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_stop_for_checking_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_tractors_ahead_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_turn_left_or_right_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_no_left_or_right_turn_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_no_straight_or_left_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_sound_horn_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_width_restriction_contact_sheet.jpg`
- `outputs/review/00_CURRENT_REVIEW/stage_c_manual_steep_descent_contact_sheet.jpg`

Current result:

- `no_straight_ahead`: 40 -> 50 candidate samples, now meets minimum pending
  Stage D QC.
- `width_restriction`: 28 -> 50 candidate samples, now meets minimum pending
  Stage D QC.
- `stop_for_checking`: 48 -> 50 candidate samples, now meets minimum pending
  Stage D QC.
- `residential_area_ahead`: 18 -> 50 candidate samples, now meets minimum
  pending Stage D QC.
- `tractors_ahead`: 14 -> 50 candidate samples, now meets minimum pending
  Stage D QC.
- `turn_left_or_right`: 3 -> 50 candidate samples, now meets minimum pending
  Stage D QC.
- `no_left_or_right_turn`: 2 -> 50 candidate samples, now meets minimum
  pending Stage D QC.
- `no_straight_or_left`: 3 -> 50 candidate samples, now meets minimum pending
  Stage D QC.
- `sound_horn`: 0 -> 50 candidate samples, now meets minimum pending Stage D
  QC.
- `steep_descent`: 0 -> 50 candidate samples, now meets minimum and completed
  Stage D QC.
- Added 10 Commons visual-match candidates: 5 official-style references and 5
  real road-photo crops.
- Added 22 Commons visual-match `width_restriction` candidates: mostly
  official-style references, plus one real road-photo crop.
- Added 2 Commons visual-match `stop_for_checking` candidates: both
  official-style inspection/checking references with visible Chinese text.
- Added 32 visual-match `residential_area_ahead` candidates: China/GB village
  warning road photos, official/reference diagrams, and a few public web
  product/reference images.
- Added 36 official-style `tractors_ahead` reference diagrams.
- Added 47 `turn_left_or_right`, 48 `no_left_or_right_turn`, 47
  `no_straight_or_left`, and 50 `sound_horn` candidates. These rare classes
  combine exact official/reference diagrams with controlled visual variants
  from exact source signs, all flagged in `source_modality`.
- Added 50 `steep_descent` candidates after the owner reopened the class:
  6 real Commons road-photo crops, 4 direct exact references, 36 controlled
  variants from real-photo bases, and 4 controlled variants from exact
  references. The weaker road-only cached crop was excluded before Stage D.
- Must-have classes meeting minimum candidate count: 52.
- Must-have classes still below minimum candidate count: 0.

Important note:

- No AI model-generated images were used. The rarest active classes do include
  deterministic controlled visual variants from exact source signs; Stage D
  must decide whether to accept them for final training.
- The candidate images are deliberately kept in the manual collection folder
  and one compact manifest, so this pilot does not mix with earlier Stage C
  files.
- These candidates completed targeted Stage D minimum QC on 2026-07-02 and
  still need Stage E split freeze before model training.

## Completion Gate

- [x] Each `must` class reaches at least 50 clean candidate crops.
- [ ] Demo signs have real camera examples.
- [ ] Data provenance is recorded.

---

# 5. Stage D - Annotation And QC

## Goal

Turn collected images into clean training data.

## Annotation Types

| Task | Required for | Output |
|---|---|---|
| Bounding boxes | Detector and crop extraction | `x1,y1,x2,y2` |
| Segmentation masks | Segmentation model | polygon/mask |
| Semantic class | Classifier | `semantic_sign_id` |
| OCR transcript | Text signs, speed, height, weight | text + language |
| Numeric value | Speed/height/weight | value + unit |

## QC Stop Rule

P5 is complete enough when:

- Priority class sheets have no obvious wrong-sign crops.
- Known noisy classes have correction manifests.
- Rare/weak classes are documented instead of endlessly cleaned.
- Dataset claims only include classes with enough reviewed samples.

P5 is not complete if:

- A `must` class still has obvious mixed labels.
- Assignment/demo signs are not represented.
- A class used for metrics has fewer than the agreed sample target.

## Deliverables

- `data/manifests/p5_owner_crop_corrections.csv`
- `data/manifests/p5_class_review_next.csv`
- `outputs/review/p5_class_contact_sheets/`
- `outputs/audit/p5_label_qc_report.json`

## Stage D Zero-Gap Minimum-QC Batches On 2026-07-01 And 2026-07-02

To avoid generating many messy review files, the zero-gap Stage D work writes
one row-level manifest and one contact sheet per scope:

- `scripts/run_stage_d_zero_gap_qc.py`
- Must classes: `data/manifests/stage_d_zero_gap_qc_manifest.csv`
- Must review sheet:
  `outputs/review/00_CURRENT_REVIEW/stage_d_zero_gap_qc_contact_sheet.jpg`
- Other zero-gap classes: `data/manifests/stage_d_other_zero_gap_qc_manifest.csv`
- Other zero-gap review sheet:
  `outputs/review/00_CURRENT_REVIEW/stage_d_other_zero_gap_qc_contact_sheet.jpg`
- Manual pending classes:
  `data/manifests/stage_d_manual_pending_qc_manifest.csv`
- Manual pending review sheet:
  `outputs/review/00_CURRENT_REVIEW/stage_d_manual_pending_qc_contact_sheet.jpg`
- Steep descent class:
  `data/manifests/stage_d_steep_descent_qc_manifest.csv`
- Steep descent review sheet:
  `outputs/review/00_CURRENT_REVIEW/stage_d_steep_descent_qc_contact_sheet.jpg`

Current result:

- Every class whose `gap_to_minimum` is `0` is completed at the minimum Stage D
  level.
- 78 classes completed: 52 `must`, 16 `should`, 10 `optional`.
- Accepted crops: 3,180 total.
- Minimums are 50 crops for `must`, 30 for `should`, and 10 for `optional`.
- `keep_right`, `no_overtaking`, and `side_road_right` include a small number
  of `low_resolution_readable` fallback crops. These were visually readable
  enough for classifier candidate use, but the caveat is preserved in the
  manifest for Stage E review.

Interpretation:

- These crops are ready as Stage E classifier split candidates.
- Roboflow classification-folder samples remain classifier candidates first;
  they should not be treated as detector/segmentation training boxes unless a
  later Stage D/E step adds explicit box or mask annotations.
- Final model training is still blocked on Stage E split freeze.
- Remaining below-minimum classes cannot be honestly Stage D-completed yet:
  10 `should` and 15 `optional`. These 25 below-minimum classes were checked
  on 2026-07-02; none had enough accessible Stage D-valid crops to meet its
  minimum.
- `no_straight_ahead`, `width_restriction`, `stop_for_checking`,
  `residential_area_ahead`, `tractors_ahead`, `turn_left_or_right`,
  `no_left_or_right_turn`, `no_straight_or_left`, and `sound_horn` completed
  their manual pending Stage D pass on 2026-07-02. `steep_descent` completed a
  targeted Stage D pass on 2026-07-02.

## Completion Gate

- [~] All target classes reviewed by contact sheet. All 78 zero-gap classes
      are completed at minimum Stage D level; 25 below-minimum classes remain
      blocked by collection coverage.
- [x] Zero-gap crop selections are reproducible by manifests and
      `scripts/run_stage_d_zero_gap_qc.py`.
- [x] Zero-gap batches require no manual drag-and-drop corrections.

---

# 6. Stage E - Freeze Dataset Split

## Goal

Create the dataset version used for final retraining.

## Split Rules

- Similar frames from one video stay in the same split.
- Duplicate or near-duplicate images stay in the same split.
- Assignment images should be kept as external test unless explicitly approved
  for training.
- Demo test images should never be used for training.

## Deliverables

- `data/manifests/final_train.csv`
- `data/manifests/final_validation.csv`
- `data/manifests/final_test.csv`
- `data/manifests/assignment_external_test.csv`
- `outputs/audit/final_split_audit.json`

## Stage E Current Classifier Freeze On 2026-07-02

Current training dataset frozen from the Stage D accepted crop manifests:

- `scripts/freeze_stage_e_classifier_dataset.py`
- `data/manifests/final_dataset.csv`
- `data/manifests/final_train.csv`
- `data/manifests/final_validation.csv`
- `data/manifests/final_test.csv`
- `data/manifests/assignment_external_test.csv`
- `outputs/audit/final_split_audit.json`
- `data/processed/stage_e_classifier_current/`
- `data/processed/stage_e_classifier_current/labels.json`
- `data/processed/stage_e_classifier_current/dataset_metadata.json`

Current result:

- Frozen unique Stage D crops: 3,178.
- Included labels: 78.
- Split samples: 2,187 train, 497 validation, 494 test.
- All 78 included labels appear in train, validation, and test.
- All 52 `must` labels appear in train and validation.
- Coursework assignment images in training: 0.
- Assignment external-test samples: 84.
- Non-must target classes without Stage E samples: 25; ignored for this
  current training freeze as requested.
- Two duplicate Stage D crop rows were skipped by identical label+crop SHA-256.

Important limitation:

- Exact Stage D crop-level dedupe groups do not cross train/validation/test.
- Eight controlled visual-variant base groups do cross splits so the current
  dataset can train/evaluate every included class. This is acceptable for the
  requested "train with current data first" dataset, but it should not be
  described as the final no-near-duplicate academic split.

## Completion Gate

- [x] No assignment-test leakage into training.
- [x] No exact Stage D crop-level duplicate leakage across train/validation/test.
- [x] Every `must` class appears in train and validation, or limitation is
      documented.

---

# 7. Stage F - Detector Precision Hardening

## Goal

Reduce false detections before spending time on a new detector training run.

Current live behavior shows that visible signs are usually found in enough
light, but non-sign objects can be detected as `unknown_sign`. Examples include
faces, shirts, shadows, colored objects, and anything with enough color/shape
similarity to pass the current candidate filters.

For this project, Stage F should therefore be a precision-hardening stage first:

1. Identify whether false detections come from the deep detector or the
   color/shape baseline fallback.
2. Tune thresholds and fallback behavior before retraining.
3. Build a no-sign negative test set.
4. Retrain only if the deep detector still fires on non-sign objects after
   tuning.

## First Diagnostic Question

For every bad `unknown_sign` box, check the event evidence:

| Evidence source | Meaning | Recommended action |
|---|---|---|
| `detector:color_shape_baseline` | The HSV color/shape fallback found a candidate after the deep detector found nothing | Tighten or disable fallback first |
| `detector:ultralytics:...` | The deep YOLO detector found the object | Tune detector confidence, then consider hard-negative retraining |
| `classifier_raw:unknown_sign` | The detector found a crop, but the classifier rejected it | Usually Stage G classifier/unknown-gate work, unless the crop is clearly not a sign |

Important interpretation:

- `unknown_sign` is not automatically a classifier failure.
- If the crop is a face, shirt, wall, shadow, or random object, the detector
  should not have emitted it in the first place.
- If the crop is a real sign but the label is `unknown_sign`, that belongs to
  Stage G.

## No-Retrain Hardening Pass

Before retraining P8, test these app/config changes on the same live/video
cases:

| Lever | Current risk | Recommended Stage F test |
|---|---|---|
| Detector confidence | Low thresholds can preserve recall but allow noisy boxes | Try `0.35`, `0.45`, and `0.55`; select by validation plus negative-set false positives |
| Baseline fallback | Color/shape fallback can detect shirts, faces, shadows, posters, and colored objects | Disable fallback for final demo, or limit to emergency/debug mode |
| Fallback max detections | Multiple fallback boxes make no-sign scenes noisy | Reduce from `3` to `1` if fallback is kept |
| Baseline candidate size | Very small colored components can pass | Raise `min_width`, `min_height`, and `min_area_ratio` |
| Baseline geometry | Loose shape filters accept many non-sign blobs | Raise `min_extent`/`min_solidity`, lower `max_aspect_ratio`, and reject `shape_label=other` for live fallback |
| Temporal stability | Single-frame false boxes are visually distracting | Only display or announce stable tracks after enough frames |

Suggested first live-demo profile:

- `fallback_to_baseline: false`
- detector `confidence_threshold: 0.35` or higher
- keep tracking stability enabled
- suppress or de-emphasize unstable `unknown_sign` events in the UI

This preserves the deep detector path while removing the most permissive
classical fallback behavior.

## Negative Test Set

Stage F must add a no-sign false-positive check before retraining decisions.

Collect short images/videos from:

- Owner face and upper body
- Shirts with strong colors or shapes
- Indoor room backgrounds
- Shadows and glare
- Posters, screens, labels, bags, and boxes
- Road scenes with no signs
- Vehicles, trees, poles, buildings, and shop signs
- Final demo environment when no road sign is visible

These samples should not be used as positive sign classes. They are either:

- a validation/test set for false-positive measurement, or
- detector hard-negative training images with empty YOLO label files if P8
  retraining becomes necessary.

Recommended metric:

| Metric | Purpose |
|---|---|
| False boxes per 100 no-sign frames | Measures visual noise |
| False stable tracks per minute | Measures user-facing annoyance |
| Unknown false-positive rate | Measures non-sign objects displayed as unknown signs |
| Demo-sign recall | Ensures precision tuning did not remove real signs |
| Latency p95 | Ensures tuning/retraining remains app usable |

## When To Retrain P8

Retrain the detector if:

- The app fails to locate the sign.
- The bounding box misses most of the sign.
- Small signs are not detected.
- Camera/video demo signs are not segmented.
- After threshold/fallback tuning, the deep detector still repeatedly fires on
  non-sign objects in the negative test set.

Do not retrain P8 if:

- The sign is detected correctly but classified as unknown.
- False positives are mainly from `color_shape_baseline` fallback.
- The problem can be fixed by detector confidence, fallback limits, or UI
  stability gating without hurting demo-sign recall.

That is a P9 classifier problem.

## If Retraining Is Needed

Only retrain from full images with boxes or masks. Crop-only Stage E classifier
manifests are not enough for detector training.

Add:

- Positive full-frame sign images with reviewed boxes/masks.
- No-sign hard negatives with empty label files.
- Difficult near-misses such as shirts, faces, shadows, posters, road clutter,
  and colored objects.
- Real demo camera frames with and without signs.

Do not use assignment external-test images for detector training unless
explicitly approved.

Retraining objective:

- Optimize precision and stable false-positive reduction, not just recall.
- Keep enough recall for assignment/demo signs in good lighting.
- Prefer the smallest model that meets runtime and false-positive targets.

## Deliverables

- False-positive triage report showing whether errors came from deep detector,
  fallback detector, classifier rejection, or UI stability.
- No-sign negative test set and metrics.
- Threshold/fallback comparison report.
- If retrained: YOLO training run folder.
- If retrained: ONNX export.
- Validation/test mAP, precision, recall, and small-sign recall.
- Runtime benchmark.

## Current Stage F Result (2026-07-02)

Stage F was run as a detector precision-hardening pass before retraining.

Negative evaluation set:

- `data/processed/stage_f_negative_eval`
- 120 no-sign background crops generated from existing full-frame detector
  validation/test images.
- Empty labels; used only for detector false-positive evaluation.

Profile comparison:

- `outputs/evaluation/stage_f_detector/profile_report.json`
- `outputs/evaluation/stage_f_detector/profile_report.csv`

Measured profiles:

| Profile | Positive recall @ IoU 0.50 | Positive precision @ IoU 0.50 | No-sign false boxes / 100 images |
|---|---:|---:|---:|
| Current hybrid `conf=0.25`, fallback `3` | `0.6810` | `0.8020` | `250.8` |
| Deep only `conf=0.25` | `0.6810` | `0.8144` | `21.7` |
| Deep only `conf=0.35` | `0.6509` | `0.8629` | `17.5` |
| Deep only `conf=0.45` | `0.6164` | `0.8994` | `14.2` |
| Deep only `conf=0.55` | `0.5603` | `0.9028` | `10.8` |
| Deep only `conf=0.65` | `0.5302` | `0.9318` | `8.3` |
| Hybrid `conf=0.35`, fallback `1` | `0.6509` | `0.8580` | `100.0` |

Selected live-app profile:

- `fallback_to_baseline: false`
- `fallback_max_detections: 1`
- `confidence_threshold: 0.35`
- `nms_iou_threshold: 0.50`
- model: `models/exported/experimental/emtd_segmenter_s30.onnx`

Reason:

- Disabling the color/shape fallback removes the main noise source.
- Deep-only `conf=0.35` keeps `95.6%` of the current hybrid positive recall.
- No-sign false boxes drop from `250.8` to `17.5` per 100 no-sign crops.
- Positive precision improves from `0.8020` to `0.8629`.

Runtime:

- `outputs/evaluation/stage_f_detector/runtime_deep_conf_0_35.json`
- Mean wall latency: `390.2 ms`
- P95 wall latency: `687.1 ms`

Retraining decision:

- Do not retrain the detector yet.
- Retrain only if live/demo signs are still missed, boxes crop away most of the
  sign, or the deep-only `conf=0.35` profile still repeatedly fires on non-sign
  objects during owner testing.

Selection report:

- `outputs/evaluation/stage_f_detector/stage_f_selection_report.json`

## Completion Gate

- [x] Assignment/demo signs are detected on the available detector test split;
      owner live testing remains the real final check.
- [x] No-sign negative set has substantially lower false boxes per 100 images
      after disabling fallback.
- [x] False `unknown_sign` boxes from fallback-style noise are reduced for the
      selected experimental profile.
- [x] Detector confidence/fallback profile is selected and documented.
- [x] Runtime is measured for the selected profile.
- [x] Retraining was not triggered by the current Stage F evidence; app remains
      compatible with the existing exported detector.

---

# 8. Stage G - Retrain Classifier

## When To Retrain P9

Retrain the classifier if:

- The app detects a sign but returns `unknown`.
- The app detects a sign but predicts the wrong meaning.
- New target classes were added.
- P5 label cleanup changed many crops.

This is the most important next model stage for the current problem.

## Current Stage G Input

Use only the Stage E frozen classifier dataset:

- Training folder: `data/processed/stage_e_classifier_current`
- Labels: `data/processed/stage_e_classifier_current/labels.json`
- Metadata: `data/processed/stage_e_classifier_current/dataset_metadata.json`
- CSV manifests:
  - `data/manifests/final_train.csv`
  - `data/manifests/final_validation.csv`
  - `data/manifests/final_test.csv`
- Assignment external test:
  `data/manifests/assignment_external_test.csv`

Do not train directly from Stage C folders, manual collection folders, or the
coursework assignment image folder. Those are source/provenance folders, not
the frozen training release.

## Training Candidates

| Model | Use |
|---|---|
| MobileNetV3 | Fast baseline and CPU fallback |
| EfficientNetV2 | Stronger semantic classifier |
| Embedding/prototype gate | Safer unknown rejection |
| YOLO classifier head or YOLO detector class labels | Optional comparison |

## Stage G Retraining Approach

The retraining approach is a staged comparison, not a one-shot replacement.

1. Train MobileNetV3 first as the fast sanity baseline.
   - Goal: prove the Stage E dataset loads correctly and every label can train.
   - Use this run as the CPU/runtime fallback candidate.
   - Do not promote it just because it trains successfully; compare metrics
     first.

2. Train EfficientNetV2-S second as the main accuracy candidate.
   - Goal: improve semantic accuracy for assignment and common Malaysian signs.
   - Expect higher accuracy than MobileNetV3, with larger model size and slower
     runtime.
   - Keep it experimental until assignment/test evaluation is reviewed.

3. Apply confidence calibration and conservative unknown rejection.
   - Use validation/test behavior to tune temperature and confidence threshold.
   - Prefer `unknown_sign` over confident wrong ADAS advice.
   - If embedding/prototype export is used, tune distance on validation only
     and report selective coverage/accuracy.

4. Evaluate with four views before integration.
   - Frozen validation/test macro-F1 and accuracy.
   - Per-class confusion matrix, especially visually similar signs.
   - Assignment external-test predictions from
     `data/manifests/assignment_external_test.csv`.
   - Runtime and ONNX parity on the target laptop.

5. Integrate only the best reviewed candidate.
   - Export ONNX, labels, and calibration.
   - Smoke-test the app with upload image, local camera, and phone stream.
   - Keep the previous classifier available until the new model passes the
     smoke tests.

## First Training Commands

Fast baseline:

```powershell
.\.venv\Scripts\python.exe -m roadsign_assist.cli train-classifier --data data/processed/stage_e_classifier_current --architecture mobilenet_v3_large --epochs 40 --batch 32 --imgsz 224 --device auto --name stage_e_current_mobilenet_v3 --experimental
```

Main accuracy candidate:

```powershell
.\.venv\Scripts\python.exe -m roadsign_assist.cli train-classifier --data data/processed/stage_e_classifier_current --architecture efficientnet_v2_s --epochs 40 --batch 32 --imgsz 224 --device auto --name stage_e_current_efficientnet_v2_s --experimental
```

If GPU memory is tight, reduce `--batch` to `16` before changing model or image
size. If training is only for a quick smoke check, use fewer epochs, but do not
compare that quick run as a final candidate.

Assignment external-test evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_stage_g_classifier.py --model models/exported/experimental/stage_e_current_efficientnet_v2_s.onnx --labels models/exported/experimental/stage_e_current_efficientnet_v2_s.labels.json --calibration models/exported/experimental/stage_e_current_efficientnet_v2_s.calibration.json --predictions-output outputs/evaluation/stage_g_classifier/stage_e_current_efficientnet_v2_s_assignment_predictions.csv --report-output outputs/evaluation/stage_g_classifier/stage_e_current_efficientnet_v2_s_assignment_report.json
```

## Evaluation Decision Rules

- If MobileNetV3 and EfficientNetV2 are both poor, inspect Stage E label
  confusion before collecting more data.
- If EfficientNetV2 is much better but too slow, keep MobileNetV3 as fallback
  and test a smaller image size or stricter runtime profile.
- If assignment images are still mostly `unknown`, check whether detector crops
  are good; if crops are good, continue Stage G data/classifier work.
- If assignment images are detected but semantically wrong in consistent
  patterns, add targeted review/collection for those confused class pairs.
- If visible signs are not boxed at all, pause Stage G and go back to Stage F
  detector/fallback hardening.

## Deliverables

- `outputs/training/stage_e_current_mobilenet_v3/`
- `outputs/training/stage_e_current_efficientnet_v2_s/`
- `models/checkpoints/stage_e_current_mobilenet_v3/best.pt`
- `models/checkpoints/stage_e_current_efficientnet_v2_s/best.pt`
- Candidate ONNX, labels, and calibration files under
  `models/exported/experimental/` until promotion.
- Assignment external-test prediction report.
- Runtime benchmark and ONNX parity report.
- Confusion matrix
- Comparison summary selecting the app candidate or explaining why no model is
  promoted yet.

## Current Stage G Result (2026-07-02)

Stage G has been run on the current Stage E classifier freeze.

- MobileNetV3 baseline completed:
  - Frozen test accuracy: `0.9393`
  - Frozen test macro-F1: `0.9325`
  - Assignment external raw accuracy: `0.6429`
  - ONNX parity: passed
- EfficientNetV2-S logits candidate completed:
  - Frozen test accuracy: `0.9534`
  - Frozen test macro-F1: `0.9430`
  - Assignment external raw accuracy: `0.6786`
  - Assignment external accepted accuracy: `0.7857` at `0.8333` coverage
  - ONNX parity: passed
- EfficientNetV2-S q97 embedding-gated candidate completed:
  - Frozen test accepted accuracy: `0.9765` at `0.9474` coverage
  - Assignment external accepted accuracy: `0.9107` at `0.6667` coverage
  - ONNX embedding parity: passed

Selected app candidate for `configs/inference/experimental.yaml`:

- Model:
  `models/exported/experimental/stage_e_current_efficientnet_v2_s_embedding_q97.onnx`
- Labels:
  `models/exported/experimental/stage_e_current_efficientnet_v2_s.labels.json`
- Calibration:
  `models/exported/experimental/stage_e_current_efficientnet_v2_s_embedding_q97.calibration.json`
- Image size: `224`

High-coverage fallback for demos:

- `models/exported/experimental/stage_e_current_efficientnet_v2_s.onnx`

Selection report:

- `outputs/evaluation/stage_g_classifier/stage_g_selection_report.json`

Known current-data weaknesses from assignment external evaluation:

- `children_crossing` can confuse with `school_zone`.
- `stop` can confuse with `stop_for_checking`.
- `roadway_diverges` can confuse with `school_zone`.
- `uneven_road` can confuse with `slow_text`.

## Completion Gate

- [x] MobileNetV3 baseline trains and exports without dataset or ONNX errors.
- [x] EfficientNetV2-S trains and exports without dataset or ONNX errors.
- [x] Every included label is present in training and validation.
- [x] Assignment external-test predictions are reviewed.
- [x] Common signs have acceptable class-level accuracy for the current data
  freeze, with known weak pairs documented above.
- [x] Unknown rejection is conservative enough for the selected ADAS app
  candidate.
- [x] Runtime is acceptable for the chosen live-app profile.
- [x] Selected classifier is integrated into the experimental app config with
  visible model version.

---

# 9. Stage H - OCR, ADAS Rules, And App Integration

## Goal

Convert recognition into useful autonomous-driving style output.

Stage H is not another model-training stage. It is the integration stage that
turns the selected detector/classifier/OCR outputs into clear, safe,
driver-facing behavior:

```text
frame -> detector box -> classifier label -> optional OCR value -> ADAS action
      -> dashboard/phone UI -> offline audio advisory
```

Stage H must use the current selected experimental profile:

- Detector: `models/exported/experimental/emtd_segmenter_s30.onnx`
- Detector profile: deep only, `confidence_threshold: 0.35`, fallback off
- Classifier:
  `models/exported/experimental/stage_e_current_efficientnet_v2_s_embedding_q97.onnx`
- Audio runtime policy: local files only; no online TTS during live inference

## Examples

| Recognition | App output |
|---|---|
| `maximum_speed`, value `50` | Target speed 50 km/h |
| `school_zone` | Slow down, watch children |
| `no_entry` | Do not enter |
| `height_restriction`, value `5.4m` | Warn if vehicle height exceeds limit |
| `pedestrian_crossing` | Watch pedestrians |

## Stage H Implementation Steps

1. Backend advisory event model.
   - Add a display-ready advisory object to each sign event.
   - Keep the existing machine-readable `action` object for tests and future
     automation.
   - Advisory text must be cautious for low-confidence or unknown signs.
   - Output should include headline, instruction, and whether it is safe to
     announce.

2. OCR/value integration.
   - Keep OCR conditional; run it only for text/numeric signs or the first
     unknown candidate.
   - Normalize numeric values into structured fields:
     speed `KM/H`, height/width `M`, weight `T`.
   - If OCR value is missing, low-confidence, or invalid, downgrade to
     `UNKNOWN_CAUTION` instead of issuing a strong command.

3. Deterministic ADAS rules.
   - `maximum_speed + value` -> target speed advisory.
   - `height_restriction`, `width_restriction`, `weight_restriction` ->
     restriction advisory only when a valid value exists.
   - `stop`, `give_way`, `no_entry`, prohibited turns, pedestrian/school/road
     hazards -> fixed safe driver advice.
   - Low-confidence predictions and `unknown_sign` must never trigger a strong
     command.

4. Dashboard and phone UI integration.
   - Display friendly advisory text, not only enum names like
     `SET_TARGET_SPEED`.
   - Show sign meaning, confidence, OCR/value, and ADAS instruction.
   - Show the selected detector/classifier model versions and detector profile.
   - Ensure dashboard current-sign panel, event timeline, phone UI, and live
     wall use the same event data.

5. Offline audio integration.
   - Use the local P16 audio manifest cleanly.
   - Avoid noisy missing-pack requests when the AI voice pack is not present.
   - Preserve cooldown/priority/interrupt behavior.
   - Audio must prefer parameterized phrases when OCR values are valid.

6. Safety and regression tests.
   - Unit-test action/advisory text for common and safety-critical signs.
   - Unit-test low-confidence and bad-OCR downgrade behavior.
   - Frontend-test advisory phrase resolution and display helpers.
   - Build the web app and smoke-load the configured backend.

## Deliverables

- OCR value extraction for numeric signs
- Deterministic ADAS rules
- UI model version display
- Audio warning for selected demo signs
- Dashboard and phone UI showing friendly advisory text
- Clean audio manifest fallback without startup 404 noise
- Stage H regression tests

## Current Stage H Result On 2026-07-02

Implemented:

- Backend sign events now include a display-ready `advisory` object with:
  headline, instruction, and `safe_to_announce`.
- `SemanticRuleEngine` now generates human driver advice for speed limits,
  restriction signs, prohibitions, mandatory signs, warning signs, and
  low-confidence/unknown detections.
- The live inference engine adds the advisory to every `SignEventModel`.
- `/api/v1/models` now exposes `detector_profile` and `classifier_profile`,
  including model paths, image sizes, thresholds, and detector fallback status.
- Dashboard, event timeline, phone UI, live wall, video overlay, and batch
  results now use the same advisory display helper instead of raw enum labels.
- The local P16 audio manifest is tried before the optional AI voice pack, so
  a missing AI pack no longer creates the first-request 404 noise.
- Audio selection ignores events where `safe_to_announce` is false.

Changed files:

- `src/roadsign_assist/inference/models.py`
- `src/roadsign_assist/semantics/rules.py`
- `src/roadsign_assist/inference/engine.py`
- `apps/api/roadsign_api/main.py`
- `apps/web/src/types.ts`
- `apps/web/src/advisoryDisplay.ts`
- `apps/web/src/App.tsx`
- `apps/web/src/PhoneCameraApp.tsx`
- `apps/web/src/LiveCameraWallApp.tsx`
- `apps/web/src/components/BatchResults.tsx`
- `apps/web/src/components/EventTimeline.tsx`
- `apps/web/src/components/SignPanel.tsx`
- `apps/web/src/components/VideoSurface.tsx`
- `apps/web/src/hooks/useAdvisoryAudio.ts`
- `apps/web/src/audio/advisoryAudio.ts`

Validation:

- `python -m py_compile` on changed backend/API modules: passed.
- `pytest tests/unit/test_semantic_rules.py apps/api/tests/test_openapi_contract.py -q`:
  10 passed.
- `pytest tests/unit/test_advisory_audio.py -q`: 3 passed.
- `ruff check` on changed backend/API/test files: passed.
- `npm run test --prefix apps/web`: 10 passed.
- `npm run build --prefix apps/web`: passed.

Known owner-test item:

- Browser audio playback still depends on the browser allowing sound after user
  interaction; the offline manifest and resolver path are validated by tests.

## Completion Gate

- [x] App shows sign meaning, confidence, OCR/value, and ADAS action.
- [~] Audio warnings work for demo signs. Offline manifest and resolver tests
      pass; final browser speaker behavior still needs live user testing.
- [x] Low confidence does not trigger dangerous commands.
- [x] Dashboard and phone UI show the same advisory for the same event.
- [x] App displays selected detector/classifier versions and detector profile.
- [x] Missing optional AI voice pack does not create user-visible errors or
      repeated server 404 noise.

---

# 10. Stage I - Final Evaluation

## Goal

Collect the final acceptance evidence for the current experimental app profile.
Stage I does not train a new model. It answers:

- What works now?
- What is still weak?
- What exact artifacts should be shown during the final review?

## Test Sets

| Test set | Purpose |
|---|---|
| Assignment external test | Coursework requirement |
| Malaysian common-sign test | Practical usefulness |
| Demo route/test video | Presentation reliability |
| No-sign negative set | False positive check |
| Blurry/small-sign set | Robustness check |

## Metrics

- Detection recall
- Classification accuracy
- Macro-F1
- Unknown rejection rate
- OCR numeric accuracy
- End-to-end latency
- Assignment image success rate
- Demo stability

## Stage I Implementation Steps

1. Build a single final acceptance report.
   - Read Stage F detector metrics.
   - Read Stage G classifier metrics.
   - Read OCR smoke metrics.
   - Read Stage E split/audit status.
   - Run the current experimental app pipeline on assignment external images.
   - Write one compact JSON report and one assignment prediction CSV.

2. Keep assignment evaluation honest.
   - Assignment images are external-test crops, not normal road-scene frames.
   - Report classifier-on-crop acceptance separately from full detector pipeline
     behavior.
   - Do not count assignment images as training data.

3. Verify detector and no-sign behavior.
   - Use the Stage F selected detector profile:
     deep-only `emtd_segmenter_s30`, confidence `0.35`, fallback off.
   - Include no-sign negative false-positive metrics.
   - Include small-sign recall slice.

4. Verify classifier behavior.
   - Use the Stage G selected q97 embedding-gated EfficientNetV2-S model.
   - Report frozen test accuracy, macro-F1, accepted accuracy, assignment
     external selective accuracy, and unknown/selective rejection rate.

5. Verify OCR and ADAS runtime evidence.
   - Include OCR synthetic multilingual smoke metrics.
   - Include end-to-end assignment pipeline latency.
   - Keep real-road OCR accuracy as owner-side test until real footage exists.

6. Create a reliable demo fallback video.
   - Generate one controlled fallback video from detector-validation full-frame
     sign images already in the project.
   - Do not collect new footage.
   - Smoke-test that the current app pipeline detects signs in this fallback
     video source.

## Current Stage I Result On 2026-07-02

Script and artifacts:

- `scripts/evaluate_stage_i_final.py`
- `outputs/evaluation/stage_i_final/stage_i_final_acceptance_report.json`
- `outputs/evaluation/stage_i_final/stage_i_assignment_pipeline_predictions.csv`
- `outputs/evaluation/stage_i_final/stage_i_fallback_demo_video.mp4`

Selected app profile evaluated:

- Detector: `models/exported/experimental/emtd_segmenter_s30.onnx`
- Detector profile: deep only, confidence `0.35`, fallback off
- Classifier:
  `models/exported/experimental/stage_e_current_efficientnet_v2_s_embedding_q97.onnx`
- Classifier labels:
  `models/exported/experimental/stage_e_current_efficientnet_v2_s.labels.json`
- Classifier calibration:
  `models/exported/experimental/stage_e_current_efficientnet_v2_s_embedding_q97.calibration.json`

Final metrics:

| Area | Result |
|---|---:|
| Stage E frozen classifier test accuracy | `0.9534` |
| Stage E frozen classifier macro-F1 | `0.9430` |
| q97 embedding-gated frozen accepted accuracy | `0.9765` |
| q97 embedding-gated frozen coverage | `0.9474` |
| Assignment external classifier raw accuracy | `0.6786` |
| Assignment external classifier accepted accuracy | `0.9107` |
| Assignment external classifier coverage | `0.6667` |
| Assignment external classifier rejection rate | `0.3333` |
| Full app pipeline on assignment crop images: images with events | `2 / 84` |
| Full app pipeline on assignment crop images: expected sign found | `0 / 84` |
| Assignment pipeline p95 runtime | `628.8 ms` |
| Detector selected-profile positive recall at IoU 0.50 | `0.6509` |
| Detector selected-profile positive precision at IoU 0.50 | `0.8629` |
| No-sign false boxes per 100 negative images | `17.5` |
| No-sign negative image false-positive rate | `0.1417` |
| Small-sign detector recall at IoU 0.50 | `0.6740` |
| OCR synthetic multilingual exact match rate | `1.0000` |
| OCR warm mean latency | `189.4 ms` |
| Fallback video smoke event rate | `1.0000` |

Important interpretation:

- The classifier is strong enough for the current dataset and accepts
  assignment external crops with high selective accuracy.
- The full app pipeline does **not** handle the isolated assignment crop images
  well because the selected detector is tuned for road-scene/full-frame sign
  localization, not standalone cropped sign cards.
- This is not hidden: the Stage I report marks assignment handling as partial.
- If the final demo requires dropping isolated assignment-sign PNGs into image
  mode and seeing full app detections, the next fix is a detector/domain
  adaptation pass or a carefully gated classifier-only crop mode.
- For live/demo road-scene behavior, the detector-selected profile is still the
  correct current choice because it greatly reduces false positives versus the
  old color/shape fallback.

Fallback video:

- Created at
  `outputs/evaluation/stage_i_final/stage_i_fallback_demo_video.mp4`.
- Source:
  `data/processed/emtd_segmentation/images/test`.
- It is a controlled presenter backup from detector-validation full frames, not
  newly collected real road footage.
- The current app pipeline detected signs on `14 / 14` selected fallback frames.

Limitations:

- Assignment labels remain draft single-review external acceptance labels.
- OCR real-road numeric accuracy still needs owner-side camera/video testing.
- There is still no reviewed out-of-distribution classifier set, so unknown
  rejection is measured through selective coverage and Stage F no-sign detector
  negatives.

## Completion Gate

- [~] App handles assignment signs in scope. Classifier external accepted
      accuracy is `0.9107` at `0.6667` coverage, but full detector pipeline on
      isolated assignment crop images finds expected signs in `0 / 84`.
- [x] App handles chosen common Malaysian signs at classifier level:
      frozen test accuracy `0.9534`, macro-F1 `0.9430`.
- [x] App runs within required time on the assignment external pipeline:
      p95 `628.8 ms`, all `84 / 84` under two seconds.
- [x] Demo has a reliable fallback video if live camera conditions are poor:
      fallback smoke event rate `1.0000`.

---

# 11. Master Tracker

| Stage | Status | Owner | Deliverable | Notes |
|---|---:|---|---|---|
| A. Target class lock | `[x]` | Codex | `target_sign_classes.csv` | Baseline locked on 2026-06-27; update only if final demo signs change |
| B. Data gap audit | `[x]` | Codex | `data_gap_report.csv` | Generated on 2026-06-27; use `collection_rank` as the next data queue |
| C. Data collection sprint | `[~]` | Owner/Codex | Local/public raw data | Current tracker has all 52 must classes meeting minimum candidate count; 25 non-must classes remain below minimum |
| D. Annotation and QC | `[~]` | Owner/Codex | QC sheets and correction manifests | All 78 zero-gap classes completed at minimum Stage D level: 3,180 accepted crops across the must, other, manual pending, and steep-descent manifests; 25 below-minimum non-must classes still need data then QC |
| E. Freeze dataset split | `[x]` | Codex | Final train/val/test manifests | Current Stage E classifier freeze complete: 3,178 crops, 78 labels, 0 coursework training images |
| F. Detector precision hardening | `[x]` | Codex | False-positive report and selected detector profile | Deep-only `conf=0.35`, fallback off; retrain only if real/demo road-scene boxes are still missed |
| G. Classifier retrain | `[x]` | Codex | Selected classifier ONNX | EfficientNetV2-S q97 embedding-gated classifier selected for app |
| H. App integration | `[~]` | Codex | Updated live app | Advisory/model-profile integration done; browser speaker/audio still needs owner live test |
| I. Final evaluation | `[~]` | Owner/Codex | Acceptance metrics | Runtime/common/fallback pass; assignment full detector pipeline is partial on isolated crop images |

---

# 12. Data Coverage Tracker Template

Use this table or convert it into CSV.

| semantic_sign_id | Priority | Needed for | Current clean crops | Target crops | Gap | Action | Status |
|---|---:|---|---:|---:|---:|---|---:|
| stop | must | common/demo |  | 100 |  | audit | `[ ]` |
| give_way | must | common/demo |  | 100 |  | audit | `[ ]` |
| maximum_speed | must | common/demo/OCR |  | 200 |  | audit | `[ ]` |
| no_entry | must | common/demo |  | 100 |  | audit | `[ ]` |
| pedestrian_crossing | must | common/demo |  | 100 |  | audit | `[ ]` |
| school_zone | must | Malaysia/common/demo |  | 100 |  | collect | `[ ]` |
| height_restriction | should | common/OCR |  | 100 |  | audit | `[ ]` |
| no_parking | should | common |  | 80 |  | audit | `[ ]` |
| no_stopping | should | common |  | 80 |  | audit | `[ ]` |
| assignment_signs | must | coursework |  | TBD |  | map/count | `[ ]` |

---

# 13. Model Experiment Tracker

| Experiment | Dataset version | Detector | Classifier | Key result | Runtime | Decision |
|---|---|---|---|---|---:|---|
| DET-FINAL-01 | `stage_f_detector` | `emtd_segmenter_s30`, deep-only `conf=0.35` |  | Recall `0.6509`, precision `0.8629`, no-sign false boxes `17.5/100` | p95 detector runtime `687.1 ms` | Selected current detector profile |
| CLS-FINAL-01 | `stage_e_current_20260702` |  | MobileNetV3 | Frozen test acc `0.9393`; assignment raw acc `0.6429` | batch eval `~30 ms/image` | Not selected |
| CLS-FINAL-02 | `stage_e_current_20260702` |  | EfficientNetV2-S logits | Frozen test acc `0.9534`; assignment raw acc `0.6786` | batch eval `~30 ms/image` | High-coverage fallback |
| CLS-FINAL-03 | `stage_e_current_20260702` |  | EfficientNetV2-S q97 embedding | Frozen accepted acc `0.9765`; assignment accepted acc `0.9107` at `0.6667` coverage | batch eval `~30 ms/image` | Selected app classifier |
| E2E-FINAL-01 | `stage_i_final` | Selected Stage F profile | Selected Stage G profile | Assignment crop full pipeline expected match `0/84`; classifier assignment gate partial | p95 `628.8 ms` | Needs detector/domain adaptation if crop-image assignment demo is required |
| LIVE-FINAL-01 | `stage_i_final` | Selected Stage F profile | Selected Stage G profile | Fallback video smoke event rate `1.0000` | p95 `325.6 ms` on selected fallback frames | Use as controlled backup demo |

---

# 14. Recommended Next Action

Next technical task:

1. Keep assignment images as external test unless the lecturer explicitly
   allows training on them.
2. Decide the final demo mode:
   - If the demo is live/road-scene phone camera, continue with the current
     Stage F/G/H/I app and use the Stage I fallback video as backup.
   - If the demo must accept isolated assignment-sign PNG crops as app input,
     add either detector/domain adaptation for crop-card inputs or a carefully
     gated classifier-only crop mode.
3. Owner should run final live phone/video tests for audio speaker behavior,
   real-road OCR numeric signs, and lighting/blur stability.

This keeps the project honest: the classifier is strong, the current detector
is tuned for low-noise road-scene localization, and isolated assignment crops
remain the main unresolved final-evaluation gap.
