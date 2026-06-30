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

- Do not create or use AI-generated/fake sign images for this project dataset.
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

## Completion Gate

- [ ] Each `must` class reaches at least 50 clean candidate crops.
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

## Stage D Zero-Gap Minimum-QC Batch On 2026-07-01

To avoid generating many messy review files, the zero-gap Stage D batch writes
one row-level manifest and one contact sheet:

- `scripts/run_stage_d_zero_gap_qc.py`
- `data/manifests/stage_d_zero_gap_qc_manifest.csv`
- `outputs/review/00_CURRENT_REVIEW/stage_d_zero_gap_qc_contact_sheet.jpg`

Current result:

- Every `must` class whose `gap_to_minimum` is `0` is completed at the minimum
  Stage D level.
- 42 classes completed.
- 50 accepted crops per class, 2,100 accepted crops total.
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
- The remaining 10 `must` classes still below minimum realistic coverage need
  more data before this Stage D process can complete them.

## Completion Gate

- [~] All `must` classes reviewed by contact sheet. All 42 zero-gap `must`
      classes are completed at minimum Stage D level; 10 below-minimum classes
      remain blocked by collection coverage.
- [x] Zero-gap crop selections are reproducible by manifest and
      `scripts/run_stage_d_zero_gap_qc.py`.
- [x] Zero-gap batch requires no manual drag-and-drop corrections.

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

## Completion Gate

- [ ] No assignment-test leakage into training.
- [ ] No duplicate leakage across train/validation/test.
- [ ] Every `must` class appears in train and validation, or limitation is
      documented.

---

# 7. Stage F - Retrain Detector If Needed

## When To Retrain P8

Retrain the detector if:

- The app fails to locate the sign.
- The bounding box misses most of the sign.
- Small signs are not detected.
- Camera/video demo signs are not segmented.

Do not retrain P8 if:

- The sign is detected correctly but classified as unknown.

That is a P9 classifier problem.

## Deliverables

- YOLO training run folder
- ONNX export
- Validation/test mAP and recall
- Runtime benchmark

## Completion Gate

- [ ] Assignment/demo signs are detected.
- [ ] Runtime stays below the project target.
- [ ] Detector model is exported and app-compatible.

---

# 8. Stage G - Retrain Classifier

## When To Retrain P9

Retrain the classifier if:

- The app detects a sign but returns `unknown`.
- The app detects a sign but predicts the wrong meaning.
- New target classes were added.
- P5 label cleanup changed many crops.

This is the most important next model stage for the current problem.

## Training Candidates

| Model | Use |
|---|---|
| MobileNetV3 | Fast baseline and CPU fallback |
| EfficientNetV2 | Stronger semantic classifier |
| Embedding/prototype gate | Safer unknown rejection |
| YOLO classifier head or YOLO detector class labels | Optional comparison |

## Deliverables

- `outputs/training/final_classifier_*/`
- `models/exported/final/semantic_classifier.onnx`
- `outputs/evaluation/final_classifier_metrics.json`
- Confusion matrix
- Assignment-sign evaluation
- Common-Malaysian-sign evaluation

## Completion Gate

- [ ] Assignment signs no longer default to unknown when visible and in scope.
- [ ] Common signs have acceptable class-level accuracy.
- [ ] Unknown rejection is still conservative.
- [ ] Final classifier is integrated into the app.

---

# 9. Stage H - OCR, ADAS Rules, And App Integration

## Goal

Convert recognition into useful autonomous-driving style output.

## Examples

| Recognition | App output |
|---|---|
| `maximum_speed`, value `50` | Target speed 50 km/h |
| `school_zone` | Slow down, watch children |
| `no_entry` | Do not enter |
| `height_restriction`, value `5.4m` | Warn if vehicle height exceeds limit |
| `pedestrian_crossing` | Watch pedestrians |

## Deliverables

- OCR value extraction for numeric signs
- Deterministic ADAS rules
- UI model version display
- Audio warning for selected demo signs

## Completion Gate

- [ ] App shows sign meaning, confidence, OCR/value, and ADAS action.
- [ ] Audio warnings work for demo signs.
- [ ] Low confidence does not trigger dangerous commands.

---

# 10. Stage I - Final Evaluation

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

## Completion Gate

- [ ] App handles assignment signs in scope.
- [ ] App handles chosen common Malaysian signs.
- [ ] App runs within required time.
- [ ] Demo has a reliable fallback video if live camera conditions are poor.

---

# 11. Master Tracker

| Stage | Status | Owner | Deliverable | Notes |
|---|---:|---|---|---|
| A. Target class lock | `[x]` | Codex | `target_sign_classes.csv` | Baseline locked on 2026-06-27; update only if final demo signs change |
| B. Data gap audit | `[x]` | Codex | `data_gap_report.csv` | Generated on 2026-06-27; use `collection_rank` as the next data queue |
| C. Data collection sprint | `[~]` | Owner/Codex | Local/public raw data | Current tracker has 42 must classes meeting minimum realistic candidate count and 10 still below minimum |
| D. Annotation and QC | `[~]` | Owner/Codex | QC sheets and correction manifests | All 42 zero-gap must classes completed at minimum Stage D level: 2,100 accepted crops in `stage_d_zero_gap_qc_manifest.csv`; 10 below-minimum must classes still need data then QC |
| E. Freeze dataset split | `[ ]` | Codex | Final train/val/test manifests | Needed before retraining |
| F. Detector retrain | `[ ]` | Codex | Final detector ONNX | Only if detection is weak |
| G. Classifier retrain | `[ ]` | Codex | Final classifier ONNX | Most important next model step |
| H. App integration | `[ ]` | Codex | Updated live app | Needs model version display |
| I. Final evaluation | `[ ]` | Owner/Codex | Acceptance metrics | Assignment and demo tests |

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
| DET-FINAL-01 |  |  |  |  |  |  |
| CLS-FINAL-01 |  |  | MobileNetV3 |  |  |  |
| CLS-FINAL-02 |  |  | EfficientNetV2 |  |  |  |
| E2E-FINAL-01 |  |  |  | Assignment test |  |  |
| LIVE-FINAL-01 |  |  |  | Camera demo |  |  |

---

# 14. Recommended Next Action

Next technical task:

1. Use `outputs/audit/post_stage_c_realistic_gap_report.csv` as the collection
   queue.
2. Collect real/public/official/local-camera data for the 24 remaining `must`
   classes below minimum.
3. Keep assignment images as external test unless the lecturer explicitly
   allows training on them.
4. After enough realistic candidates exist, run Stage D targeted QC and Stage E
   split freeze.
5. Retrain the classifier, then only retrain the detector if the app cannot
   locate visible signs.

This prevents the project from spending unlimited time cleaning or generating
low-reliability data while the app still cannot recognize assignment signs.
