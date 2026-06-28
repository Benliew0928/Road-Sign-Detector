# Data Source And Citation Ledger

This ledger records every dataset, image source, document, and external resource
used or planned for the road-sign recognition project. It is technical
provenance, not report prose.

Last updated: 2026-06-28

---

## Source Status Legend

| Status | Meaning |
|---|---|
| `accepted` | Source is allowed for project use and has been imported or partially imported. |
| `candidate` | Source is relevant but not imported yet. |
| `review_required` | Source may be useful, but labels/licence/export details must be checked before training use. |
| `external_test_only` | Source should be kept out of training and used for evaluation/reference. |
| `reference_only` | Source can support UI/testing/research notes, but must not be counted as final realistic training coverage. |

---

## Current Accepted Sources

| Source ID | Name | Publisher | Licence | URL / DOI | Local path | Status | Use |
|---|---|---|---|---|---|---|---|
| `emtd_zenodo_1217105` | Extended Malaysian Traffic Sign Dataset | Zenodo | CC BY 4.0 | https://zenodo.org/records/1217105, DOI `10.5281/zenodo.1217105` | `data/raw/emtd/`, `data/processed/emtd_classification/` | `accepted` | Current public Malaysian training/evaluation source. |
| `coursework_assignment_images` | Official coursework assignment images | Coursework package | Assignment-provided academic material | Local coursework files | `data/official/assignment_images/` | `external_test_only` | Assignment mapping, reference, and external test. Not used as normal training data unless explicitly approved. |

---

## Stage C Roboflow Candidate Sources

| Source ID | Name | Publisher / Owner | Licence | URL | Planned local path | Status | Use |
|---|---|---|---|---|---|---|---|
| `roboflow_malaysia_road_sign_v1` | Malaysia Road Sign Dataset | Roboflow Universe / Malaysia Road Sign | CC BY 4.0 according to public project metadata | https://universe.roboflow.com/malaysia-road-sign/malaysia-road-sign-dataset/dataset/1 | `data/raw/roboflow/malaysia_road_sign_v1/` | `review_required` | Candidate source for Malaysian sign detector/classifier data. |
| `roboflow_dr_samsudin_malaysia_road_sign` | Malaysia's Road Sign | Roboflow Universe / Dr Samsudin | Public-domain status shown on public project page, to be rechecked at export time | https://universe.roboflow.com/dr-samsudin/malaysia-s-road-sign | `data/raw/roboflow/dr_samsudin_malaysia_road_sign/` | `review_required` | Candidate source for Malaysian sign detector/classifier data. |

---

## Stage C Downloaded Roboflow Exports

Downloaded on: 2026-06-27

| Source ID | Export format | Raw zip | SHA-256 | Extracted folder | Staged folder | Import status |
|---|---|---|---|---|---|---|
| `roboflow_malaysia_road_sign_v1` | YOLOv8 object detection | `data/raw/roboflow/malaysia_road_sign_dataset_v1/test.v1i.yolov8.zip` | `FEACEF39DE0017069C3E67B2EEC437823EDB84114B620262C8B5418AABE95EF1` | `data/raw/roboflow/malaysia_road_sign_dataset_v1/extracted_full/` | `data/staging/roboflow/roboflow_malaysia_road_sign_v1/` | Staged as public mixed candidate data. |
| `roboflow_dr_samsudin_malaysia_road_sign` | Folder classification | `data/raw/roboflow/dr_samsudin_malaysia_road_sign/Malaysia-s Road Sign.v1-1.folder.zip` | `3D9D69461C2651A6FFCAE521AD258512F65FCF6DB8A74DB9EAA70FB033055C0A` | `data/raw/roboflow/dr_samsudin_malaysia_road_sign/extracted/` | `data/staging/roboflow/roboflow_dr_samsudin_malaysia_road_sign/` | Staged as public mixed candidate data. |

Import script:

- `scripts/import_roboflow_stage_c.py`

Import manifests:

- `data/manifests/roboflow_class_mapping.csv`
- `data/manifests/roboflow_source_manifest.csv`
- `outputs/audit/roboflow_import_audit.json`
- `outputs/audit/post_roboflow_data_gap_report.csv`
- `outputs/audit/post_roboflow_data_gap_report.json`

Important Stage C exclusions:

- `Pass either side` from the first Roboflow dataset was excluded because visual
  audit showed it mixes speed-limit signs and pass-either-side signs.
- `Roadway diverges` from the first Roboflow dataset was excluded because visual
  audit showed it mixes roadway-diverges signs with obstruction/caution signs.
- `Road cones` was excluded because traffic cones are objects, not road-sign
  classes.
- `Selekoh` from the second Roboflow dataset was excluded because it mixes left
  and right chevron boards under one direction-losing label.

---

## Stage C Sprint 01 Commons Top-Up

Downloaded on: 2026-06-28

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `wikimedia_commons_stage_c_sprint_01` | Curated Wikimedia Commons files, per-file licence metadata retained | `data/raw/local_collection/stage_c_sprint_01/wikimedia_commons/` | `data/manifests/stage_c_sprint_01_commons_candidates.csv` | `candidate` | Near-minimum top-up candidates for `no_overtaking`, `keep_right`, and `no_heavy_vehicle`. |

Sprint report:

- `docs/STAGE_C_SPRINT_01_COMMONS_TOPUP_REPORT.md`

Audit files:

- `outputs/audit/stage_c_sprint_01_commons_candidates.json`
- `outputs/audit/post_stage_c_sprint_01_gap_report.csv`
- `outputs/audit/post_stage_c_sprint_01_gap_report.json`

Important notes:

- Licence and attribution are per file; use the candidate manifest, not a single
  blanket licence statement.
- Downloaded files remain candidate data only until Stage D QC/annotation and
  Stage E split freeze.
- `S01-003` was downloaded but excluded because visual review showed a
  heavy-vehicle no-overtaking sign, not generic `no_overtaking`.

---

## Stage C Online Reference Sources 01

Collected on: 2026-06-28

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `stage_c_online_reference_sources_01` | Curated Wikimedia Commons and HuggingFace/Wikipedia Malaysian road-sign references | `data/raw/online_sources/stage_c_reference_01/wikimedia_commons/` | `data/manifests/stage_c_online_reference_sources_01.csv` | `reference_only` | Exact online Malaysian reference signs for unresolved Stage C classes; not real-photo coverage. |

Sprint report:

- `docs/STAGE_C_ONLINE_REFERENCE_SOURCES_01_REPORT.md`

Audit files:

- `outputs/audit/stage_c_online_reference_sources_01.json`
- `outputs/review/stage_c_online_reference_sources_01/`

Important notes:

- These sources are reliable for sign appearance and class anchoring.
- They do not replace real road-scene coverage by themselves.
- `motor_vehicles_only` was downloaded but marked possible mismatch after visual
  review.
- Unresolved classes still require direct JKR artwork extraction or another
  exact labelled public source.

---

## Stage C China/GB Reference Sources 01

Collected on: 2026-06-28

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `stage_c_china_reference_sources_01` | Curated Wikimedia Commons China/GB-style road-sign references discovered through the Road signs in China gallery | `data/raw/online_sources/stage_c_china_reference_01/wikimedia_commons/` | `data/manifests/stage_c_china_reference_sources_01.csv` | `reference_only` | Official-style class anchors for rare assignment signs; not real-photo coverage. |

Sprint report:

- `docs/STAGE_C_CHINA_REFERENCE_SOURCES_01_REPORT.md`

Audit files:

- `outputs/audit/stage_c_china_reference_sources_01.json`
- `outputs/review/stage_c_china_reference_sources_01/`

Important notes:

- These references explain why many rare assignment signs were not found in
  Malaysian public road-scene datasets: they visually match China/GB-style sign
  families.
- 19 reference files were downloaded; 18 are high-confidence class anchors and
  `motor_vehicles_only` is medium-confidence because the downloaded reference is
  a rectangular automobiles-only lane sign.
- `no_lane_changing` remains unresolved; do not force-map it without a verified
  standalone sign-board source.
- These files are more reliable than generated fake signs, but they still do
  not close realistic-photo coverage gaps by themselves.

---

## Stage C Sprint 02 Synthetic Top-Up

Policy update on 2026-06-28:

- This batch is now `reference_only`.
- It is excluded from final realistic dataset coverage.
- Use `outputs/audit/post_stage_c_realistic_gap_report.*` for current Stage C
  coverage decisions.

Generated on: 2026-06-28

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `stage_c_sprint_02_synthetic_topup` | Synthetic variants generated from P5-cleaned EMTD crops and Sprint 01 Commons seed | `data/generated/stage_c_sprint_02_synthetic_topup/` | `data/manifests/stage_c_sprint_02_synthetic_candidates.csv` | `reference_only` | UI/testing/reference support for `side_road_right` and `no_heavy_vehicle`; not final realistic coverage. |

Sprint report:

- `docs/STAGE_C_SPRINT_02_SYNTHETIC_TOPUP_REPORT.md`

Audit files:

- `outputs/audit/stage_c_sprint_02_synthetic_candidates.json`
- `outputs/audit/post_stage_c_sprint_02_gap_report.csv`
- `outputs/audit/post_stage_c_sprint_02_gap_report.json`

Important notes:

- Synthetic top-ups are not real camera coverage.
- Seed-derived generated candidates must be handled carefully during Stage E
  split freeze to avoid leakage between seed and generated variants.
- These files do not close final realistic coverage gaps.

---

## Stage C Sprint 03 Mandatory Direction Symbols

Policy update on 2026-06-28:

- This batch is now `reference_only`.
- It is excluded from final realistic dataset coverage.
- Use `outputs/audit/post_stage_c_realistic_gap_report.*` for current Stage C
  coverage decisions.

Generated on: 2026-06-28

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `stage_c_sprint_03_mandatory_direction_symbols` | Project-owned generated reference-symbol candidates | `data/generated/stage_c_sprint_03_mandatory_direction_symbols/` | `data/manifests/stage_c_sprint_03_mandatory_direction_symbols.csv` | `reference_only` | Symbol reference/UI support for `straight_ahead`, `turn_left`, and `turn_right`; not final realistic coverage. |

Sprint report:

- `docs/STAGE_C_SPRINT_03_MANDATORY_DIRECTION_SYMBOLS_REPORT.md`

Audit files:

- `outputs/audit/stage_c_sprint_03_mandatory_direction_symbols.json`
- `outputs/audit/post_stage_c_sprint_03_gap_report.csv`
- `outputs/audit/post_stage_c_sprint_03_gap_report.json`

Important notes:

- These are generated reference-symbol candidates, not real road-scene data.
- They do not close final realistic coverage gaps and should be replaced or
  supplemented with real camera/demo samples.

---

## Stage C Sprint 04 Compound Mandatory Symbols

Policy update on 2026-06-28:

- This batch is now `reference_only`.
- It is excluded from final realistic dataset coverage.
- Use `outputs/audit/post_stage_c_realistic_gap_report.*` for current Stage C
  coverage decisions.

Generated on: 2026-06-28

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `stage_c_sprint_04_compound_mandatory_symbols` | Project-owned generated reference-symbol candidates | `data/generated/stage_c_sprint_04_compound_mandatory_symbols/` | `data/manifests/stage_c_sprint_04_compound_mandatory_symbols.csv` | `reference_only` | Symbol reference/UI support for compound mandatory direction signs and `pass_either_side`; not final realistic coverage. |

Sprint report:

- `docs/STAGE_C_SPRINT_04_COMPOUND_MANDATORY_SYMBOLS_REPORT.md`

Audit files:

- `outputs/audit/stage_c_sprint_04_compound_mandatory_symbols.json`
- `outputs/audit/post_stage_c_sprint_04_gap_report.csv`
- `outputs/audit/post_stage_c_sprint_04_gap_report.json`

Important notes:

- These are generated reference-symbol candidates, not real road-scene data.
- `pass_either_side` generation was based on the current P5 contact sheet shape.
- Roundabout and compound direction symbols should be supplemented with real or
  official-reference samples when possible.
- These files do not close final realistic coverage gaps.

---

## Stage C Sprint 05 Prohibitory Direction Symbols

Policy update on 2026-06-28:

- This batch is now `reference_only`.
- It is excluded from final realistic dataset coverage.
- Use `outputs/audit/post_stage_c_realistic_gap_report.*` for current Stage C
  coverage decisions.

Generated on: 2026-06-28

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `stage_c_sprint_05_prohibitory_direction_symbols` | Project-owned generated reference-symbol candidates | `data/generated/stage_c_sprint_05_prohibitory_direction_symbols/` | `data/manifests/stage_c_sprint_05_prohibitory_direction_symbols.csv` | `reference_only` | Symbol reference/UI support for prohibitory direction signs; not final realistic coverage. |

Sprint report:

- `docs/STAGE_C_SPRINT_05_PROHIBITORY_DIRECTION_SYMBOLS_REPORT.md`

Audit files:

- `outputs/audit/stage_c_sprint_05_prohibitory_direction_symbols.json`
- `outputs/audit/post_stage_c_sprint_05_gap_report.csv`
- `outputs/audit/post_stage_c_sprint_05_gap_report.json`

Important notes:

- These are generated reference-symbol candidates, not real road-scene data.
- Prohibition slash placement should be reviewed during Stage D visual QC.
- These files do not close final realistic coverage gaps.

---

## Stage C Sprint 06 Warning Symbols

Generated on: 2026-06-28

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `stage_c_sprint_06_warning_symbols` | Project-owned generated warning-symbol references | `data/generated/stage_c_sprint_06_warning_symbols/` | `data/manifests/stage_c_sprint_06_warning_symbols.csv` | `reference_only` | UI/testing/reference support for `bicycle_crossing`, `school_zone`, `steep_descent`, and `uneven_road`; not final realistic coverage. |

Audit files:

- `outputs/audit/stage_c_sprint_06_warning_symbols.json`
- `outputs/review/stage_c_sprint_06_warning_symbols/`

Important notes:

- Visual review found this generated batch too fake/unrealistic for final model
  coverage.
- These files do not close final realistic coverage gaps.

---

## Stage C Sprint 07 Regulatory/Text Symbols

Generated on: 2026-06-28

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `stage_c_sprint_07_regulatory_text_symbols` | Project-owned generated regulatory/text references | `data/generated/stage_c_sprint_07_regulatory_text_symbols/` | `data/manifests/stage_c_sprint_07_regulatory_text_symbols.csv` | `reference_only` | UI/testing/reference support for `motor_vehicles_only`, `no_lane_changing`, `no_motor_vehicles`, `slow_text`, `sound_horn`, `stop_for_checking`, and `width_restriction`; not final realistic coverage. |

Audit files:

- `outputs/audit/stage_c_sprint_07_regulatory_text_symbols.json`
- `outputs/review/stage_c_sprint_07_regulatory_text_symbols/`

Important notes:

- Visual review found this generated batch too fake/unrealistic for final model
  coverage.
- These files are not reliable enough for final OCR or multilingual text
  claims.
- These files do not close final realistic coverage gaps.

---

## Import And Cleaning Rules

- Preserve every source dataset in its own raw folder.
- Do not overwrite source labels; store source labels and mapped project labels
  separately.
- Map labels to existing `semantic_sign_id` values when the sign meaning is the
  same.
- Create a new class only when the road sign is visually and semantically
  different from all existing catalogue entries.
- Keep public dataset images separate from official coursework images.
- Keep duplicate or near-duplicate images from leaking across train,
  validation, and test splits.
- Record source URL, licence, export date, export format, local path, and any
  class-mapping decision before using the data for training.

---

## Generated Artifacts To Update During Stage C

| Artifact | Purpose |
|---|---|
| `data/raw/roboflow/...` | Unmodified downloaded Roboflow exports. |
| `data/staging/roboflow/...` | Normalized images and annotations before final dataset merge. |
| `data/manifests/roboflow_source_manifest.csv` | One row per imported image/annotation. |
| `data/manifests/roboflow_class_mapping.csv` | Source class to project `semantic_sign_id` mapping. |
| `outputs/audit/roboflow_import_audit.json` | Import counts, rejected samples, new classes, and data-quality warnings. |
| `docs/STAGE_C_ROBOFLOW_IMPORT_REPORT.md` | Human-readable import result and local path index. |
| `data/manifests/stage_c_sprint_01_commons_candidates.csv` | Per-file Commons candidate provenance and review status. |
| `docs/STAGE_C_SPRINT_01_COMMONS_TOPUP_REPORT.md` | Human-readable Stage C Sprint 01 result. |
| `data/manifests/stage_c_sprint_02_synthetic_candidates.csv` | Synthetic candidate provenance and seed linkage. |
| `docs/STAGE_C_SPRINT_02_SYNTHETIC_TOPUP_REPORT.md` | Human-readable Stage C Sprint 02 result. |
| `data/manifests/stage_c_sprint_03_mandatory_direction_symbols.csv` | Generated mandatory-direction reference candidate manifest. |
| `docs/STAGE_C_SPRINT_03_MANDATORY_DIRECTION_SYMBOLS_REPORT.md` | Human-readable Stage C Sprint 03 result. |
| `data/manifests/stage_c_sprint_04_compound_mandatory_symbols.csv` | Generated compound mandatory-direction candidate manifest. |
| `docs/STAGE_C_SPRINT_04_COMPOUND_MANDATORY_SYMBOLS_REPORT.md` | Human-readable Stage C Sprint 04 result. |
| `data/manifests/stage_c_sprint_05_prohibitory_direction_symbols.csv` | Generated prohibitory-direction candidate manifest. |
| `docs/STAGE_C_SPRINT_05_PROHIBITORY_DIRECTION_SYMBOLS_REPORT.md` | Human-readable Stage C Sprint 05 result. |
| `data/manifests/stage_c_sprint_06_warning_symbols.csv` | Generated warning-symbol reference manifest. |
| `data/manifests/stage_c_sprint_07_regulatory_text_symbols.csv` | Generated regulatory/text reference manifest. |
| `docs/STAGE_C_GENERATED_REFERENCE_POLICY.md` | Policy explaining generated reference-only status and realistic coverage rules. |
| `data/manifests/stage_c_china_reference_sources_01.csv` | Per-file China/GB-style reference-sign provenance for rare assignment classes. |
| `docs/STAGE_C_CHINA_REFERENCE_SOURCES_01_REPORT.md` | Human-readable Stage C China/GB reference-source result. |
| `outputs/audit/post_stage_c_realistic_gap_report.csv` | Corrected Stage C report that excludes generated references from final realistic coverage. |
