# Data Source And Citation Ledger

This ledger records every dataset, image source, document, and external resource
used or planned for the road-sign recognition project. It is technical
provenance, not report prose.

Last updated: 2026-06-29

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
- These files support class definition, but they still do
  not close realistic-photo coverage gaps by themselves.

---

## Stage C Gap Fill 01 TT100K Exact Real Crops

Collected on: 2026-06-29

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `stage_c_gap_fill_01_tt100k` | Tsinghua-Tencent 100K / TT100K via Hugging Face mirror | `data/raw/online_sources/stage_c_gap_fill_01_tt100k/` | `data/manifests/stage_c_gap_fill_01_tt100k_candidates.csv` | `candidate` | Real road-scene crop candidates for exact rare assignment-style signs. |

Source links:

- Hugging Face mirror: https://huggingface.co/datasets/Genius-Society/tt100k
- Original dataset: https://cg.cs.tsinghua.edu.cn/traffic-sign/
- Visual class legend: https://lijiancheng0614.github.io/2019/04/16/2019_04_16_TT100K/

Sprint report:

- `docs/STAGE_C_GAP_FILL_01_TT100K_REPORT.md`

Important notes:

- No AI-generated images were used.
- Four train shards were processed in this sprint.
- The collected real-road crop candidates are still pending Stage D visual QC
  and Stage E split freeze before final model training.
- The active tracker CSV was locked, so a pending updated tracker was written
  to `data/manifests/CURRENT_DATA_PROGRESS.pending_stage_c_gap_fill_01.csv`.

---

## Stage C Gap Fill 02 Public Real Sources

Collected on: 2026-06-29

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `stage_c_gap_fill_02_gtsrb` | German Traffic Sign Recognition Benchmark / GTSRB | `data/raw/online_sources/stage_c_gap_fill_02_public_real_sources/` | `data/manifests/stage_c_gap_fill_02_public_real_sources_candidates.csv` | `candidate` | Exact public real-road cropped classification candidates for 9 rare or under-covered classes. |
| `stage_c_gap_fill_02_tt100k_p6` | TT100K local shards from Stage C Gap Fill 01 | `data/raw/online_sources/stage_c_gap_fill_02_public_real_sources/crops/no_bicycle/` | `data/manifests/stage_c_gap_fill_02_public_real_sources_candidates.csv` | `candidate` | Real-road detection crops for `no_bicycle`. |

Source links:

- GTSRB official benchmark: https://benchmark.ini.rub.de/
- GTSRB training zip: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip
- TT100K Hugging Face mirror: https://huggingface.co/datasets/Genius-Society/tt100k
- Original TT100K dataset: https://cg.cs.tsinghua.edu.cn/traffic-sign/
- TT100K visual class legend: https://lijiancheng0614.github.io/2019/04/16/2019_04_16_TT100K/

Sprint report:

- `docs/STAGE_C_GAP_FILL_02_PUBLIC_REAL_SOURCES_REPORT.md`

Audit files:

- `outputs/audit/stage_c_gap_fill_02_public_real_sources_candidates.json`
- `outputs/audit/post_stage_c_realistic_gap_report.csv`
- `outputs/audit/post_stage_c_gap_fill_02_gap_report.json`
- `outputs/review/stage_c_gap_fill_02_public_real_sources/`

Important notes:

- No AI-generated images were used.
- GTSRB was used only for exact source classes, not near-meaning substitutes.
- The collector uses class-spread sampling to avoid taking one near-duplicate
  source sequence as the whole class top-up.
- Samples remain candidate data pending Stage D visual QC and Stage E split
  freeze.

---

## Stage C Gap Fill 03-06 Rare Class Mining

Collected on: 2026-06-29

| Source ID | Source | Local root | Manifest | Status | Use |
|---|---|---|---|---|---|
| `stage_c_gap_fill_03_remaining_tt100k` | Tsinghua-Tencent 100K / TT100K via `Genius-Society/tt100k` | `data/raw/online_sources/stage_c_gap_fill_03_remaining_tt100k/` | `data/manifests/stage_c_gap_fill_03_remaining_tt100k_candidates.csv` | `candidate` | Exact TT100K real road-scene crops for rare China/assignment-style classes. |
| `stage_c_gap_fill_04_mapillary_exact` | Mapillary Traffic Sign Dataset via `ThankGod/mapillary_traffic_sign_dataset` | `data/raw/online_sources/stage_c_gap_fill_04_mapillary/` | `data/manifests/stage_c_gap_fill_04_mapillary_exact_candidates.csv` | `exhausted_rejected` | Rejection audit only; no accepted candidates. |
| `stage_c_gap_fill_05_prashant_tt100k_remote` | Alternate TT100K mirror via `PrashantDixit0/TT-100K` | `data/raw/online_sources/stage_c_gap_fill_05_prashant_tt100k/` | `data/manifests/stage_c_gap_fill_05_prashant_tt100k_remote_candidates.csv` | `candidate` | Remote metadata-scanned exact TT100K labels; accepted only visually valid rows. |
| `stage_c_gap_fill_06_roboflow_roadway_diverges` | Malaysia Road Sign Dataset v1 via Roboflow | `data/raw/online_sources/stage_c_gap_fill_06_roboflow_roadway_diverges/` | `data/manifests/stage_c_gap_fill_06_roboflow_roadway_diverges_candidates.csv` | `candidate` | Exact real-road `roadway_diverges` crops. |

Source links:

- TT100K Hugging Face mirror: https://huggingface.co/datasets/Genius-Society/tt100k
- Alternate TT100K mirror: https://huggingface.co/datasets/PrashantDixit0/TT-100K
- Original TT100K dataset: https://cg.cs.tsinghua.edu.cn/traffic-sign/
- Mapillary HF mirror: https://huggingface.co/datasets/ThankGod/mapillary_traffic_sign_dataset
- Mapillary Traffic Sign Dataset: https://www.mapillary.com/dataset/trafficsign
- Roboflow Malaysia Road Sign Dataset v1: https://universe.roboflow.com/test-22a9b/test-fwdci/dataset/1

Sprint report:

- `docs/STAGE_C_GAP_FILL_03_TO_06_RARE_CLASS_MINING_REPORT.md`

Audit files:

- `outputs/audit/stage_c_gap_fill_03_remaining_tt100k_candidates.json`
- `outputs/audit/stage_c_gap_fill_04_mapillary_exact_candidates.json`
- `outputs/audit/stage_c_gap_fill_05_prashant_tt100k_remote_candidates.json`
- `outputs/audit/stage_c_gap_fill_06_roboflow_roadway_diverges_candidates.json`
- `outputs/audit/post_stage_c_realistic_gap_report.csv`
- `outputs/audit/post_stage_c_gap_fill_06_gap_report.json`

Important notes:

- No AI-generated images were used.
- Stage 03 completed the full local `Genius-Society/tt100k` train,
  validation, and test mirror.
- Stage 05 scanned all 171 remote parquet files / 16,811 rows in
  `PrashantDixit0/TT-100K` without downloading the full 73 GB mirror.
- Mapillary `warning--pass-left-or-right` is not `roadway_diverges`.
- Mapillary `regulatory--no-straight-through` is not the red
  `no_straight_ahead` prohibition.
- Prashant TT100K `pb`/`pb5` was visually rejected for `width_restriction`.
- Accepted crops remain candidate data pending Stage D visual QC and Stage E
  split freeze.

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

## Artifacts To Update During Stage C

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
| `data/manifests/stage_c_china_reference_sources_01.csv` | Per-file China/GB-style reference-sign provenance for rare assignment classes. |
| `docs/STAGE_C_CHINA_REFERENCE_SOURCES_01_REPORT.md` | Human-readable Stage C China/GB reference-source result. |
| `outputs/audit/post_stage_c_realistic_gap_report.csv` | Current Stage C realistic coverage report. |
