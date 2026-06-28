# Stage C Roboflow Import Report

Generated: 2026-06-27

This report records the Roboflow public-dataset import performed for the road
sign recognition project. These files are candidate training data, not the final
frozen dataset.

---

## Imported Sources

| Source ID | Source | Task | Licence | Raw images reported by source | Local raw zip |
|---|---|---|---|---:|---|
| `roboflow_malaysia_road_sign_v1` | Malaysia Road Sign Dataset | Object detection | CC BY 4.0 | 23,068 generated images | `data/raw/roboflow/malaysia_road_sign_dataset_v1/test.v1i.yolov8.zip` |
| `roboflow_dr_samsudin_malaysia_road_sign` | Malaysia's Road Sign | Classification | Public Domain / CC0 metadata | 844 images | `data/raw/roboflow/dr_samsudin_malaysia_road_sign/Malaysia-s Road Sign.v1-1.folder.zip` |

Source ledger:

- `docs/DATA_SOURCE_CITATION_LEDGER.md`
- `data/manifests/dataset_sources.json`

---

## Local Paths

Raw exports:

- `data/raw/roboflow/malaysia_road_sign_dataset_v1/`
- `data/raw/roboflow/dr_samsudin_malaysia_road_sign/`

Extracted source folders:

- `data/raw/roboflow/malaysia_road_sign_dataset_v1/extracted_full/`
- `data/raw/roboflow/dr_samsudin_malaysia_road_sign/extracted/`

Class-separated staged folders:

- `data/staging/roboflow/roboflow_malaysia_road_sign_v1/classification_crops/`
- `data/staging/roboflow/roboflow_dr_samsudin_malaysia_road_sign/classification/`

Manifests and audits:

- `data/manifests/roboflow_class_mapping.csv`
- `data/manifests/roboflow_source_manifest.csv`
- `outputs/audit/roboflow_import_audit.json`
- `outputs/audit/post_roboflow_data_gap_report.csv`
- `outputs/audit/post_roboflow_data_gap_report.json`

Review sheets:

- `outputs/review/stage_c_roboflow_mapping_sheets/`
- `outputs/review/stage_c_roboflow_import_sheets/`

Import script:

- `scripts/import_roboflow_stage_c.py`
- `scripts/generate_post_roboflow_gap_report.py`

---

## Import Result

| Metric | Count |
|---|---:|
| Source rows inspected | 29,394 |
| Accepted candidate samples | 22,611 |
| Accepted candidate samples for target classes | 22,274 |
| Rejected samples | 6,783 |
| Accepted semantic classes | 57 |
| Classes with Roboflow gain against target list | 52 |
| `must` classes meeting minimum after Roboflow candidate import | 25 |
| `must` classes still below minimum or no Roboflow gain after import | 27 |

Rejected samples:

| Reason | Count |
|---|---:|
| Tiny crop | 2,257 |
| Noisy mixed source class | 3,948 |
| Non-sign object | 489 |
| Ambiguous direction label | 29 |
| Exact duplicate crop | 51 |
| Known filename-level outlier | 9 |

---

## New Class Candidates

These are not yet approved catalogue classes. They are staged separately so they
do not get forced into the wrong existing class.

| New candidate | Source label | Reason |
|---|---|---|
| `disabled_parking` | `OKU Parking` | Distinct accessible/OKU parking sign. |
| `lane_merges_left` | `Traffic merging to the left` | Direction differs from traffic merging from left. |
| `motorcycles_only` | `Motorcycles only` | Blue mandatory motorcycle-only sign. |
| `narrow_bridge` | `Narrow bridge` | Bridge-specific warning sign. |
| `tow_away_zone` | `Towing area` | Tow-away/towing restriction sign. |

Candidate registry:

- `configs/catalogue/stage_c_new_class_candidates.json`

---

## Important Exclusions

These were intentionally not staged as training data:

| Source label | Source | Reason |
|---|---|---|
| `Pass either side` | Malaysia Road Sign Dataset | Visual audit showed a mix of speed-limit and pass-either-side signs. |
| `Roadway diverges` | Malaysia Road Sign Dataset | Visual audit showed a mix of roadway-diverges and obstruction/caution signs. |
| `Road cones` | Malaysia Road Sign Dataset | Traffic cones are objects, not road-sign classes. |
| `Selekoh` | Malaysia's Road Sign | Mixed left/right chevrons under one label, losing direction. |

---

## Remaining `must` Class Gaps

After this import, the highest-priority classes still needing data include:

- `bicycle_crossing`
- `keep_right`
- `motor_vehicles_only`
- `no_heavy_vehicle`
- `no_left_or_right_turn`
- `no_motor_vehicles`
- `no_overtaking`
- `no_straight_ahead`
- `no_straight_or_left`
- `pass_either_side`
- `residential_area_ahead`
- `roadway_diverges`
- `roundabout_mandatory`
- `school_zone`
- `side_road_right`
- `slow_text`
- `sound_horn`
- `steep_descent`
- `stop_for_checking`
- `straight_ahead`
- `straight_or_right`
- `turn_left`
- `turn_left_or_right`
- `turn_right`
- `uneven_road`
- `width_restriction`
- `tractors_ahead`

Near-minimum classes:

- `no_overtaking`: 47 candidate total, needs 3 more.
- `keep_right`: 47 candidate total, needs 3 more.
- `side_road_right`: 45 candidate total, needs 5 more.
- `no_heavy_vehicle`: 42 candidate total, needs 8 more.

Full gap table:

- `outputs/audit/post_roboflow_data_gap_report.csv`

---

## How To Check

Open these files first:

1. `outputs/audit/roboflow_import_audit.json`
2. `data/manifests/roboflow_class_mapping.csv`
3. `outputs/audit/post_roboflow_data_gap_report.csv`

For visual checking, open:

1. `outputs/review/stage_c_roboflow_import_sheets/roboflow_malaysia_road_sign_v1_accepted_classes.jpg`
2. `outputs/review/stage_c_roboflow_import_sheets/roboflow_dr_samsudin_malaysia_road_sign_accepted_classes.jpg`

The staged class folders are already separated by project semantic label under:

- `data/staging/roboflow/`

Do not train final models directly from these folders yet. Next step is Stage D:
targeted QC, source balancing, leakage-safe split freeze, then classifier
retraining.
