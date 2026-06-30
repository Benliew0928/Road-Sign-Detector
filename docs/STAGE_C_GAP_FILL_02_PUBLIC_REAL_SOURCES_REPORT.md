# Stage C Gap Fill 02 Public Real Sources Report

Generated: 2026-06-29

## Purpose

Fill 10 realistic candidate-data gaps using exact public dataset labels only.
This sprint deliberately avoided AI-generated images and avoided near-meaning
label guesses.

## Accepted Sources

| Source | Use | Local storage |
|---|---|---|
| German Traffic Sign Recognition Benchmark / GTSRB | Cropped real-road classification images for exact warning/mandatory/prohibition classes. | `data/raw/online_sources/stage_c_gap_fill_02_public_real_sources/gtsrb/` |
| Tsinghua-Tencent 100K / TT100K | Real-road detection crops from locally downloaded arrow shards for `no_bicycle`. | `data/raw/online_sources/stage_c_gap_fill_02_public_real_sources/crops/no_bicycle/` |

Source URLs:

- GTSRB official benchmark: https://benchmark.ini.rub.de/
- GTSRB training zip used by torchvision: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip
- TT100K Hugging Face mirror: https://huggingface.co/datasets/Genius-Society/tt100k
- TT100K original dataset: https://cg.cs.tsinghua.edu.cn/traffic-sign/
- TT100K class legend: https://lijiancheng0614.github.io/2019/04/16/2019_04_16_TT100K/

## Result

| Project class | Source label | New candidates |
|---|---|---:|
| `bicycle_crossing` | GTSRB class 29, Bicycles crossing | 50 |
| `no_bicycle` | TT100K `p6` | 20 |
| `no_heavy_vehicle` | GTSRB class 16, Vehicles over 3.5 metric tons prohibited | 50 |
| `roundabout_mandatory` | GTSRB class 40, Roundabout mandatory | 50 |
| `straight_ahead` | GTSRB class 35, Ahead only | 50 |
| `straight_or_left` | GTSRB class 37, Go straight or left | 50 |
| `straight_or_right` | GTSRB class 36, Go straight or right | 50 |
| `turn_left` | GTSRB class 34, Turn left ahead | 50 |
| `turn_right` | GTSRB class 33, Turn right ahead | 50 |
| `uneven_road` | GTSRB class 22, Bumpy road | 50 |

Total new candidates: 470.

## Quality Controls Applied

- Used only exact source labels with known class IDs.
- Kept source labels and mapped project labels separately in the manifest.
- Used GTSRB class-spread sampling instead of the first sequential files, because
  the first-pass audit showed near-duplicate sequence frames.
- Regenerated contact sheets after the sampler fix.
- Kept all samples as `auto_exact_label_pending_stage_d_visual_qc`; they are not
  final frozen training data yet.

## Artifacts

| Artifact | Path |
|---|---|
| Collector script | `scripts/collect_stage_c_gap_fill_02_public_real_sources.py` |
| Candidate manifest | `data/manifests/stage_c_gap_fill_02_public_real_sources_candidates.csv` |
| Audit JSON | `outputs/audit/stage_c_gap_fill_02_public_real_sources_candidates.json` |
| Current gap report | `outputs/audit/post_stage_c_realistic_gap_report.csv` |
| Crop root | `data/raw/online_sources/stage_c_gap_fill_02_public_real_sources/crops/` |
| Review sheets | `outputs/review/stage_c_gap_fill_02_public_real_sources/` |
| Compact QA sheet | `outputs/review/stage_c_gap_fill_02_public_real_sources/_qa_10class_6each.jpg` |

## Tracker Impact

The active tracker was updated:

- `data/manifests/CURRENT_DATA_PROGRESS.csv`

Current must-have class coverage after this sprint:

- Must-have classes: 52
- Must-have classes meeting minimum candidate count: 37
- Must-have classes still below minimum candidate count: 15

## Remaining Caveat

These candidates close the collection-count gap for 10 classes, but they still
need Stage D visual QC before Stage E split freeze and model training.
