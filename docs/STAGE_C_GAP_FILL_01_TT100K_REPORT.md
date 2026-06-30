# Stage C Gap Fill 01: TT100K Exact Real Crops

Generated: 2026-06-29

## Purpose

This sprint targeted five minimum-coverage gaps:

- `bicycle_crossing`
- `motor_vehicles_only`
- `no_left_or_right_turn`
- `no_motor_vehicles`
- `no_straight_ahead`

Only real road crops from a labelled public dataset were accepted. No
AI-generated or fake sign images were used.

## Source

Primary source:

- Tsinghua-Tencent 100K / TT100K via Hugging Face mirror:
  <https://huggingface.co/datasets/Genius-Society/tt100k>
- Original dataset site:
  <https://cg.cs.tsinghua.edu.cn/traffic-sign/>
- TT100K visual class legend used for mapping evidence:
  <https://lijiancheng0614.github.io/2019/04/16/2019_04_16_TT100K/>

Licence note:

- The Hugging Face dataset card records `cc-by-nc-4.0`.
- Ultralytics documentation for TT100K records non-commercial licensing.
- Treat this source as academic/non-commercial only unless the final project
  receives a clearer licence approval.

## Exact Class Mapping

The mapping was not guessed from label numbers alone. It was verified from the
TT100K visual legend and the coursework images.

| Project class | TT100K label | TT100K index | Evidence |
|---|---:|---:|---|
| `bicycle_crossing` | `w56` | 161 | Legend icon shows triangular bicycle warning sign. |
| `motor_vehicles_only` | `i4` | 9 | Legend icon shows blue circular car-only sign. |
| `no_left_or_right_turn` | `p20` | 32 | Legend icon shows compound no-left/no-right-turn sign. |
| `no_motor_vehicles` | `p10` | 21 | Legend icon shows red-ring motor-vehicle/car prohibition. |
| `no_straight_ahead` | `p14` | 25 | Legend icon shows no-straight-ahead sign. |

## Result

Processed TT100K shards:

- `default/train/data-00000-of-00014.arrow`
- `default/train/data-00001-of-00014.arrow`
- `default/train/data-00002-of-00014.arrow`
- `default/train/data-00003-of-00014.arrow`

Accepted real-road crop candidates:

| Project class | New TT100K crops | Status |
|---|---:|---|
| `motor_vehicles_only` | 85 | Meets minimum candidate count; still needs Stage D QC. |
| `no_motor_vehicles` | 47 | Short by 3 crops; still needs Stage D QC. |
| `no_straight_ahead` | 5 | Still far below minimum. |
| `bicycle_crossing` | 0 | Not found in first four train shards. |
| `no_left_or_right_turn` | 0 | Not found in first four train shards. |

Tiny crops skipped automatically:

- `motor_vehicles_only`: 45
- `no_motor_vehicles`: 23
- `no_straight_ahead`: 2

## Local Artifacts

- Manifest:
  `data/manifests/stage_c_gap_fill_01_tt100k_candidates.csv`
- Audit JSON:
  `outputs/audit/stage_c_gap_fill_01_tt100k_candidates.json`
- Crops:
  `data/raw/online_sources/stage_c_gap_fill_01_tt100k/crops/`
- Review sheets:
  `outputs/review/stage_c_gap_fill_01_tt100k/`
- Combined review sheet:
  `outputs/review/stage_c_gap_fill_01_tt100k/_all_tt100k_gap_fill_candidates.jpg`
- Pending updated tracker:
  `data/manifests/CURRENT_DATA_PROGRESS.pending_stage_c_gap_fill_01.csv`

The active tracker CSV was locked during the run, so the updated version was
written as a pending replacement instead of overwriting the locked file.

## Visual QC Note

The combined contact sheet was reviewed after extraction. The accepted crops are
real road-scene crops and visually match the three classes collected in this
sprint. They are still marked as pending Stage D QC because final training data
must pass the normal project review/freeze process.

## Commons Attempt

A curated Wikimedia Commons official-reference collector was created, but the
Commons API returned HTTP 429 rate-limit errors during this sprint. The failed
candidate manifest was removed so it would not be mistaken for usable data.

Reusable script:

- `scripts/collect_stage_c_gap_fill_01_commons_official.py`

Run it later after the Commons rate limit cools down if official reference
diagrams are needed.

