# Online Gap-Fill Mining Handoff

Generated: 2026-06-29

## Current Goal

Continue filling clean, non-AI data gaps after Stage C Gap Fill 02. Do not redo
classes already filled by Stage 02 unless Stage D QC later rejects them.

Stage 02 successfully filled 10 tracked classes to minimum candidate coverage:

- `bicycle_crossing`
- `no_bicycle`
- `no_heavy_vehicle`
- `roundabout_mandatory`
- `straight_ahead`
- `straight_or_left`
- `straight_or_right`
- `turn_left`
- `turn_right`
- `uneven_road`

Previous five-class TT100K goal:

- `bicycle_crossing`
- `motor_vehicles_only`
- `no_left_or_right_turn`
- `no_motor_vehicles`
- `no_straight_ahead`

## Current Completed Work

TT100K real-road extraction has produced:

- `motor_vehicles_only`: 85 crops
- `no_motor_vehicles`: 47 crops
- `no_straight_ahead`: 5 crops
- `bicycle_crossing`: 0 crops so far
- `no_left_or_right_turn`: 0 crops so far

Stage C Gap Fill 02 then added:

- `bicycle_crossing`: 50 GTSRB real-road crops
- `no_bicycle`: 20 TT100K real-road crops
- `no_heavy_vehicle`: 50 GTSRB real-road crops
- `roundabout_mandatory`: 50 GTSRB real-road crops
- `straight_ahead`: 50 GTSRB real-road crops
- `straight_or_left`: 50 GTSRB real-road crops
- `straight_or_right`: 50 GTSRB real-road crops
- `turn_left`: 50 GTSRB real-road crops
- `turn_right`: 50 GTSRB real-road crops
- `uneven_road`: 50 GTSRB real-road crops

Stage 02 artifacts:

- `docs/STAGE_C_GAP_FILL_02_PUBLIC_REAL_SOURCES_REPORT.md`
- `data/manifests/stage_c_gap_fill_02_public_real_sources_candidates.csv`
- `outputs/review/stage_c_gap_fill_02_public_real_sources/`
- `outputs/review/stage_c_gap_fill_02_public_real_sources/_qa_10class_6each.jpg`

Processed TT100K train shards 0-3. Shard 4 has a partial download:

- `data/raw/online_sources/stage_c_gap_fill_01_tt100k/_hf_arrow_shards/default/train/data-00004-of-00014.arrow.part`

Do not delete that `.part` file; the downloader now resumes it.

## Important Rule

Do not use AI-generated sign images. Accept only:

- real road images from public datasets with exact labels, or
- official/reference diagrams from traceable sources, kept separate from real
  road crops.

## Continue TT100K Mining

Use:

```powershell
cd C:\MiniProject
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -u scripts\collect_stage_c_gap_fill_01_tt100k.py --target-per-class 100
```

Do not pass `--reset` unless you intentionally want to rebuild this sprint.
Do not pass `--delete-shards`; keeping downloaded shards and `.part` files makes
resume much faster.

The extractor writes progress after every completed shard:

- `data/manifests/stage_c_gap_fill_01_tt100k_candidates.csv`
- `outputs/audit/stage_c_gap_fill_01_tt100k_candidates.json`
- `outputs/review/stage_c_gap_fill_01_tt100k/`

## Exact TT100K Mapping

Use only these mappings:

| Project class | TT100K label |
|---|---|
| `bicycle_crossing` | `w56` |
| `motor_vehicles_only` | `i4` |
| `no_left_or_right_turn` | `p20` |
| `no_motor_vehicles` | `p10` |
| `no_straight_ahead` | `p14` |

Do not use `w13` for bicycle crossing; `w13` is crossroads.
Do not use `p6` for no motor vehicles; `p6` is bicycle/motorcycle prohibition.
Do not use `p25` for no left/right turn; `p25` is a car prohibition variant.

## Added Stage 02 Exact Public Mappings

| Project class | Source label |
|---|---|
| `bicycle_crossing` | GTSRB class 29, Bicycles crossing |
| `no_heavy_vehicle` | GTSRB class 16, Vehicles over 3.5 metric tons prohibited |
| `uneven_road` | GTSRB class 22, Bumpy road |
| `turn_left` | GTSRB class 34, Turn left ahead |
| `turn_right` | GTSRB class 33, Turn right ahead |
| `straight_ahead` | GTSRB class 35, Ahead only |
| `straight_or_right` | GTSRB class 36, Go straight or right |
| `straight_or_left` | GTSRB class 37, Go straight or left |
| `roundabout_mandatory` | GTSRB class 40, Roundabout mandatory |
| `no_bicycle` | TT100K label `p6` |

Bad lesson learned:

- Do not take the first 50 sequential GTSRB files. They can be near-duplicate
  frames. The Stage 02 collector now samples across each whole source class.

## Next Best Sequence

1. Continue TT100K from shard 4 until `no_motor_vehicles` reaches at least 50.
2. Check whether later shards contain `p20` and `w56`. If they remain rare,
   stop relying on TT100K alone.
3. Re-run the Commons official-reference collector after rate limiting cools:

```powershell
cd C:\MiniProject
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\collect_stage_c_gap_fill_01_commons_official.py --reset
```

4. Generate/view the contact sheets in `outputs/review/`.
5. Only after visual QC, merge accepted crops into the Stage E frozen split.

## Source Links To Cite

- TT100K Hugging Face mirror:
  <https://huggingface.co/datasets/Genius-Society/tt100k>
- Original TT100K dataset:
  <https://cg.cs.tsinghua.edu.cn/traffic-sign/>
- TT100K class legend used for mapping:
  <https://lijiancheng0614.github.io/2019/04/16/2019_04_16_TT100K/>
- Wikimedia Commons:
  <https://commons.wikimedia.org/>
