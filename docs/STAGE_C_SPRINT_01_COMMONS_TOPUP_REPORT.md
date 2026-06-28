# Stage C Sprint 01 Commons Top-Up Report

Generated: 2026-06-28

This sprint is a small, controlled Stage C collection pass. It targets only
near-minimum `must` classes instead of attempting broad data collection.

---

## Scope

Targeted classes:

- `no_overtaking`
- `keep_right`
- `no_heavy_vehicle`

Not targeted in this sprint:

- `side_road_right`, because the first public-source probe showed many
  Malaysia/Commonwealth filenames mix side-road, narrow-road, and merge-like
  signs. It needs a more careful local-photo or official-reference pass.

---

## Source

Curated Wikimedia Commons files were downloaded through the Commons API.

Each file keeps its own licence and attribution metadata in:

- `data/manifests/stage_c_sprint_01_commons_candidates.csv`

Raw files are stored under:

- `data/raw/local_collection/stage_c_sprint_01/wikimedia_commons/`

These are candidate files only. They are not part of the final training split
until Stage D QC/annotation and Stage E split freeze are complete.

---

## Result

| Metric | Count |
|---|---:|
| Curated source candidates | 13 |
| Downloaded usable candidates | 8 |
| Downloaded but excluded by visual QC | 1 |
| Failed downloads / rate-limited files | 4 |

Class impact:

| Class | Before sprint | Usable sprint candidates | With sprint | Minimum | Result |
|---|---:|---:|---:|---:|---|
| `no_overtaking` | 47 | 4 | 51 | 50 | Meets minimum pending Stage D QC |
| `keep_right` | 47 | 3 | 50 | 50 | Meets minimum pending Stage D QC |
| `no_heavy_vehicle` | 42 | 1 | 43 | 50 | Still short by 7 |

Overall `must` class count after this sprint:

- Meeting minimum with sprint candidates: 28
- Still below minimum with sprint candidates: 23

Audit files:

- `outputs/audit/stage_c_sprint_01_commons_candidates.json`
- `outputs/audit/post_stage_c_sprint_01_gap_report.csv`
- `outputs/audit/post_stage_c_sprint_01_gap_report.json`

Review sheets:

- `outputs/review/stage_c_sprint_01_commons_candidates/no_overtaking.jpg`
- `outputs/review/stage_c_sprint_01_commons_candidates/keep_right.jpg`
- `outputs/review/stage_c_sprint_01_commons_candidates/no_heavy_vehicle.jpg`
- `outputs/review/stage_c_sprint_01_commons_candidates/_all_candidates.jpg`

---

## QC Notes

- `S01-003` was downloaded but excluded because visual review showed it is
  heavy-vehicle no-overtaking, not generic `no_overtaking`.
- `keep_right` includes two real Malaysian road-scene photos. These are useful
  for realism, but they need bounding boxes before detector training and must be
  kept in a leakage-safe group during split freeze.
- Wikimedia Commons rate-limited several later files. Failed files are recorded
  in the manifest and are not counted.

---

## Next Step

Do not keep retrying random public-source searches for these same near-minimum
classes.

Recommended next Stage C slice:

1. Collect or create 7 more clean `no_heavy_vehicle` candidates.
2. Collect or create 5 more clean `side_road_right` candidates.
3. Prefer local phone photos, official-reference graphics, or very tightly
   curated public files over broad search results.
