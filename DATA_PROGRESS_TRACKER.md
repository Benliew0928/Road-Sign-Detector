# Data progress tracker

Updated: 2026-08-12

The machine-readable tracker is
`data/manifests/CURRENT_DATA_PROGRESS.csv`. The canonical classifier release is
`data/manifests/classifier_release.csv`; its `split` column replaces the old
separate train/validation/test CSV files.

## Quick check

Run this whenever you want the current collection target list without opening
the CSV manually:

```powershell
.\.venv\Scripts\python.exe scripts\report_data_collection_progress.py
```

It shows the clean-release status and the must-have gaps that block a final
classifier claim. Add `--all` to include should/optional future-catalogue
gaps, or use `--priority should` / `--priority optional` to focus one tier.

## Current state

- Active prepared detector datasets: `emtd_detection`, `emtd_segmentation`
- Active prepared classifier datasets: `emtd_classification`,
  `classifier_no_controlled_variants_20260812`
- Clean classifier release: 2,976 samples, 78 labels, zero controlled variants
- External coursework test: 84 images, zero used for training
- Active reviewed decisions: 2,978 accepted, 202 rejected controlled variants
- Clean-final classifier claim: blocked by seven minimum-coverage gaps
- Current app classifier: retained only as `legacy_synthetic_assisted`

## Collection priority

Collect real road-scene or correctly licensed reference images for:

1. `no_left_or_right_turn`: 43
2. `no_straight_or_left`: 41
3. `steep_descent`: 40
4. `sound_horn`: 39
5. `turn_left_or_right`: 39
6. `residential_area_ahead`: 1
7. `side_road_right`: 1

After these gaps, prioritize real Malaysian night, rain, rural, highway,
small-sign, no-sign, and text-heavy/OCR scenes. Use independent review for
ontology mappings and segmentation masks before final claims.

## Daily rule

Do not edit a prepared dataset in place. Create a contributor batch, review it,
produce a new versioned release, run its audit, then `dvc add`, commit, and push.
The ignored local archive is recovery evidence, not an active dataset.
