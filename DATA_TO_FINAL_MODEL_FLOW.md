# Data to model flow

This project now starts from reviewed, prepared datasets. The 23 GiB research
download archive is local recovery material and is not required for normal
training.

## Active pipeline

1. Teammates put new photos in the shared **Contributor Upload Inbox**, together
   with the contributor batch manifest described in
   `docs/DATA_COLLECTION_PROTOCOL.md`.
2. A maintainer downloads the batch into a new isolated review folder, verifies
   its source/licence fields, deduplicates it by SHA-256, and reviews labels.
3. Accepted images are added to a new versioned prepared dataset. Coursework
   images remain external test material and are never copied into training.
4. The release tool creates deterministic train/validation/test folders and one
   canonical manifest with a `split` column.
5. The prepared datasets are committed through DVC. Git stores only code,
   metadata, and `.dvc` pointer files.
6. Detector training uses `data/processed/emtd_detection` or
   `data/processed/emtd_segmentation`. Classifier training defaults to
   `data/processed/classifier_no_controlled_variants_20260812`.
7. Evaluation happens before any model is promoted. The current runtime
   classifier is explicitly `legacy_synthetic_assisted`; it is not the clean
   release trained in this cleanup.

## Current clean classifier release

- Release: `classifier_no_controlled_variants_20260812`
- Samples: 2,976 unique reviewed crops across 78 labels
- Splits: 2,049 train, 465 validation, 462 test
- Controlled/synthetic-background variants: 0
- Coursework training images: 0
- Status: `coverage_gaps_block_final`

Seven must-have classes are below the unique-sample minimum:

| Class | Clean samples | Gap to 50 |
|---|---:|---:|
| `no_left_or_right_turn` | 7 | 43 |
| `no_straight_or_left` | 9 | 41 |
| `residential_area_ahead` | 49 | 1 |
| `side_road_right` | 49 | 1 |
| `sound_horn` | 11 | 39 |
| `steep_descent` | 10 | 40 |
| `turn_left_or_right` | 11 | 39 |

The first five large gaps result from rejecting 202 controlled visual variants.
The two one-sample gaps result from exact duplicate removal. Collect real,
class-correct replacements before claiming a clean final classifier.

## Training commands

```powershell
.\.venv\Scripts\roadsign-assist.exe train-detector --data data/processed/emtd_detection/data.yaml --model yolo26n.pt --experimental
.\.venv\Scripts\roadsign-assist.exe train-detector --task segment --data data/processed/emtd_segmentation/data.yaml --model yolo26s-seg.pt --experimental
.\.venv\Scripts\roadsign-assist.exe train-classifier --data data/processed/classifier_no_controlled_variants_20260812
```

The classifier release is training-ready, but its coverage status prevents a
clean-final promotion until the gaps above are closed and a new model passes
independent evaluation. The detector commands require `--experimental` because
the EMTD source boxes and SAM masks have not completed manual annotation/mask
review; the flag records that boundary instead of presenting those runs as
approved production training.
