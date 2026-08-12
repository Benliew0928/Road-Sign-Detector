# Data source and citation ledger

Updated: 2026-08-12

The authoritative per-sample provenance is
`data/manifests/classifier_release.csv`. Every active classifier row records its
source group/dataset, URL, licence note, mapping evidence, review decision, and
SHA-256. `data/manifests/dataset_sources.json` is the compact source registry.

## Active sources

| Source | Licence/status | Active use |
|---|---|---|
| Extended Malaysian Traffic Sign Dataset (EMTD), DOI `10.5281/zenodo.1217105` | CC BY 4.0 | Prepared detection, segmentation, and classification data |
| Malaysia Road Sign Dataset v1, Roboflow Universe | CC BY 4.0 according to project metadata; preserve attribution | Reviewed classifier crops |
| Malaysia's Road Sign, Dr Samsudin / Roboflow | CC0/public-domain status according to project metadata; recheck before redistribution | Reviewed classifier crops |
| GTSRB | Official site permits research use with citation | Reviewed real-road classifier crops |
| TT100K and mirrors | Academic/non-commercial or CC BY-NC terms vary by mirror | Reviewed classifier crops; internal academic use only pending final terms review |
| Wikimedia Commons references | Per-file licence and attribution | Reviewed road photos and sign references |
| Other public/product/reference pages | Some rows have no stated licence | Internal coursework evidence only; do not publicly redistribute until cleared |
| Official coursework images | Assignment-provided material | External acceptance test only; never training |

## Archive status

Raw downloads, staging data, historical manifests, rejected candidates, and old
review sheets are stored only in
`_archive/2026-08-12-pre-dvc-cleanup/`. The archive is ignored by Git and DVC.
Its `archive_index.csv` records group paths/counts/sizes/tree hashes, and
`archive_files.sha256.csv` records and verifies every moved file.

The shared DVC remote contains only active prepared datasets and selected model
artifacts. It does not contain the local research archive.

## Redistribution boundary

Team access for academic development does not automatically permit public
redistribution. Before publishing a dataset or trained artifact, review all
rows whose `license_notes` contain `license_not_stated`, `academic`,
`non-commercial`, `review`, or unspecified manual-source wording. Preserve the
required EMTD, Roboflow, GTSRB, TT100K, and per-file Commons attribution.
