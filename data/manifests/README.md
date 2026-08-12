# Active manifests

Only current, decision-making metadata lives here:

- `classifier_release.csv`: canonical 2,976-sample classifier release; includes
  the train/validation/test `split` column and per-sample provenance.
- `assignment_external_test.csv`: coursework-only external acceptance data.
- `CURRENT_DATA_PROGRESS.csv`: collection gaps and next actions.
- `dataset_sources.json`: compact source/licence registry.

Reviewed classifier decisions live in `data/annotations/`, which is DVC-managed.
Detector metadata lives inside each prepared detector dataset. Historical and
pending manifests are in the ignored local archive and must not be used for
training.
