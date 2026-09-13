# Final submission runtime

## Start the application

Public launch using the promoted V4 classifier:

```powershell
cd C:\MiniProject
.\scripts\run.ps1 -Public
```

Local launch:

```powershell
.\scripts\run.ps1
```

Immediate legacy rollback:

```powershell
.\scripts\run.ps1 -Profile legacy
```

## Deployed model composition

- Detector: Candidate01 and V12 YOLO26s S640 fusion.
- Classifier: 80-class EfficientNetV2-M V4, 320-pixel input, confidence threshold 0.72.
- OCR: local PaddleOCR assets.
- Semantic catalogue: 104 entries, including `oku_accessibility` and `expressway`.

The active composition is declared in `configs/inference/default.yaml`. The rollback composition is declared in `configs/inference/legacy.yaml`.

## Final measured results

- Human-adjudicated classifier benchmark: V4 706/710 (99.44%); prior Candidate03 690/710 (97.18%).
- Coursework close-up exact semantic accuracy: 69/84 (82.14%).
- Coursework close-up stop-equivalent accuracy: 70/84 (83.33%).
- Full-scene end-to-end semantic recall: V4 227/267 (85.02%); previous runtime 220/267 (82.40%).
- Full-scene accuracy after localization: V4 98.70%; previous runtime 95.65%.
- Full-scene paired result: seven gains and zero regressions.
- Detector semantic-box localization recall at IoU 0.50: 230/267 (86.14%).

## Retained evidence

- Dataset and licences: `data/manifests/dataset_sources.json`, `docs/DATA_LICENCE_POLICY.md`, and `docs/DATA_SOURCE_CITATION_LEDGER.md`.
- Human-reviewed V4 dataset: `outputs/local_recovery_week1/phase7g_human_reviewed_classifier_dataset_20260913_v4`.
- Corrected benchmark: `outputs/local_recovery_week1/phase7n_human_adjudicated_benchmark_20260913_v1`.
- Detector integration gate: `outputs/local_recovery_week1/phase7o_full_scene_detector_gate_20260913_v1`.
- Promotion record and rollback snapshot: `outputs/local_recovery_week1/phase7p_v4_runtime_promotion_20260913_v1`.

## Archive

Superseded scripts, training runs, models, working datasets, reviews, caches, and documents were moved without deletion to `_archive/final_submission_cleanup_20260914/payload`. That directory is ignored by Git. `MOVE_PLAN.json` preserves original and archive paths for restoration.
