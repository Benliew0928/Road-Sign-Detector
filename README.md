# RoadSign Assist

RoadSign Assist is an offline Malaysian road-sign detection and classification
system. It accepts close-up sign images, road-scene images, recorded video,
laptop cameras and connected phone cameras through one FastAPI and React
website. The runtime combines detector fusion, an 80-class classifier, local
OCR, temporal tracking and multilingual advisory audio.

## Quick start

Run the local website from PowerShell:

```powershell
cd C:\MiniProject
.\scripts\run.ps1
```

To create a temporary public HTTPS link for phone-camera access:

```powershell
cd C:\MiniProject
.\scripts\run.ps1 -Public
```

Run `.\scripts\setup.ps1` once on a new machine. If required model or data
artifacts are missing, restore the tracked DVC artifacts with:

```powershell
.\.venv\Scripts\dvc.exe pull
```

Keep the launcher window open while using the application. Press `Ctrl+C` to
stop the API, website and temporary public tunnel. See the
[website guide](./docs/WEBSITE_GUIDE.md) for image modes, camera pairing,
public access and troubleshooting.

## Image modes

- **Close-up sign** sends an already cropped sign directly to classification.
- **Road scene** first locates signs, crops each detected region and then
  classifies the crops.

These modes measure different parts of the system. A correct classifier cannot
recover a sign that the detector did not localize.

## Active runtime

The default composition is declared in
[`configs/inference/default.yaml`](./configs/inference/default.yaml):

- Detector: Candidate01 and V12 YOLO26s models fused at 640-pixel input.
- Classifier: EfficientNetV2-M V4, 320-pixel input and 80 semantic classes.
- Classifier confidence threshold: 0.72.
- OCR: local PaddleOCR assets for Latin and Chinese text.
- Tracking: BoT-SORT configuration with temporal stability filtering.
- Catalogue: 104 semantic entries with multilingual meanings and guidance.

The default detector requires CUDA. The promoted runtime remains labelled as an
academic candidate in the interface because its datasets and models have
assignment-only or internal-academic usage restrictions. Start the former
runtime immediately with:

```powershell
.\scripts\run.ps1 -Profile legacy
```

## Current measured results

### Classifier

- Human-adjudicated benchmark: **706/710 = 99.44%** top-1 accuracy.
- Historical retained images: **440/443 = 99.32%**.
- Recording sign crops: **266/267 = 99.63%**.
- Previous Candidate03 on the corrected 710-image benchmark:
  **690/710 = 97.18%**.
- Coursework close-up exact semantic accuracy: **69/84 = 82.14%**.
- Coursework close-up stop-equivalent accuracy: **70/84 = 83.33%**.

The 710-image benchmark and 84 coursework images are evaluation data. They are
not part of classifier training.

### Detector and full road-scene pipeline

- Candidate01 original validation mAP@50: **86.56%**.
- Candidate01 original validation mAP@50-95: **66.36%**.
- Readable semantic signs localized at IoU 0.50: **230/267 = 86.14%**.
- Full-scene end-to-end semantic recall: **227/267 = 85.02%**.
- Correct classification after successful localization:
  **227/230 = 98.70%**.
- All annotated boxes localized: **275/351 = 78.35%**.
- Coursework road-scene exact semantic accuracy: **36/84 = 42.86%**.

The detector mAP values summarize areas under precision-recall curves and are
not correct-image counts. The fixed operating-point results use explicit
localized-sign counts. On the final 200-frame integration benchmark, V4 gained
seven correct signs with zero regressions against the preceding runtime.

## V4 classifier dataset

The frozen human-reviewed V4 classifier dataset contains **4,725 images across
80 classes**:

| Split | Images |
| --- | ---: |
| Training | 3,733 |
| Development | 512 |
| Calibration | 34 |
| Historical compatibility test | 446 |

Its composition is 3,219 inherited human-audited examples, 1,437 retained
human-reviewed road-scene crops and 69 owner-accepted licensed additions.
During the main crop review, the team reviewed 1,672 proposed crops from 890
source scenes before eligibility, duplicate and source-group checks.

`roadway_diverges` is the acknowledged weak class. V4 contains only 11 examples
for it: 9 training, 1 development and 1 calibration. The targeted shortage of
21 additional training and 9 development examples remains unresolved; visually
similar `divided_road_begins` signs were not substituted.

## Website functions

- Batch image analysis with per-image close-up or road-scene mode.
- Video analysis with annotated output and sign-moment navigation.
- Laptop-camera and phone-camera live analysis.
- Multi-phone live wall and focused stream view.
- Local OCR, tracking and multilingual advisory audio.
- Recent image results retained in the current browser session.
- Runtime health and active-model information.

The application has no account system or cloud-storage workflow. Phone frames
are processed on the laptop and the live-camera path does not record footage to
disk.

## Verification

Run the backend and frontend checks from the project root:

```powershell
.\.venv\Scripts\pytest.exe
npm test --prefix apps\web
npm run build --prefix apps\web
npm run test:e2e --prefix apps\web
```

Camera permissions, phone pairing and long-duration streaming still require a
physical-device check.

## Evidence and provenance

- Runtime summary: [FINAL_SUBMISSION_RUNTIME.md](./FINAL_SUBMISSION_RUNTIME.md)
- Dataset source manifest:
  [`data/manifests/dataset_sources.json`](./data/manifests/dataset_sources.json)
- Data licence policy:
  [docs/DATA_LICENCE_POLICY.md](./docs/DATA_LICENCE_POLICY.md)
- Source citation ledger:
  [docs/DATA_SOURCE_CITATION_LEDGER.md](./docs/DATA_SOURCE_CITATION_LEDGER.md)
- V4 dataset:
  `outputs/local_recovery_week1/phase7g_human_reviewed_classifier_dataset_20260913_v4`
- Corrected benchmark:
  `outputs/local_recovery_week1/phase7n_human_adjudicated_benchmark_20260913_v1`
- Full-scene integration gate:
  `outputs/local_recovery_week1/phase7o_full_scene_detector_gate_20260913_v1`
- Runtime promotion and rollback record:
  `outputs/local_recovery_week1/phase7p_v4_runtime_promotion_20260913_v1`

## Important boundaries

- Official coursework inputs under `data/official/` are external evaluation
  data and must not be used to train or tune models.
- Source filenames and directory names are never model features.
- Dataset licences and assignment-only restrictions must remain attached to
  every derived artifact.
- The driver-assistance layer is advisory and cannot control vehicle hardware.
- Low-confidence or unsupported observations may be reported as unknown rather
  than forced into one of the 80 classifier classes.

## Archive

Superseded scripts, training runs, models, working datasets, reviews, caches and
documents were moved without deletion to
`_archive/final_submission_cleanup_20260914/payload`. The archive is ignored by
Git. `_archive/final_submission_cleanup_20260914/MOVE_PLAN.json` records the
original and archived paths for restoration.
