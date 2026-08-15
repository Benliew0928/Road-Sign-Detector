# RoadSign Assist

RoadSign Assist is a greenfield, offline Malaysian road-sign intelligence
system. It combines an explainable color/shape baseline with deep-learning sign
segmentation, semantic classification, multilingual OCR, temporal tracking, and
deterministic ADAS recommendations.

The authoritative implementation tracker is
[GREENFIELD_TECHNICAL_DEVELOPMENT_PLAN.md](./GREENFIELD_TECHNICAL_DEVELOPMENT_PLAN.md).
The clean data collection and cleaning tracker is
[DATA_PROGRESS_TRACKER.md](./DATA_PROGRESS_TRACKER.md).
The verified P0-P14 implementation notes and manual instructions are in
[TECHNICAL_DEVELOPMENT_REPORT_P0_P14.md](./TECHNICAL_DEVELOPMENT_REPORT_P0_P14.md).

## Quick Start

```powershell
.\scripts\setup.ps1
.\scripts\run.ps1
```

Setup is needed once. After that, `.\scripts\run.ps1` is the single command:
it builds the current frontend, starts the complete HTTPS-capable FastAPI
runtime, and opens the dashboard. Image, video, laptop camera, phone camera,
multi-phone live wall, OCR, tracking, and offline advisory audio are all in the
same site. See the [website guide](./docs/WEBSITE_GUIDE.md) for operation,
public-tunnel options, and troubleshooting.

## Data and model artifacts

The active project is tracked with targeted DVC pointers, including prepared
datasets, selected model/OCR artifacts, required YOLO base weights, and the
offline multilingual audio bundle. After the shared Google Drive remote is
configured, restore artifacts with:

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\dvc.exe pull
```

The clean classifier dataset is
`data/processed/classifier_no_controlled_variants_20260812` (2,976 samples,
zero controlled variants). The selected website candidate achieved 94.59%
locked-test accuracy and remains labelled `coverage_gaps_block_final`. See
[the model-run inventory](./docs/MODEL_RUN_INVENTORY.md) for every completed
candidate and artifact location,
[DATA_PROGRESS_TRACKER.md](./DATA_PROGRESS_TRACKER.md)
for its coverage gaps and [docs/DVC_COLLABORATION.md](./docs/DVC_COLLABORATION.md)
for the maintainer workflow. New contributors should follow the
[step-by-step DVC getting-started guide](./docs/TEAMMATE_DVC_GETTING_STARTED.md).

Phone-camera streaming is available from the same dashboard. Scan its QR code
on local Wi-Fi/hotspot, or start the same launcher with `-Public` when the
network blocks peer-to-peer access.

## Development

```powershell
.\.venv\Scripts\roadsign-assist.exe doctor
.\.venv\Scripts\pytest.exe
.\.venv\Scripts\roadsign-assist.exe verify-ocr-assets
.\.venv\Scripts\roadsign-assist.exe compare-classifiers
.\.venv\Scripts\roadsign-assist.exe verify-reset
.\.venv\Scripts\roadsign-assist.exe baseline-benchmark --experimental

cd apps\web
npm install
npm run dev
```

## Current Measured State

- 510 EMTD images and 1,227 source boxes, DVC-backed.
- 1,220 SAM 2.1 draft masks on 507 images.
- YOLO26s-seg test mask mAP50: 0.598 and mask recall: 0.573.
- Normalized-area small-sign recall: 0.674 at 640 px and 0.687 at 960 px;
  640 px remains the live default because the 960 ONNX CUDA path was much
  slower.
- YOLO26s-seg ONNX CPU wall latency: 522 ms mean, 1,041 ms p95 on 63
  development-laptop test images.
- Classical comparison: six SVM/Random-Forest feature experiments; best
  macro-F1 0.570.
- Clean EfficientNetV2-S locked-test accuracy: 0.946 (437/462; 95% CI
  0.921-0.963) and macro-F1: 0.929 across all 78 labels.
- Clean EfficientNetV2-S external assignment accuracy: 0.702 on all 84 images.
- Offline PP-OCRv6 synthetic multilingual smoke CER: 0.000.
- 84/84 coursework images produce candidates under two seconds with the
  YOLO26s hybrid profile on the development RTX laptop; draft semantic exact
  match is still only 15.5%.

See [TECHNICAL_DEVELOPMENT_REPORT_P0_P14.md](./TECHNICAL_DEVELOPMENT_REPORT_P0_P14.md)
for commands, evidence, limitations, and manual testing.

## Important Boundaries

- Official coursework inputs live under `data/official/`.
- Large datasets, annotations, models, and generated audio are DVC-managed.
- Coursework images are external acceptance data and must not be used to train
  or tune models.
- Source filenames and folder names are never model features.
- The ADAS layer is advisory and cannot control real vehicle hardware.
- Unknown or low-confidence signs are reported as unknown rather than forced
  into a supported class.
