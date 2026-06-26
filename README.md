# RoadSign Assist

RoadSign Assist is a greenfield, offline Malaysian road-sign intelligence
system. It combines an explainable color/shape baseline with deep-learning sign
segmentation, semantic classification, multilingual OCR, temporal tracking, and
deterministic ADAS recommendations.

The authoritative implementation tracker is
[GREENFIELD_TECHNICAL_DEVELOPMENT_PLAN.md](./GREENFIELD_TECHNICAL_DEVELOPMENT_PLAN.md).
The verified P0-P14 implementation notes and manual instructions are in
[TECHNICAL_DEVELOPMENT_REPORT_P0_P14.md](./TECHNICAL_DEVELOPMENT_REPORT_P0_P14.md).

## Quick Start

```powershell
.\scripts\setup.ps1
.\scripts\run.ps1
```

The setup script creates a Python 3.11 environment and installs the web
application. The run script starts the FastAPI backend and serves the built web
interface.

To run the current unreviewed YOLO26, embedding-gated EfficientNet, and
PP-OCRv6 experiments:

```powershell
.\scripts\run_experimental.ps1
```

The experimental profile is visibly labelled and is not promoted to the
production model registry.

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
- EfficientNetV2-S test macro-F1: 0.706.
- Embedding-gated EfficientNet selective accuracy: 0.829 at 0.745 coverage.
- Offline PP-OCRv6 synthetic multilingual smoke CER: 0.000.
- 84/84 coursework images produce candidates under two seconds with the
  YOLO26s hybrid profile on the development RTX laptop; draft semantic exact
  match is still only 15.5%.

See [TECHNICAL_DEVELOPMENT_REPORT_P0_P14.md](./TECHNICAL_DEVELOPMENT_REPORT_P0_P14.md)
for commands, evidence, limitations, and manual testing.

## Important Boundaries

- Official coursework inputs live under `data/official/`.
- Large datasets, annotations, models, and generated outputs are DVC-managed.
- Coursework images are external acceptance data and must not be used to train
  or tune models.
- Source filenames and folder names are never model features.
- The ADAS layer is advisory and cannot control real vehicle hardware.
- Unknown or low-confidence signs are reported as unknown rather than forced
  into a supported class.
