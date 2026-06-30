# RoadSign Assist Technical Development Report

**Project:** Greenfield Malaysian Road-Sign Intelligence System  
**Implementation period:** 24-25 June 2026  
**Scope:** Technical phases P0 through P14  
**Tracker:** [GREENFIELD_TECHNICAL_DEVELOPMENT_PLAN.md](./GREENFIELD_TECHNICAL_DEVELOPMENT_PLAN.md)

## 1. Executive Summary

The legacy color-label experiment was replaced with a clean Python 3.11,
PyTorch, ONNX Runtime, PaddleOCR, FastAPI, React, TypeScript, Git, and DVC
project.

The repository now contains a complete P0-P14 software platform and a working
experimental AI pipeline:

- Classical red, blue, and yellow sign segmentation.
- Frozen-split SVM and Random Forest comparison across HOG, HSV, geometry,
  and Hu-moment feature sets.
- SAM 2.1 box-prompted draft mask generation with geometry QA.
- YOLO26n and stronger YOLO26s instance segmentation experiments trained on
  Malaysian road scenes.
- MobileNetV3 and EfficientNetV2 semantic classifier experiments.
- Frozen offline PP-OCRv6 models for Malay/English, Chinese, and numbers.
- Motion-aware global track association, temporal fusion, and conditional OCR.
- Deterministic advisory ADAS recommendations with confidence, unit, range,
  and unknown-sign safety gates.
- FastAPI image, batch, video, health, catalogue, and WebSocket APIs.
- React live dashboard for camera, image, batch, and video workflows.
- A deep-first hybrid detector that uses the classical detector only when the
  deep segmenter finds no sign.

The project is intentionally not called production-ready. The Malaysian class
mapping, SAM masks, and coursework semantic mapping have only one review. The
measured AI accuracy is below the final targets, and the 84 coursework images
have a substantial domain and taxonomy difference from EMTD.

## 2. What The System Can Do

### Default Safe Profile

`scripts/run.ps1` starts the application without experimental models being
presented as approved. Missing production models cause a safe baseline and
`unknown_sign` fallback.

### Experimental Deep Profile

`scripts/run_experimental.ps1` starts:

```text
YOLO26s-seg ONNX
  -> classical fallback on empty deep frames
  -> EfficientNetV2-S ONNX classifier
  -> conditional offline PP-OCRv6
  -> semantic catalogue and ADAS rules
  -> FastAPI/WebSocket
  -> React dashboard
```

The interface visibly warns:

```text
Experimental unreviewed models are active; results are not production claims.
```

### Demonstrated End-To-End Example

A held-out EMTD test image was processed through the live HTTP application:

```text
Detected sign: maximum_speed
Segmentation polygon: 7 points
OCR text: 30
Numeric value: 30
ADAS action: SET_TARGET_SPEED
Advisory only: true
End-to-end latency after server restart: 1,147 ms
```

Evidence:

```text
outputs/evaluation/live_api_s30_speed30.json
```

## 3. Implemented Architecture

```text
Camera / image / video / batch
                 |
                 v
       YOLO26 instance segmenter
                 |
       no detections? -> color/shape fallback
                 |
                 v
       Rectified sign crop
                 |
          +------+------+
          |             |
          v             v
 EfficientNetV2-S   Conditional PP-OCRv6
          |             |
          +------+------+
                 |
                 v
      Confidence and temporal fusion
                 |
                 v
      Malaysian catalogue and ADAS rules
                 |
       +---------+----------+
       |                    |
       v                    v
 FastAPI/WebSocket     React dashboard
```

OCR is not executed for every contour. It runs for parameterized/text signs,
or for the highest-confidence unknown candidate. Camera/video OCR is cached by
stable track. Still-image and batch detections are treated as stable
immediately.

Track assignment now uses a configurable BoT-SORT/GMC adapter when available,
with a sparse optical-flow IoU tracker fallback. It maps custom detector outputs
to stable session-local track state, compensates camera translation in video,
rebinds short occlusions, expires stale tracks, fuses semantic scores, and
applies per-track announcement cooldowns. Synthetic stationary, moving-camera,
occlusion, blur, and cooldown checks pass; owner real-footage testing remains
recommended for final demonstration confidence.

## 4. Data And Provenance

### Reset And Official Backup

The external backup and restored official tree were re-audited:

| Item | Result |
|---|---:|
| Backup/restored images | 84 / 84 |
| Backup/restored coursework documents | 24 / 24 |
| Checksum mismatches | 0 |
| Unreadable images/documents | 0 |

Evidence: `outputs/evaluation/reset_audit.json`.

### Accepted Public Source

The Extended Malaysian Traffic Sign Dataset was imported from Zenodo under
CC BY 4.0.

Current balanced subset:

| Item | Result |
|---|---:|
| Source classes | 66 |
| Downloaded images | 510 |
| Raw size | 2.35 GiB |
| Source boxes | 1,227 |
| SHA-256 mismatches | 0 |
| Perceptual duplicate groups | 507 |
| Semantic crop labels | 35 |
| Semantic crops | 1,064 |

Relevant files:

```text
docs/DATA_LICENCE_POLICY.md
docs/DATA_COLLECTION_PROTOCOL.md
data/manifests/dataset_sources.json
data/raw/emtd/metadata/subset_manifest.csv
configs/catalogue/emtd_class_mapping.v0.1.json
```

### Split Construction

The split builder is deterministic, duplicate-group safe, and
class-stratified where group constraints allow.

```text
Images: 382 train / 64 validation / 64 test
Classifier crops: 670 train / 190 validation / 204 test
```

All 35 semantic labels occur in training. `slippery_road` has only two mapped
instances and is absent from validation and test; this limitation is recorded
rather than hidden.

### DVC

The following are DVC-backed and pushed to the configured local remote:

```text
data/raw
data/annotations
data/processed
models
```

The data/model release transferred 2,699 content-addressed files, followed by
the regenerated split artifacts and nine later model/pipeline artifacts.
`dvc repro` and `dvc status -c` confirm that the current local cache and
configured remote are synchronized.

## 5. Annotation And Segmentation

SAM 2.1-small was frozen under `models/pretrained` and used only as a
semi-automatic annotation tool.

| Item | Result |
|---|---:|
| Images attempted | 510 |
| Images accepted by geometry QA | 507 |
| Images rejected | 3 |
| Accepted draft masks | 1,220 |

QA checks include:

- Polygon point count.
- Finite coordinates.
- Image bounds.
- Polygon-to-source-box area ratio.
- Polygon bounding-box IoU.
- Mask count versus source box count.

Visual evidence:

```text
outputs/review/emtd_masks_accepted.jpg
outputs/review/emtd_masks_failed.jpg
data/processed/emtd_segmentation/mask_qa.csv
```

The masks remain `sam2_box_prompt_unreviewed`; production training is blocked
unless review metadata is approved.

## 6. YOLO26 Segmentation Results

Selected experimental run:

```text
Model: YOLO26s-seg
Input: 640 x 640
Epochs: 30
Training images: 380 accepted-mask images
Validation images: 64
Test images: 63
```

Held-out test metrics:

| Metric | Result |
|---|---:|
| Box precision | 0.810 |
| Box recall | 0.681 |
| Box mAP50 | 0.740 |
| Box mAP50-95 | 0.541 |
| Mask precision | 0.771 |
| Mask recall | 0.573 |
| Mask mAP50 | 0.598 |
| Mask mAP50-95 | 0.329 |
| PyTorch inference | 26.4 ms/image |
| ONNX CUDA inference | 80.2 ms/image |

ONNX metric parity passed. The largest shared metric difference was
`0.002454`.

Artifacts:

```text
models/exported/experimental/emtd_segmenter_s30.pt
models/exported/experimental/emtd_segmenter_s30.onnx
outputs/evaluation/emtd_segmenter_s30/metrics.json
```

The recall target is not met, so this model is not promoted to the production
registry.

Compared with YOLO26n, YOLO26s increased test box recall from 0.461 to 0.681,
mask recall from 0.440 to 0.573, and mask mAP50 from 0.487 to 0.598. The
stronger model therefore replaced YOLO26n as the experimental default, while
YOLO26n remains available as a lower-power fallback.

### Threshold And CPU Evaluation

Five confidence thresholds were evaluated on the validation split. Confidence
0.10 maximized the recorded mask F1, but the live profile remains at 0.25
because a reviewed no-sign set is not yet available to quantify false
positives.

The ONNX segmenter processed all 63 frozen test images using
`CPUExecutionProvider`:

| Metric | Result |
|---|---:|
| Mean wall latency | 522 ms |
| p95 wall latency | 1,041 ms |
| Maximum wall latency | 1,285 ms |
| Mean model inference | 202 ms |

Evidence:

```text
outputs/evaluation/emtd_segmenter_s30/threshold_tuning.json
outputs/evaluation/emtd_segmenter_s30/cpu_runtime.json
```

This supports the two-second target on the development laptop. The actual lab
CPU still requires confirmation.

### High-Resolution Small-Sign Experiment

A second YOLO26s-seg model was trained for 20 epochs at 960 px and evaluated
against the same frozen 63-image test split. Small signs were defined before
evaluation as ground-truth boxes occupying at most 1% of the source image.

| Model | Small signs matched | Small-sign recall | Mask mAP50 | CPU p95 | ONNX CUDA inference |
|---|---:|---:|---:|---:|---:|
| YOLO26s 640 | 153/227 | 0.674 | 0.598 | 1,041 ms | 80.2 ms |
| YOLO26s 960 | 156/227 | 0.687 | 0.708 | 853 ms | 1,906 ms |

The 960 model gained only 1.3 percentage points of small-sign recall. Its
exported CUDA path was approximately 24 times slower in this measured run, so
it is retained as a research artifact rather than selected for the live
application. The 640 model remains the best measured balance.

Evidence:

```text
outputs/evaluation/emtd_segmenter_s30/recall_slices.json
outputs/evaluation/emtd_segmenter_s960_20/metrics.json
outputs/evaluation/emtd_segmenter_s960_20/cpu_runtime.json
outputs/evaluation/emtd_segmenter_s960_20/recall_slices.json
```

### Classical SVM And Random Forest Comparison

The classical baseline was trained on the same frozen crop splits using only
image pixels. Six classifier/feature combinations were evaluated:

| Classifier | Feature set | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Random Forest | HOG + HSV + shape/Hu | 0.632 | 0.570 |
| Random Forest | HOG | 0.652 | 0.568 |
| Random Forest | HOG + HSV | 0.647 | 0.568 |
| SVM | HOG | 0.593 | 0.406 |
| SVM | HOG + HSV | 0.588 | 0.374 |
| SVM | HOG + HSV + shape/Hu | 0.588 | 0.374 |

Evidence:

```text
outputs/evaluation/baseline_classifiers/comparison.csv
models/baseline
```

## 7. Semantic Classifier Results

Both required CNN approaches were trained with identical grouped splits,
balanced sampling, realistic augmentation, AdamW, cosine scheduling, label
smoothing, temperature calibration, and held-out testing.

| Model | Test accuracy | Test macro-F1 | ECE | ONNX parity |
|---|---:|---:|---:|---:|
| MobileNetV3-Large | 0.730 | 0.675 | 0.152 | Pass |
| EfficientNetV2-S | 0.711 | 0.706 | 0.094 | Pass |

EfficientNetV2-S is the provisional selection because macro-F1 and calibration
are more important than raw accuracy for imbalanced safety classes.

At the provisional `0.72` confidence threshold:

```text
Selective coverage: 72.5%
Accuracy among accepted signs: 81.8%
Accepted-and-correct rate over all test crops: 59.3%
```

ONNX preserves 100% top-1 and known/unknown decisions.

The live experimental classifier was then upgraded to export two ONNX outputs:
the usual logits and a normalized 1,280-dimensional EfficientNet embedding.
Training-split embeddings form one prototype per semantic class. Validation
data tunes a cosine-distance gate, and runtime prediction now rejects a crop if
either confidence is too low, embedding distance is too high, or the nearest
prototype disagrees with the predicted class.

Embedding-gated classifier result:

| Item | Result |
|---|---:|
| Distance threshold | 0.3775 |
| Validation coverage | 78.9% |
| Validation accepted accuracy | 86.0% |
| Test coverage | 74.5% |
| Test accepted accuracy | 82.9% |
| Test confidence rejections | 51 |
| Test distance rejections | 25 |
| Test prototype disagreements | 9 |

The embedding ONNX parity gate passed with maximum logit drift 0.0138 and
maximum embedding drift 0.00085. The FastAPI live profile now loads:

```text
models/exported/experimental/emtd_classifier_efficientnet15_embedding.onnx
models/exported/experimental/emtd_classifier_efficientnet15_embedding.calibration.json
```

Runtime event evidence includes entries such as:

```text
classifier_raw:crossroads:0.865
embedding:crossroads:0.055
classifier_rejection:confidence
```

Critical-class recall is uneven and based on small supports: stop,
children-crossing, and weight-restriction are 100%; maximum-speed is 78.6%;
height-restriction is 53.6%; give-way is 44.4%; no-entry and
width-restriction are 0%. This is below the 90% safety target and is another
reason the classifier remains experimental.

The observed critical-class macro and micro recall are both approximately
59.6%. Evidence:

```text
outputs/evaluation/classifier_comparison/critical_class_recall.json
```

Artifacts:

```text
outputs/evaluation/classifier_comparison/comparison.csv
outputs/evaluation/classifier_comparison/embedding_rejection.json
outputs/training/emtd_classifier_efficientnet15/metrics.json
models/exported/experimental/emtd_classifier_efficientnet15.onnx
models/exported/experimental/emtd_classifier_efficientnet15_embedding.onnx
models/exported/experimental/emtd_classifier_efficientnet15.labels.json
models/exported/experimental/emtd_classifier_efficientnet15_embedding.calibration.json
```

The 85% macro-F1 target is not met. Unknown AUROC also remains unmeasured
because no independently reviewed out-of-distribution set exists.

## 8. Offline Multilingual OCR

Official PP-OCRv6-small detector and recognizer assets are stored locally with
file size and SHA-256 verification:

```text
models/ocr/PP-OCRv6_small_det
models/ocr/PP-OCRv6_small_rec
models/ocr/manifest.json
```

Paddle's Windows oneDNN path failed on a PP-OCRv6 attribute, so the engine now
uses the supported non-MKLDNN CPU path.

Synthetic pipeline smoke result:

| Item | Result |
|---|---:|
| Malay/English/Chinese/numeric samples | 5 |
| Exact matches | 5/5 |
| Mean CER | 0.000 |
| Cold model initialization | 6.43 s |
| Warm OCR mean | 189 ms |

The API preloads OCR during startup, so the cold cost is not paid on the first
stable sign.

This smoke result proves local multilingual execution and parsing. It does not
claim real-road OCR accuracy.

## 9. Coursework Acceptance Run

All 84 official coursework images remain excluded from training and model
selection.

The final experimental hybrid run produced:

| Item | Result |
|---|---:|
| Images completed | 84/84 |
| Images with one or more candidates | 84/84 |
| Under two seconds | 84/84 |
| Mean runtime | 787 ms |
| p95 runtime | 1,222 ms |
| Maximum runtime | 1,413 ms |
| Draft semantic images scored | 58 |
| Draft semantic exact matches | 9 |
| Draft semantic exact-match rate | 15.5% |

Evidence:

```text
outputs/evaluation/coursework_experimental_s30/summary.json
outputs/evaluation/coursework_experimental_s30/results.csv
outputs/evaluation/coursework_experimental_s30/annotated
```

The runtime and segmentation-candidate coverage are strong on this RTX 4050
laptop. Semantic accuracy is weak because the EMTD-trained 35-label taxonomy
does not cover many coursework meanings and because coursework images differ
substantially from road scenes. A lab-machine CPU benchmark is still required.

## 10. React And FastAPI Application

Implemented API:

```text
GET  /api/v1/health
GET  /api/v1/catalogue
GET  /api/v1/models
POST /api/v1/infer/image
POST /api/v1/infer/batch
POST /api/v1/infer/video
WS   /api/v1/ws/camera/{session_id}
```

React workflows:

- Laptop camera with reconnecting WebSocket.
- One-image inference.
- Up to 100 image batch inference.
- Video upload and sampled processing.
- Mask polygons and boxes.
- Track IDs and confidence.
- Current meaning, OCR, and ADAS action.
- English, Bahasa Melayu, and Chinese UI selection.
- Mute state.
- Presenter mode.
- Desktop and mobile layouts.

The API preloads detector, classifier, and OCR sessions at startup. Health
output distinguishes:

- Environment health.
- Production model presence.
- Experimental artifact count.
- Deep backend availability.
- Backend loaded state.

Chrome verification confirmed deep mode, the experimental warning, presenter
mode, language selection, mute state, and zero console warnings/errors.

## 11. Bugs Found And Corrected

1. **Experimental classifier files used production names**  
   Experimental exports now stay under `models/exported/experimental`.

2. **No detector ONNX parity gate**  
   Test evaluation, ONNX export, and shared-metric parity are mandatory.

3. **Classifier parity gate used only maximum logit drift**  
   The gate now checks calibrated probability drift, top-1 agreement, and
   known/unknown acceptance agreement.

4. **Classifier runtime assumed 224 px and ignored calibration**  
   Input size and calibration paths are explicit configuration fields.

5. **Raw EMTD images were hardlinked into processed data**  
   Ultralytics repaired four JPEG endings and changed raw files through the
   hardlinks. All four raw members were restored from Zenodo, all 510 hashes
   revalidated, and processed images now use independent copies.

6. **Paddle oneDNN runtime failure on Windows**  
   OCR now disables MKLDNN and passes offline inference.

7. **First live request loaded every model lazily**  
   Detector, classifier, and OCR are warmed during FastAPI startup.

8. **Still images never became stable enough for OCR**  
   Image and batch workflows now treat detections as stable immediately.

9. **OCR ran on every fallback contour**  
   OCR is conditional, and unknown fallback OCR is limited to the best
   candidate.

10. **Classical fallback produced too many events**  
    Emergency fallback output is capped at three highest-confidence
    detections.

11. **Hash-only splits had uneven class coverage**  
    Splits are now deterministic, group-safe, and label-stratified.

12. **Zenodo returned HTTP 429 during parallel download**  
    The downloader now implements server-aware exponential backoff and
    resumable caching.

13. **Tracking tests covered only one stationary sign**  
    Association now uses global assignment and motion prediction. Tests cover
    multiple signs, short occlusion, reappearance, expiration, and movement.

14. **Unvalidated OCR values could reach parameterized actions**  
    Speed, height, width, and weight actions now require sufficient OCR
    confidence, compatible units, and safe numeric ranges. Invalid values
    degrade to `UNKNOWN_CAUTION`.

15. **Ultralytics attempted package installation during offline inference**  
    The CLI sets `YOLO_AUTOINSTALL=false` before importing model code.

16. **The initial dashboard status contradicted backend health**  
    Before the first frame, the footer said `Classical baseline` even when the
    deep pipeline was loaded. It now uses backend model mode until a frame
    result is available.

17. **ONNX Runtime CUDA provider could silently fall back on Windows**  
    Import order could leave ONNX Runtime unable to find CUDA 13 DLLs. The
    classifier backend now imports PyTorch before ONNX Runtime session creation
    and exposes the actual provider list through health output.

18. **Unknown rejection used confidence only**  
    The selected classifier now combines temperature-scaled confidence with
    EfficientNet embedding-distance rejection and prototype label agreement.
    Tests cover distance rejection, prototype disagreement, and compatibility
    with older logits-only ONNX models.

19. **The dashboard loading fallback could still imply baseline mode**  
    Before health resolved, an unknown pipeline mode could fall through to the
    `Classical baseline` label. The footer now has an explicit `Checking
    pipeline` state, and the live metrics show backend-reported detector device
    and classifier providers once health arrives.

20. **Health response shape drifted from the React type**  
    The frontend expected `diagnostics.healthy`, but the health endpoint returned
    only raw dataclass fields. The API now uses explicit Pydantic response
    models for health, model status, batch inference, and video summaries, and
    contract tests assert the runtime fields used by React.

21. **Catalogue was strict internally but untyped in OpenAPI**  
    `/api/v1/catalogue` returned a validated `SignCatalogue`, but FastAPI had no
    response model for the route. The endpoint now exposes `SignCatalogue` in
    OpenAPI, and `scripts/export_openapi.py` writes the reproducible schema to
    `outputs/api/openapi.json`.

22. **Camera message handling was too trusting**  
    The backend could return WebSocket error payloads, but the React camera hook
    parsed messages inline and assumed valid JSON. Camera messages now pass
    through a tested parser, malformed responses surface as recoverable UI
    errors, and the API test suite confirms that an invalid camera frame does
    not prevent the next valid frame from succeeding.

23. **Upload boundary behavior needed direct regression coverage**  
    The API had image-size, batch-count, and video cleanup protections, but they
    were not directly tested. The API suite now verifies oversized image
    rejection, 101-image batch rejection, and cleanup of temporary files after an
    invalid video upload.

24. **E2E coverage focused too much on happy paths**  
    Playwright now covers backend-offline behavior, refresh recovery, deep
    runtime diagnostics, and classifier provider display on both desktop and
    mobile, in addition to the original presenter, batch, and video workflows.

## 12. Verification Results

| Check | Result |
|---|---|
| Ruff | Pass |
| Pyright strict | 0 errors, 0 warnings |
| Python unit/integration/API | 70 passed |
| Frontend ESLint | Pass |
| Frontend Vitest | 1 passed |
| Frontend production build | Pass |
| Playwright desktop/mobile | 10 passed |
| npm audit | 0 vulnerabilities |
| Raw EMTD checksum audit | 510/510 pass |
| Detector ONNX parity | Pass |
| MobileNet ONNX parity | Pass |
| EfficientNet embedding ONNX parity | Pass |
| OCR asset verification | Pass |
| Reset backup checksum audit | 84 images and 24 documents; pass |
| Classical SVM/RF experiments | 6 runs completed |
| Detector CPU benchmark | YOLO26s: 522 ms mean, 1,041 ms p95 |
| DVC push | 3 new model cache files pushed; remote synchronized |
| DVC deterministic split rerun | Pass; no changes |
| Chrome live UI QA | Pass |
| Live experimental health | Detector `cuda:0`, classifier `CUDAExecutionProvider`, OCR loaded |
| Live image API smoke | Pass; evidence includes embedding distance and confidence rejection traces |
| In-app browser live UI QA | Pass; shows Semantic AI Pipeline, `cuda:0`, classifier CUDA/CPU providers, no overflow, and no console warnings |
| API response-model contract | Pass; health includes `healthy`, detector device, classifier providers, and OCR status |
| OpenAPI contract | Pass; six HTTP paths exported with typed response schemas |
| Camera WebSocket recovery | Pass; bad frame returns an error and the next valid frame succeeds |
| Upload boundary tests | Pass; image 413, batch 400, and invalid-video cleanup are covered |

## 13. Manual Setup

Open PowerShell:

```powershell
cd C:\MiniProject
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
```

No Codex runtime path or custom package folder is required.

## 14. Start The Application

### Safe Default

```powershell
cd C:\MiniProject
.\scripts\run.ps1
```

### Experimental Deep Models

```powershell
cd C:\MiniProject
.\scripts\run_experimental.ps1
```

Open:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

Model initialization can take several seconds. Wait for `System ready`.

## 15. Manual Test Flow

### Image

1. Select **Image**.
2. Choose a Malaysian road-sign image.
3. Inspect masks, boxes, semantic meaning, OCR, numeric value, action, and
   latency.

### Camera

1. Select **Camera**.
2. Start the camera and allow browser permission.
3. Hold a printed or physical sign in view for several frames.
4. Observe stable track IDs and conditional OCR.

### Batch

1. Select **Batch**.
2. Choose several images.
3. Inspect the gallery and result table.

### Video

1. Select **Video**.
2. Choose a supported video.
3. Inspect frames read, sampled frames, and events.

### Presenter Mode

Use the expand button to hide side panels and maximize the live view.

## 16. Reproduce Key Evaluations

```powershell
cd C:\MiniProject

.\.venv\Scripts\roadsign-assist.exe verify-ocr-assets
.\.venv\Scripts\roadsign-assist.exe evaluate-ocr-smoke
.\.venv\Scripts\roadsign-assist.exe compare-classifiers
.\.venv\Scripts\roadsign-assist.exe verify-reset
.\.venv\Scripts\roadsign-assist.exe baseline-benchmark --experimental
.\.venv\Scripts\python.exe scripts\export_openapi.py

.\.venv\Scripts\roadsign-assist.exe finalize-classifier-embeddings `
  --checkpoint models\checkpoints\emtd_classifier_efficientnet15\best.pt `
  --data data\processed\emtd_classification `
  --model-output models\exported\experimental\emtd_classifier_efficientnet15_embedding.onnx `
  --calibration-output models\exported\experimental\emtd_classifier_efficientnet15_embedding.calibration.json `
  --report-output outputs\evaluation\classifier_comparison\embedding_rejection.json `
  --device 0 --batch 64 --workers 0 --retention-quantile 0.95

.\.venv\Scripts\roadsign-assist.exe benchmark-detector `
  --model models\exported\experimental\emtd_segmenter_s30.onnx `
  --output outputs\evaluation\emtd_segmenter_s30\cpu_runtime.json `
  --device cpu

.\.venv\Scripts\roadsign-assist.exe evaluate-detector-slices `
  --model models\exported\experimental\emtd_segmenter_s30.pt `
  --output outputs\evaluation\emtd_segmenter_s30\recall_slices.json `
  --imgsz 640 --device 0 --confidence 0.25

.\.venv\Scripts\roadsign-assist.exe evaluate-coursework `
  --config configs\inference\experimental.yaml `
  --output outputs\evaluation\coursework_experimental_s30
```

Full quality checks:

```powershell
.\.venv\Scripts\ruff.exe check src apps\api tests scripts
.\.venv\Scripts\pyright.exe src apps\api tests
.\.venv\Scripts\pytest.exe tests\unit tests\integration apps\api\tests -q

cd apps\web
npm run lint
npm run test
npm run build
npm run test:e2e -- --workers=1 --reporter=line
npm audit --audit-level=moderate
```

## 17. Remaining Work

The software platform through P14 is implemented. The remaining scientific and
approval work is:

1. Obtain independent review of the ontology and EMTD class mapping.
2. Review/correct all SAM mask drafts in CVAT.
3. Collect real Malaysian night, rain, rural, highway, text-heavy, and
   no-sign scenes.
4. Add data for coursework-only semantic classes.
5. Improve small-sign data, augmentation, and architecture; add difficult
   negative scenes.
6. Raise detector recall toward 90%.
7. Raise classifier macro-F1 toward 85%.
8. Build reviewed real-road Malay/English/Chinese/numeric OCR test sets.
9. Build an OOD set and measure unknown AUROC for the confidence-plus-embedding gate.
10. Repeat the under-two-second test on the actual lab CPU.

## 18. Phase Completion Audit

| Phase | Engineering state | Remaining gate |
|---|---|---|
| P0 | Verified | None |
| P1 | Implemented | Owner-controlled first Git commit and second-machine setup |
| P2 | Draft catalogue implemented | Independent two-person safety review and local standards archive |
| P3 | Draft 84-image mapping implemented | Second reviewer and reviewed masks/boxes |
| P4 | Public-data pipeline implemented | Local field and no-sign collection |
| P5 | Draft annotation pipeline implemented | Human acceptance, OCR transcripts, and agreement audit |
| P6 | Reproducible grouped splits implemented; adjacent-session leakage regression test passes | Real captured video-session review and reviewed OCR release |
| P7 | Verified | None for the technical baseline |
| P8 | Experimental models and small-sign comparison implemented | Recall/mask targets, negatives, and human mask review |
| P9 | Experimental CNNs and embedding-gated unknown rejection implemented | Macro-F1/ECE/critical-recall targets and reviewed OOD AUROC |
| P10 | Offline OCR implemented | Reviewed real-road multilingual evaluation |
| P11 | Verified with configurable BoT-SORT/GMC adapter, sparse optical-flow fallback, and synthetic motion/blur report | Owner real-footage demonstration test |
| P12 | Deterministic safety rules implemented | Independent safety review |
| P13 | Verified | None |
| P14 | Verified | None |

The incomplete gates above cannot be honestly converted into completed checks
by writing more application code alone. They require additional data, human
review, the lab machine, or measured model improvement.

## 19. Honest Completion Statement

P0-P14 engineering infrastructure is developed and verified: clean repository,
data provenance, DVC, classical segmentation, deep segmentation,
classification, offline OCR, tracking, ADAS rules, API, and React live
application.

The project now demonstrates genuine end-to-end semantic recognition, including
semantic classifier evidence, OCR-conditioned parameters, and deterministic ADAS
actions. The live experimental profile also reports actual CUDA provider use and
returns explicit embedding/rejection evidence for classifier decisions.

The models remain experimental. Measured detector recall, classifier macro-F1,
and coursework semantic transfer are below the plan targets. Human annotation
review, reviewed OOD data, broader Malaysian data, and lab-machine confirmation
are required before any production or safety claim.
