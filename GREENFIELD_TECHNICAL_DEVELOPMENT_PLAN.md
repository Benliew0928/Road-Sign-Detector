# Greenfield Malaysian Road-Sign Intelligence System

## Document Purpose

This document is the technical implementation plan and progress tracker for a
complete restart of the Malaysian road-sign recognition project.

It covers:

- Project reset and clean environment setup.
- Classical color and shape segmentation.
- Deep-learning sign segmentation and detection.
- Exact semantic road-sign recognition.
- Bahasa Melayu, English, Chinese, and numeric OCR.
- Sign tracking and confidence stabilization.
- ADAS-oriented safety recommendations.
- Laptop, USB, video, and phone-camera input.
- Offline multilingual audio warnings.
- Coursework-compatible batch processing.
- Model evaluation, packaging, and presentation demonstration.

It does not cover report writing, presentation slide writing, or academic
chapter preparation.

## Related Official Inputs

- [Coursework images](./data/official/assignment_images/)
- [Coursework documents](./data/official/coursework_documents/)
- [P0-P14 implementation report](./TECHNICAL_DEVELOPMENT_REPORT_P0_P14.md)

The new implementation must be developed independently from the old generated
models, labels, measurements, and source code.

---

# 1. Final Technical Objective

Develop an offline road-sign intelligence application that can:

1. Locate and segment every visible road sign in an image or video frame.
2. Recognize the actual meaning of supported Malaysian road signs.
3. Identify coursework signs as `sign_001` through `sign_099` where an official
   or independently verified mapping exists.
4. Read sign text written in Bahasa Melayu, English, and Chinese.
5. Extract variable values such as speed, height, weight, time, and distance.
6. Convert recognition results into useful ADAS safety information.
7. Process live video from a laptop webcam, USB camera, or phone camera.
8. Display masks, bounding boxes, sign meaning, confidence, OCR text, and
   recommended action in real time.
9. Announce stable warnings in selectable English, Bahasa Melayu, or Mandarin.
10. Process every coursework image in less than two seconds on a lab machine.
11. Handle unfamiliar signs safely through an explicit `unknown_sign` result.
12. Run fully offline during the final demonstration.

## 1.1 Honest Supported-Coverage Claim

The evaluated production target is:

- 60 to 80 validated Malaysian semantic sign classes.
- Every class represented by the 84 coursework images, where its meaning can be
  verified.
- OCR support for Malay/English Latin text, Chinese text, and numbers.
- Unknown-sign handling for signs outside the validated catalogue.

The project must not claim that it recognizes every possible sign in Malaysia
unless that claim is supported by a complete labelled dataset and measured
evaluation.

## 1.2 Final Processing Pipeline

```text
Camera, image, video, or inputFiles.txt
                 |
                 v
       Road-sign instance segmentation
                 |
                 v
        Detection and track association
                 |
                 v
       Rectified road-sign image crop
                 |
          +------+------+
          |             |
          v             v
 Semantic classifier   Multilingual OCR
          |             |
          +------+------+
                 |
                 v
       Temporal confidence fusion
                 |
                 v
      Malaysian sign catalogue lookup
                 |
                 v
        Deterministic ADAS decision
                 |
       +---------+----------+
       |         |          |
       v         v          v
      UI      JSON API    Audio warning
```

---

# 2. Fixed Technology Decisions

| Area | Selected technology | Purpose |
|---|---|---|
| Main language | Python 3.11 | Training, inference, evaluation, and backend |
| Environment | `uv`, `pyproject.toml`, `uv.lock` | Reproducible dependency management |
| Training | PyTorch | Deep-learning model development |
| Sign segmentation | YOLO26 segmentation family | Road-sign masks and bounding boxes |
| GPU production model | YOLO26s-seg candidate | Accurate real-time segmentation |
| CPU fallback model | YOLO26s-seg candidate with YOLO26n fallback | Lab-machine and low-power inference |
| Semantic classifier | EfficientNetV2-S | Main exact sign classifier |
| Lightweight classifier | MobileNetV3-Large | Speed comparison and fallback |
| OCR | PaddleOCR 3.x PP-OCRv6 | Malay, English, Chinese, and numeric text |
| Tracking | BoT-SORT | Stable identities in moving-camera video |
| Deployment inference | ONNX Runtime | Consistent GPU and CPU execution |
| Backend | FastAPI and Pydantic | Local API and event validation |
| Live communication | WebSockets | Camera frames and recognition events |
| Frontend | React, TypeScript, Vite | Live operational dashboard |
| Camera I/O | OpenCV | Webcam, video, image, and frame processing |
| Annotation | CVAT | Masks, boxes, classes, and OCR transcripts |
| Data/model versioning | DVC and Git | Reproducible datasets and models |
| Testing | pytest, Vitest, Playwright | Backend, frontend, and end-to-end tests |
| Python quality | Ruff and Pyright | Formatting, linting, and type checking |
| Windows packaging | PyInstaller and Inno Setup | One-click offline application |

## 2.1 Architecture Decision

The production recognizer will use:

```text
Generic sign segmenter -> semantic crop classifier -> conditional OCR
```

This is preferred over a single multi-class detector because:

- Similar signs can be classified from higher-resolution crops.
- Speed-limit values and sign text can be processed separately.
- OCR can support sign content not represented by a fixed visual class.
- Detection data can be shared across all semantic classes.
- New sign meanings can be added without retraining the generic detector.

---

# 3. Clean Repository Design

```text
C:\MiniProject\
  apps\
    api\
      roadsign_api\
      tests\
    web\
      src\
      public\
      tests\
  src\
    roadsign_assist\
      baseline\
      catalogue\
      classification\
      datasets\
      detection\
      evaluation\
      inference\
      ocr\
      semantics\
      tracking\
  configs\
    baseline\
    data\
    inference\
    models\
  data\
    official\
    raw\
    annotations\
    processed\
    splits\
    manifests\
  models\
    checkpoints\
    exported\
  assets\
    audio\
    certificates\
    demo\
  outputs\
    evaluations\
    predictions\
    benchmarks\
    logs\
  scripts\
    setup.ps1
    train.ps1
    evaluate.ps1
    run.ps1
    package.ps1
  tests\
    acceptance\
    integration\
    model\
    unit\
  .dvc\
  dvc.yaml
  dvc.lock
  params.yaml
  pyproject.toml
  uv.lock
  README.md
```

## 3.1 Repository Rules

- Source code and small configuration files are tracked by Git.
- Datasets, annotations, checkpoints, exports, and large evaluation artifacts
  are tracked by DVC.
- Raw data is never modified in place.
- Generated data is reproducible from source data and configuration.
- Training, validation, and test splits are stored and versioned.
- No model can read source filenames or folder names as prediction features.
- No private faces or number plates are retained without anonymization.
- Every imported dataset must include provenance and licence records.

---

# 4. Master Progress Dashboard

Use the status values:

- `[ ]` Not started
- `[-]` In progress
- `[x]` Completed and verified
- `[!]` Blocked
- `[~]` Completed with an accepted limitation

| Phase | Name | Status | Owner | Target date | Evidence |
|---|---|---:|---|---|---|
| P0 | Destructive reset and backup | `[x]` | Codex | 2026-06-25 | External backup and restored tree verified: 84 images, 24 documents, zero checksum mismatches |
| P1 | Repository and environment foundation | `[~]` | Codex | 2026-06-24 | Reproducible CUDA environment, CI, DVC remote, scripts, and quality gates verified; initial Git commit left to owner |
| P2 | Malaysian sign ontology | `[-]` | Codex | 2026-06-24 | 81-entry validated draft catalogue; authoritative two-person review pending |
| P3 | Coursework image mapping | `[-]` | Codex | 2026-06-24 | 84-image inventory and mapping template created; semantic review pending |
| P4 | Data acquisition and provenance | `[~]` | Codex | 2026-06-25 | Licence policy, Zenodo provenance, 510-image/66-class EMTD subset, checksums, and DVC remote verified; local field collection remains |
| P5 | Annotation and quality control | `[~]` | Codex | 2026-06-25 | 1,227 source boxes validated; SAM 2.1 produced 1,220 draft masks on 507 images with 3 QA rejects; independent human review remains |
| P6 | Leakage-safe dataset construction | `[~]` | Codex | 2026-06-26 | 507 duplicate groups, deterministic group-stratified splits, adjacent-video session leakage test, 1,064 semantic crops, DVC release; one rare class absent from validation/test |
| P7 | Classical color/shape baseline | `[x]` | Codex | 2026-06-25 | 84/84 processed; masks/crops/features saved; six frozen-split SVM/RF comparisons completed |
| P8 | Deep-learning sign segmentation | `[~]` | Codex | 2026-06-25 | 640/960 YOLO26s-seg comparison complete; live 640 model has small-sign recall 0.674 and CPU p95 1.04 s; final targets/review remain |
| P9 | Semantic sign classification | `[~]` | Codex | 2026-06-26 | MobileNet and EfficientNet trained/exported; live EfficientNet ONNX now combines confidence and embedding-distance rejection; test selective accuracy 0.829 at 0.745 coverage; macro-F1 target remains unmet |
| P10 | Multilingual OCR | `[~]` | Codex | 2026-06-25 | Frozen offline PP-OCRv6-small assets; synthetic Malay/English/Chinese/numeric smoke CER 0 and 189 ms warm mean; real-road OCR set remains |
| P11 | Tracking and temporal fusion | `[~]` | Codex | 2026-06-25 | Global motion-aware association, stable IDs, occlusion rebinding, expiration, fusion, and cooldown pass tests; BoT-SORT/GMC field validation remains |
| P12 | Sign semantics and ADAS rules | `[~]` | Codex | 2026-06-25 | Every catalogue class produces a deterministic advisory action; numeric units/ranges, unknown fallback, and cooldown are tested; independent safety review pending |
| P13 | FastAPI inference backend | `[x]` | Codex | 2026-06-26 | Health, model-status, catalogue, image, batch, video, and WebSocket contracts pass with explicit response models and exported OpenAPI schema |
| P14 | React live application | `[x]` | Codex | 2026-06-26 | Camera/image/video/batch workflows, masks, presenter mode, runtime/provider diagnostics, responsive QA, browser verification, and tests pass |
| P15 | Phone-camera streaming | `[ ]` |  |  |  |
| P16 | Offline multilingual audio | `[ ]` |  |  |  |
| P17 | Coursework batch compatibility | `[ ]` |  |  |  |
| P18 | Evaluation and optimization | `[ ]` |  |  |  |
| P19 | Windows packaging | `[ ]` |  |  |  |
| P20 | Final technical acceptance | `[ ]` |  |  |  |

## 4.1 Overall Metrics Tracker

| Metric | Required target | Current result | Status | Evidence |
|---|---:|---:|---:|---|
| Supported semantic classes | 60-80 | 81 catalogue drafts; 35 trained experimental labels | `[~]` | Catalogue and `data/processed/emtd_classification/labels.json` |
| Coursework images processed | 84/84 | 84/84 baseline | `[x]` | `outputs/baseline/results.csv` |
| Coursework runtime | <2 seconds/image on lab CPU | 787 ms mean, 1,413 ms max; 84/84 under 2 s with YOLO26s on development RTX laptop | `[~]` | `outputs/evaluation/coursework_experimental_s30/summary.json`; lab CPU pending |
| Sign detector recall | >=90% | Box 68.1%; mask 57.3% on YOLO26s EMTD test | `[~]` | `outputs/evaluation/emtd_segmenter_s30/metrics.json` |
| Small-sign recall | >=80% | 67.4% at 640 px; 68.7% at 960 px | `[~]` | `outputs/evaluation/emtd_segmenter_s30/recall_slices.json`; `outputs/evaluation/emtd_segmenter_s960_20/recall_slices.json` |
| Segmentation mask IoU | >=0.75 |  | `[ ]` |  |
| Semantic classifier macro-F1 | >=85% | 70.6% all-label test macro-F1; embedding-gated selective accuracy 82.9% at 74.5% coverage | `[~]` | `outputs/training/emtd_classifier_efficientnet15/metrics.json`; `outputs/evaluation/classifier_comparison/embedding_rejection.json` |
| Critical-class recall | >=90% | 59.6% macro/micro; uneven from 100% to 0% with tiny supports | `[~]` | `outputs/evaluation/classifier_comparison/critical_class_recall.json` |
| Unknown-sign AUROC | >=0.85 | Not measured; reviewed OOD set unavailable | `[ ]` | `outputs/evaluation/classifier_comparison/embedding_rejection.json` |
| GPU live display rate | >=15 FPS |  | `[ ]` |  |
| Stable warning latency | <=1 second |  | `[ ]` |  |
| Offline operation | Pass | Deep ONNX models and PP-OCRv6 assets run locally; Ultralytics auto-install disabled | `[~]` | Installer and clean-machine network-disabled test remain |
| Phone-camera soak test | 30 minutes |  | `[ ]` |  |
| Laptop-camera soak test | 60 minutes |  | `[ ]` |  |

---

# 5. P0 - Destructive Reset And Official Backup

## Goal

Remove the old research implementation and preserve only official coursework
inputs before creating the new system.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P0.1 | List every current top-level file and directory | `[x]` | Codex | External backup inventory and checksum manifest |
| P0.2 | Create `C:\MiniProject_OfficialBackup` | `[x]` | Codex | Backup directory outside workspace |
| P0.3 | Copy `Color Inputs` into the backup | `[x]` | Codex | 84 preserved images |
| P0.4 | Copy `Practical 1` into the backup | `[x]` | Codex | 24 preserved documents |
| P0.5 | Generate SHA-256 checksums for backup files | `[x]` | Codex | `C:\MiniProject_OfficialBackup\SHA256SUMS.csv` |
| P0.6 | Verify all 84 images are readable | `[x]` | Codex | Pillow integrity audit |
| P0.7 | Verify official documents open successfully | `[x]` | Codex | DOCX ZIP integrity audit |
| P0.8 | Delete old source, models, outputs, generated labels, plans, environments, and imported datasets | `[x]` | Codex | Greenfield workspace tree |
| P0.9 | Restore official inputs under `data/official` | `[x]` | Codex | 84 images and 24 documents |
| P0.10 | Compare restored checksums with backup | `[x]` | Codex | `outputs/evaluation/reset_audit.json` |

## Completion Gate

- [x] Exactly 84 official images are present and readable.
- [x] Official coursework documents are present.
- [x] All restored checksums match the backup.
- [x] No old model, generated label, output, or source file remains.
- [x] The backup path is outside the reset workspace.

---

# 6. P1 - Repository And Environment Foundation

## Goal

Create a reproducible Windows development environment without custom package
folders or hard-coded Python paths.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P1.1 | Initialize a fresh Git repository | `[x]` | Codex | `.git/` |
| P1.2 | Initialize DVC | `[x]` | Codex | `.dvc/` |
| P1.3 | Configure local DVC remote at `C:\MiniProjectData\dvc-remote` | `[x]` | Codex | `.dvc/config` |
| P1.4 | Initialize a Python 3.11 `uv` project | `[x]` | Codex | `pyproject.toml` |
| P1.5 | Create and commit `uv.lock` | `[~]` | Codex | Lock created; owner has not requested a Git commit |
| P1.6 | Create CPU, GPU, training, and development dependency groups | `[x]` | Codex | Dependency extras and CUDA wheel index |
| P1.7 | Initialize React, TypeScript, and Vite frontend | `[x]` | Codex | `apps/web` |
| P1.8 | Configure Ruff formatting and linting | `[x]` | Codex | Ruff passes |
| P1.9 | Configure Pyright strict type checking | `[x]` | Codex | 0 errors and warnings |
| P1.10 | Configure pytest and coverage | `[x]` | Codex | 22 tests pass |
| P1.11 | Configure Vitest and Playwright | `[x]` | Codex | 1 component and 6 E2E runs pass |
| P1.12 | Add structured application logging | `[x]` | Codex | `logging_config.py` |
| P1.13 | Add deterministic random seed handling | `[x]` | Codex | `params.yaml` seed 2513 |
| P1.14 | Add CPU, CUDA, camera, disk, and model diagnostics | `[x]` | Codex | Healthy diagnostics on RTX 4050 |
| P1.15 | Create setup, run, train, evaluate, and package PowerShell scripts | `[x]` | Codex | `scripts/*.ps1` |
| P1.16 | Add continuous-integration checks | `[x]` | Codex | `.github/workflows/quality.yml` |
| P1.17 | Make the first clean repository commit | `[ ]` | Owner | Commit intentionally not created without owner request |

## Completion Gate

- [~] `uv sync` creates the environment on this Windows machine; a second clean machine remains untested.
- [x] No long Codex-specific or user-specific Python path is required.
- [x] Backend health command runs.
- [x] Frontend production and development toolchains run.
- [x] Ruff, Pyright, pytest, Vitest, and Playwright pass.
- [x] DVC pushed 108 official files and 801 pipeline artifacts; pull/status verification passes.

---

# 7. P2 - Malaysian Sign Ontology

## Goal

Define the authoritative meanings that the models will learn and the ADAS layer
will use.

## Canonical Catalogue Fields

| Field | Meaning |
|---|---|
| `semantic_sign_id` | Stable machine identifier |
| `category` | Regulatory, warning, mandatory, information, temporary, or text |
| `name_en` | English display name |
| `name_ms` | Bahasa Melayu display name |
| `name_zh` | Chinese display name |
| `aliases` | Alternative names and OCR phrases |
| `visual_family` | Circle, triangle, rectangle, octagon, or other |
| `base_action` | ADAS action code |
| `severity` | Information, caution, warning, or critical |
| `parameter_type` | Speed, height, weight, time, distance, direction, or none |
| `default_parameter` | Optional default value |
| `audio_key` | Offline audio phrase identifier |
| `standard_reference` | Source standard and page |
| `review_status` | Draft, reviewed, or approved |

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P2.1 | Obtain and archive authoritative Malaysian sign references | `[~]` | Codex | Four official JKR references registered; local PDF archive pending |
| P2.2 | Define ontology schema and validation rules | `[x]` | Codex | Strict Pydantic models and tests |
| P2.3 | Select the first 60-80 supported classes | `[x]` | Codex | 81-entry experimental catalogue |
| P2.4 | Define English names | `[x]` | Codex | Catalogue entries |
| P2.5 | Define Bahasa Melayu names | `[x]` | Codex | Catalogue entries |
| P2.6 | Define Chinese names | `[x]` | Codex | UTF-8 catalogue entries and validation |
| P2.7 | Define aliases and common sign text | `[x]` | Codex | Exact and conservative fuzzy alias matching |
| P2.8 | Assign visual families and parameter types | `[x]` | Codex | Catalogue metadata |
| P2.9 | Assign severity levels | `[x]` | Codex | Safety metadata |
| P2.10 | Define deterministic ADAS action codes | `[x]` | Codex | Action enum and catalogue |
| P2.11 | Add standard references for every class | `[x]` | Codex | Every entry references a registered JKR source |
| P2.12 | Perform independent two-person review | `[ ]` |  | Review record |
| P2.13 | Freeze ontology version `v1.0` | `[~]` | Codex | Versioned draft catalogue; approval pending |

## Completion Gate

- [x] Every supported class has all required fields.
- [x] Every meaning and action has a cited reference.
- [x] No semantic class is defined only from an unverified web image.
- [ ] Two reviewers approve every safety-critical entry.
- [x] Catalogue validation tests pass.

---

# 8. P3 - Coursework Image Mapping

## Goal

Create trustworthy coursework ID and semantic labels without relying on the old
generated labels.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P3.1 | Inventory the 84 official images | `[x]` | Codex | `data/manifests/official_images.csv` |
| P3.2 | Extract filename prefixes as candidate coursework IDs | `[x]` | Codex | Candidate ID column |
| P3.3 | Independently inspect every image | `[~]` | Codex | One full visual review; second reviewer pending |
| P3.4 | Assign visible semantic meanings where unambiguous | `[x]` | Codex | Draft semantic mapping |
| P3.5 | Mark uncertain signs as unresolved | `[x]` | Codex | Explicit null mappings and notes |
| P3.6 | Resolve disagreements through a third review | `[ ]` |  | Resolution record |
| P3.7 | Confirm one-to-one and one-to-many ID relationships | `[x]` | Codex | 84-image expanded mapping |
| P3.8 | Create `coursework_manifest.csv` | `[x]` | Codex | `data/manifests/coursework_manifest.csv` |
| P3.9 | Create a visual contact sheet for all IDs | `[x]` | Codex | Full and representative review sheets |
| P3.10 | Freeze coursework ground truth with DVC | `[~]` | Codex | Draft single-review manifest DVC-backed |

## Required Separation

The final output must include two independent fields:

```text
coursework_id: sign_020
semantic_sign_id: school_zone
```

An unresolved semantic meaning must not prevent evaluation of the verified
coursework ID, and an unresolved coursework ID must not prevent semantic
recognition.

## Completion Gate

- [x] All 84 images have verified file integrity.
- [ ] All visible signs have manually reviewed masks or bounding boxes.
- [x] Every mapping includes reviewer identity and confidence.
- [x] Ambiguous mappings remain explicitly unresolved.
- [x] The model evaluation code never reads the ground-truth filename prefix as
      an input feature.

---

# 9. P4 - Data Acquisition And Provenance

## Goal

Build a broad Malaysian dataset with enough real variation to support the
selected semantic classes.

## Source Record Fields

- Source name and URL.
- Owner or publisher.
- Licence and usage restrictions.
- Download date.
- SHA-256 checksum.
- Original class definitions.
- Imported class mapping.
- Geographic relevance.
- Known quality concerns.
- Number of accepted and rejected samples.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P4.1 | Define data licence acceptance policy | `[x]` | Codex | `docs/DATA_LICENCE_POLICY.md` |
| P4.2 | Register Malaysian public datasets | `[x]` | Codex | `data/manifests/dataset_sources.json` |
| P4.3 | Download accepted public datasets | `[~]` | Codex | 510-image balanced EMTD subset, DVC-backed; full 7.47 GB archive not required |
| P4.4 | Verify checksums and archive licences | `[x]` | Codex | 510/510 SHA-256 checks and Zenodo CC BY 4.0 provenance |
| P4.5 | Design local photo/video collection protocol | `[x]` | Codex | `docs/DATA_COLLECTION_PROTOCOL.md` |
| P4.6 | Collect daytime urban footage | `[ ]` |  | Raw videos |
| P4.7 | Collect highway and rural footage | `[ ]` |  | Raw videos |
| P4.8 | Collect night, rain, glare, and low-light footage | `[ ]` |  | Raw videos |
| P4.9 | Collect multilingual and text-heavy signs | `[ ]` |  | Raw images/videos |
| P4.10 | Extract useful frames | `[ ]` |  | Candidate frames |
| P4.11 | Remove perceptual duplicates | `[x]` | Codex | 507 duplicate-safe groups from 510 images |
| P4.12 | Blur faces and number plates | `[ ]` |  | Anonymized dataset |
| P4.13 | Collect no-sign negative scenes | `[ ]` |  | Negative dataset |
| P4.14 | Create approved synthetic sign templates | `[ ]` |  | Template library |
| P4.15 | Generate realistic synthetic scenes | `[ ]` |  | Synthetic dataset |
| P4.16 | Enforce synthetic-data proportion limits | `[ ]` |  | Class statistics |
| P4.17 | Produce per-class coverage report | `[x]` | Codex | Split diagnostics and classifier metadata |

## Class Eligibility Rules

| Level | Minimum evidence | Allowed claim |
|---|---|---|
| Supported | 200 real training instances and 40 real validation/test instances where feasible | Included in primary accuracy |
| Experimental | Some verified data but below the supported threshold | Demo only, reported separately |
| Unsupported | Insufficient or unverified data | Must return unknown |

## Completion Gate

- [ ] Every source has an acceptable licence record.
- [ ] At least 20% of road-scene frames contain no target sign.
- [ ] Dataset covers difficult lighting, blur, occlusion, distance, and rotation.
- [ ] Synthetic examples are no more than 50% of any supported class.
- [ ] A coverage report identifies weak classes before annotation begins.

---

# 10. P5 - Annotation And Quality Control

## Goal

Create reviewed segmentation, classification, and OCR ground truth.

## Annotation Requirements

Each sign instance should contain:

- Instance segmentation polygon.
- Bounding box.
- Semantic sign class.
- Coursework ID when applicable.
- Exact visible text transcript.
- Script/language label.
- Numeric parameter and unit.
- Visibility level.
- Blur level.
- Occlusion level.
- Lighting condition.
- Sign condition.
- Source route/session.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P5.1 | Install and configure CVAT | `[ ]` |  | Annotation service |
| P5.2 | Create labels and attribute schema | `[x]` | Codex | CVAT label configuration |
| P5.3 | Write annotation instructions | `[x]` | Codex | `docs/ANNOTATION_GUIDE.md` |
| P5.4 | Create annotator training task | `[ ]` |  | Training set |
| P5.5 | Measure inter-annotator agreement | `[ ]` |  | Agreement report |
| P5.6 | Annotate official coursework images | `[ ]` |  | Verified masks/boxes |
| P5.7 | Annotate public dataset samples | `[~]` | Codex | 1,227 EMTD boxes and 1,220 SAM draft masks; human acceptance pending |
| P5.8 | Annotate local collected data | `[ ]` |  | Accepted annotations |
| P5.9 | Annotate OCR text and numeric values | `[ ]` |  | OCR ground truth |
| P5.10 | Perform independent review | `[ ]` |  | Review status |
| P5.11 | Run polygon and box validation | `[x]` | Codex | Bounds, area-ratio, box-IoU QA; 3 mask drafts rejected |
| P5.12 | Run transcript consistency validation | `[ ]` |  | OCR QA report |
| P5.13 | Randomly audit 10% of accepted labels | `[ ]` |  | Audit report |
| P5.14 | Correct rejected annotations | `[ ]` |  | Final annotation set |
| P5.15 | Freeze annotation release `v1.0` | `[~]` | Codex | Experimental draft release pushed to DVC; reviewed v1.0 pending |

## Completion Gate

- [ ] Sampled annotation correctness is at least 98%.
- [ ] Every accepted instance has valid geometry.
- [ ] Every OCR instance has a reviewed transcript.
- [ ] Safety-critical classes receive full second-person review.
- [ ] Annotation exports can be converted reproducibly into training formats.

---

# 11. P6 - Leakage-Safe Dataset Construction

## Goal

Build fair, reproducible training, validation, and test sets.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P6.1 | Create a unified sample manifest | `[x]` | Codex | `data/manifests/dataset.csv` |
| P6.2 | Group samples by physical sign, route, and recording session | `[x]` | Codex | Effective duplicate/source groups |
| P6.3 | Detect exact file duplicates | `[x]` | Codex | SHA-256 manifest |
| P6.4 | Detect perceptual near-duplicates | `[x]` | Codex | dHash grouping, threshold 6 |
| P6.5 | Reserve all 84 coursework images as external test data | `[x]` | Codex | Training safety gate enforces zero coursework images |
| P6.6 | Generate 70/15/15 grouped splits | `[x]` | Codex | Deterministic group-stratified manifests |
| P6.7 | Confirm adjacent video frames stay together | `[x]` | Codex | `tests/unit/test_split.py::test_adjacent_video_frames_with_same_session_never_cross_splits` |
| P6.8 | Produce segmentation dataset | `[~]` | Codex | 507 images/1,220 SAM draft masks; manual review pending |
| P6.9 | Produce classification crop dataset | `[x]` | Codex | 1,064 crops across 35 semantic labels |
| P6.10 | Produce OCR dataset | `[~]` | Codex | Synthetic multilingual smoke set; reviewed real-road OCR set pending |
| P6.11 | Generate class-distribution reports | `[x]` | Codex | `data/splits/diagnostics.json` and dataset metadata |
| P6.12 | Add automatic leakage refusal checks | `[x]` | Codex | Training rejects coursework and unreviewed production data |
| P6.13 | Freeze dataset release `v1.0` | `[~]` | Codex | Experimental DVC release pushed; reviewed v1.0 pending |

## Completion Gate

- [x] No physical sign appears across multiple splits.
- [x] No adjacent video sequence is divided across splits in the implemented session-grouping test; real captured video-session review remains pending.
- [x] No coursework image is used for training or model selection.
- [x] Duplicate and leakage tests pass automatically.
- [x] Re-running the DVC stage reproduces identical manifests.

---

# 12. P7 - Classical Color And Shape Baseline

## Goal

Rebuild the Report 1 foundation cleanly and use it as an explainable comparison,
not as the final production recognizer.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P7.1 | Implement image loading and preprocessing | `[x]` | Codex | Baseline pipeline |
| P7.2 | Implement HSV red segmentation using two hue ranges | `[x]` | Codex | Red masks |
| P7.3 | Implement HSV blue segmentation | `[x]` | Codex | Blue masks |
| P7.4 | Implement HSV yellow segmentation | `[x]` | Codex | Yellow masks |
| P7.5 | Implement configurable morphology | `[x]` | Codex | Open/close and background refinement |
| P7.6 | Extract contours and connected components | `[x]` | Codex | Candidate regions |
| P7.7 | Calculate area, aspect ratio, extent, and solidity | `[x]` | Codex | Geometry features |
| P7.8 | Calculate circularity, vertices, and Hu moments | `[x]` | Codex | Shape features |
| P7.9 | Extract HOG and color-histogram features | `[x]` | Codex | Three frozen feature sets |
| P7.10 | Train SVM baseline | `[x]` | Codex | Three Joblib models |
| P7.11 | Train Random Forest baseline | `[x]` | Codex | Three Joblib models |
| P7.12 | Save masks, crops, annotations, and results | `[x]` | Codex | `outputs/baseline` |
| P7.13 | Measure runtime and recognition metrics | `[x]` | Codex | Six-run comparison CSV/JSON |
| P7.14 | Test without filename/folder information | `[x]` | Codex | Pixel-only feature contract and tests |
| P7.15 | Add baseline unit and regression tests | `[x]` | Codex | Synthetic multi-color and feature tests |

## Completion Gate

- [x] The baseline runs on arbitrary image paths.
- [x] Expected sign colors are never read from folder names.
- [x] Red, blue, and yellow masks are saved for technical inspection.
- [x] SVM and Random Forest use frozen dataset splits.
- [x] Baseline runtime and failure cases are recorded.

---

# 13. P8 - Deep-Learning Sign Segmentation

## Goal

Find every road sign independently of its color and output an instance mask and
bounding box.

## Experiments

| Experiment | Model | Input size | Purpose | Status |
|---|---|---:|---|---:|
| DET-01 | YOLO26n-seg | 512 | Experimental balanced baseline | `[~]` |
| DET-02 | YOLO26s-seg | 640 | Main balanced model | `[~]` |
| DET-03 | YOLO26s-seg | 960 | Small-sign comparison | `[~]` |
| DET-04 | YOLO26 P2 segmentation configuration | 640/960 | Dedicated small-object comparison | `[ ]` |

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P8.1 | Convert annotations to YOLO segmentation format | `[x]` | Codex | SAM-box prompt conversion pipeline |
| P8.2 | Validate converted masks visually | `[~]` | Codex | Accepted/failure contact sheets; full second-person review pending |
| P8.3 | Define detector parameters in `params.yaml` | `[x]` | Codex | Reproducible detector config |
| P8.4 | Train YOLO26n-seg | `[x]` | Codex | 20-epoch experimental checkpoint |
| P8.5 | Train YOLO26s-seg | `[x]` | Codex | 30 epochs at 640 px |
| P8.6 | Train small-sign candidate | `[x]` | Codex | 20-epoch 960 px checkpoint, ONNX export, parity, metrics, and CPU benchmark |
| P8.7 | Tune confidence and mask thresholds on validation data | `[x]` | Codex | Five-threshold validation study; 0.10 candidate, 0.25 retained until negative-scene evaluation |
| P8.8 | Evaluate box mAP and mask mAP | `[x]` | Codex | YOLO26s test box mAP50 0.740; mask mAP50 0.598 |
| P8.9 | Evaluate overall and small-sign recall | `[x]` | Codex | At IoU 0.50, 640 px small-sign recall 0.674 and 960 px recall 0.687 on 227 small instances |
| P8.10 | Evaluate false positives on no-sign scenes | `[ ]` |  | Negative test |
| P8.11 | Select GPU and CPU production candidates | `[~]` | Codex | YOLO26s selected experimentally; YOLO26n retained as lower-power fallback; no production promotion |
| P8.12 | Export selected models to ONNX | `[x]` | Codex | Experimental ONNX export |
| P8.13 | Verify PyTorch/ONNX output parity | `[x]` | Codex | Maximum metric difference 0.000129 |
| P8.14 | Benchmark RTX 4050 and CPU inference | `[x]` | Codex | 640 ONNX: CUDA 80.2 ms, CPU p95 1,041 ms; 960 ONNX: CUDA 1,906 ms anomaly, CPU p95 853 ms |

## Completion Gate

- [ ] Overall test recall is at least 90%.
- [ ] Small-sign recall is at least 80%.
- [ ] Mean mask IoU is at least 0.75.
- [ ] No-sign false-positive rate is at most 5%.
- [x] ONNX outputs remain within the approved numerical tolerance.
- [x] The selected CPU detector supports the coursework runtime requirement on the development laptop; lab CPU confirmation remains.

---

# 14. P9 - Semantic Sign Classification

## Goal

Recognize the exact semantic meaning of each detected sign crop.

## Experiments

| Experiment | Model | Role | Status |
|---|---|---|---:|
| CLS-01 | MobileNetV3-Large | Lightweight baseline | `[x]` |
| CLS-02 | EfficientNetV2-S | Main accuracy candidate | `[x]` |
| CLS-02E | EfficientNetV2-S + prototype embedding gate | Unknown rejection candidate | `[x]` |
| CLS-03 | Classical SVM on deep embeddings | Report 2 comparison | `[ ]` |

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P9.1 | Generate rectified classification crops | `[x]` | Codex | 1,064 padded semantic crops |
| P9.2 | Define realistic crop augmentation | `[x]` | Codex | Resize, perspective, rotation, blur, color transforms |
| P9.3 | Implement class-balanced sampling/loss | `[x]` | Codex | Weighted sampler and label smoothing |
| P9.4 | Train MobileNetV3-Large | `[x]` | Codex | Test macro-F1 0.675 |
| P9.5 | Train EfficientNetV2-S | `[x]` | Codex | Test macro-F1 0.706 |
| P9.6 | Compare accuracy, macro-F1, size, and latency | `[x]` | Codex | Reproducible comparison CSV/JSON |
| P9.7 | Add temperature confidence calibration | `[x]` | Codex | Temperature 1.213; ECE 0.094 |
| P9.8 | Add embedding-distance unknown rejection | `[x]` | Codex | ONNX logits+embedding export, cosine prototype gate, and unit tests |
| P9.9 | Tune class and unknown thresholds | `[~]` | Codex | Confidence 0.72 plus cosine distance 0.3775; reviewed OOD AUROC set pending |
| P9.10 | Evaluate safety-critical class recall | `[x]` | Codex | Frozen critical-class recall report; 59.6%, target unmet |
| P9.11 | Evaluate confusion between visually similar classes | `[x]` | Codex | 35x35 confusion matrix stored in metrics |
| P9.12 | Export selected classifiers to ONNX | `[x]` | Codex | MobileNet, EfficientNet, and EfficientNet logits+embedding experimental exports |
| P9.13 | Verify PyTorch/ONNX parity | `[x]` | Codex | Embedding export parity passed; logits max diff 0.0138 and embedding max diff 0.00085 |
| P9.14 | Select production and CPU fallback models | `[~]` | Codex | Embedding-gated EfficientNet selected experimentally; no production promotion |

## Completion Gate

- [ ] Supported-class macro-F1 is at least 85%.
- [ ] Critical-class recall is at least 90%.
- [ ] Expected calibration error is at most 5%.
- [ ] Unknown-sign AUROC is at least 0.85.
- [x] Low-confidence inputs return `unknown_sign`.
- [ ] No unsupported class is presented as confidently recognized.

---

# 15. P10 - Multilingual OCR

## Goal

Read and interpret sign text in Bahasa Melayu, English, Chinese, and numbers.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P10.1 | Audit current PaddleOCR 3.x model options | `[x]` | Codex | PP-OCRv6-small selected |
| P10.2 | Build Latin-script OCR test set | `[~]` | Codex | Synthetic Malay/English smoke set; real set pending |
| P10.3 | Build Chinese OCR test set | `[~]` | Codex | Synthetic Chinese smoke set; real set pending |
| P10.4 | Build numeric and unit OCR test set | `[~]` | Codex | Synthetic numeric smoke set; real set pending |
| P10.5 | Implement sign perspective rectification | `[x]` | Codex | Quadrilateral perspective correction |
| P10.6 | Implement local contrast and sharpness preprocessing | `[x]` | Codex | CLAHE and unsharp masking |
| P10.7 | Integrate Latin OCR recognition | `[x]` | Codex | Frozen PP-OCRv6 local model |
| P10.8 | Integrate Chinese OCR recognition | `[x]` | Codex | Frozen multilingual PP-OCRv6 model |
| P10.9 | Implement script/language routing | `[x]` | Codex | Latin/Chinese/mixed detection |
| P10.10 | Normalize Unicode, punctuation, and spacing | `[x]` | Codex | Normalization module and tests |
| P10.11 | Extract numbers and units | `[x]` | Codex | Speed/height/weight/distance parser |
| P10.12 | Match OCR results against catalogue aliases | `[x]` | Codex | Exact alias lookup |
| P10.13 | Add fuzzy matching with strict safety limits | `[x]` | Codex | Threshold and margin-based fuzzy match |
| P10.14 | Cache OCR by stable sign track | `[x]` | Codex | Per-track cache; still images explicitly stable |
| P10.15 | Measure character and word error rates | `[~]` | Codex | Synthetic CER 0; real-road CER/WER pending |

## Completion Gate

- [~] Malay/English, Chinese, and numeric results are evaluated separately on synthetic smoke data; reviewed real-road sets remain.
- [x] Raw OCR text is preserved for inspection.
- [x] Uncertain text does not generate a confident safety action.
- [x] OCR is not executed unnecessarily on every frame.
- [x] Numeric values include validated units and allowed ranges.

---

# 16. P11 - Tracking And Temporal Fusion

## Goal

Stabilize live predictions and prevent flickering labels or repeated warnings.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P11.1 | Integrate BoT-SORT with custom detector outputs | `[~]` | Codex | Global motion-aware association implemented; full BoT-SORT ReID/GMC remains |
| P11.2 | Assign stable track IDs | `[x]` | Codex | Session-local track state |
| P11.3 | Configure moving-camera compensation | `[~]` | Codex | Constant-velocity prediction; camera homography/GMC pending |
| P11.4 | Fuse classifier confidence over time | `[x]` | Codex | Decayed semantic score fusion |
| P11.5 | Fuse OCR results over time | `[x]` | Codex | Stable-track OCR cache |
| P11.6 | Require stable frames before action | `[x]` | Codex | Configurable stability gate |
| P11.7 | Implement stale-track expiration | `[x]` | Codex | Missed-frame lifecycle |
| P11.8 | Implement duplicate-warning cooldown | `[x]` | Codex | Monotonic per-track cooldown |
| P11.9 | Handle temporary occlusion and reappearance | `[x]` | Codex | Motion prediction and rebinding test |
| P11.10 | Test multiple simultaneous signs | `[x]` | Codex | Global assignment identity test |
| P11.11 | Test camera movement and motion blur | `[ ]` |  | Tracking report |

## Completion Gate

- [x] Labels remain stable for a stationary visible sign.
- [x] The same tracked sign does not repeatedly announce without reason.
- [x] New signs can still trigger promptly.
- [x] Expired signs no longer affect vehicle recommendations.
- [x] Multiple signs retain independent track IDs.

---

# 17. P12 - Sign Semantics And ADAS Rules

## Goal

Convert model outputs into deterministic, explainable driving-assistance events.

## Initial Action Codes

```text
STOP_REQUEST
YIELD
REDUCE_SPEED
SET_TARGET_SPEED
PROHIBIT_LEFT_TURN
PROHIBIT_RIGHT_TURN
PROHIBIT_ENTRY
KEEP_LEFT
KEEP_RIGHT
FOLLOW_DIRECTION
WATCH_PEDESTRIANS
WATCH_CHILDREN
WATCH_ROAD_HAZARD
WATCH_TRAFFIC_SIGNAL
HEIGHT_RESTRICTION
WEIGHT_RESTRICTION
INFORMATION_ONLY
UNKNOWN_CAUTION
```

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P12.1 | Define ADAS event schema | `[x]` | Codex | Strict Pydantic API models |
| P12.2 | Map each supported class to a base action | `[x]` | Codex | 81-entry rule catalogue |
| P12.3 | Define severity levels | `[x]` | Codex | Information/caution/warning/critical |
| P12.4 | Define confidence thresholds by action risk | `[x]` | Codex | Critical and normal thresholds |
| P12.5 | Implement parameterized speed actions | `[x]` | Codex | Confidence, KM/H, and 5-160 range validation |
| P12.6 | Implement height and weight restrictions | `[x]` | Codex | Unit/range validation and CM-to-M normalization |
| P12.7 | Implement directional and prohibition rules | `[x]` | Codex | Deterministic direction metadata |
| P12.8 | Implement school and pedestrian warnings | `[x]` | Codex | Catalogue actions |
| P12.9 | Implement unknown-sign caution behavior | `[x]` | Codex | Safe unknown fallback |
| P12.10 | Add evidence and confidence to every event | `[x]` | Codex | Versioned event contract |
| P12.11 | Add rule unit tests for every class | `[x]` | Codex | Determinism/advisory loop over full catalogue |
| P12.12 | Perform independent safety review | `[ ]` |  | Review record |

## Completion Gate

- [x] Every supported class has a deterministic action.
- [x] Every action is explainable from catalogue data.
- [x] No language model is used to make a safety decision.
- [x] Low confidence results degrade to caution or no action.
- [x] The system is clearly advisory and cannot control real hardware.

---

# 18. P13 - FastAPI Inference Backend

## Goal

Expose the complete inference pipeline to the web application and batch tools.

## Required Endpoints

```text
GET  /api/v1/health
GET  /api/v1/catalogue
GET  /api/v1/models
POST /api/v1/infer/image
POST /api/v1/infer/batch
POST /api/v1/infer/video
WS   /api/v1/ws/camera/{session_id}
```

## Core Event Contract

```json
{
  "schema_version": "1.0",
  "frame_id": 314,
  "track_id": 12,
  "coursework_id": "sign_020",
  "semantic_sign_id": "school_zone",
  "meaning": {
    "en": "School zone",
    "ms": "Zon sekolah",
    "zh": "School-zone Chinese label"
  },
  "ocr_text": "ZON SEKOLAH",
  "confidence": 0.93,
  "bbox_xyxy": [120, 80, 210, 260],
  "mask": {
    "encoding": "polygon",
    "points": []
  },
  "action": {
    "code": "REDUCE_SPEED",
    "target_speed_kmh": 30
  },
  "severity": "warning",
  "latency_ms": 64,
  "device": "cuda"
}
```

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P13.1 | Create versioned API schemas | `[x]` | Codex | Pydantic event, catalogue, health, model-status, batch, and video response models |
| P13.2 | Implement model registry and loading | `[x]` | Codex | Shared model backends and status registry |
| P13.3 | Implement health and diagnostics endpoint | `[x]` | Codex | `/api/v1/health` |
| P13.4 | Implement single-image inference | `[x]` | Codex | `/api/v1/infer/image` |
| P13.5 | Implement batch inference | `[x]` | Codex | `/api/v1/infer/batch` |
| P13.6 | Implement video-file inference | `[x]` | Codex | `/api/v1/infer/video` |
| P13.7 | Implement WebSocket camera sessions | `[x]` | Codex | `/api/v1/ws/camera/{session_id}` |
| P13.8 | Implement bounded latest-frame queues | `[x]` | Codex | One outstanding frame with sequential server backpressure |
| P13.9 | Implement GPU/CPU provider selection | `[x]` | Codex | CUDA/ONNX providers and automatic baseline fallback |
| P13.10 | Implement session cleanup and cancellation | `[x]` | Codex | Independent trackers, temporary-video cleanup, and invalid-upload cleanup tests |
| P13.11 | Add structured error responses | `[x]` | Codex | HTTP status details, upload limit errors, WebSocket error objects, and recovery after bad camera frames |
| P13.12 | Add API and WebSocket contract tests | `[x]` | Codex | Image, batch, video, health, model-status, catalogue, OpenAPI, WS, and bad-frame recovery tests |
| P13.13 | Add latency and resource instrumentation | `[x]` | Codex | Frame latency, device, CUDA, input, and model health |

## Completion Gate

- [x] All endpoints validate inputs and outputs.
- [x] One failed frame does not terminate the camera session.
- [x] Queues remain bounded under slow client conditions.
- [x] CPU fallback is available when deep weights or CUDA are unavailable.
- [x] Contract and integration tests pass.

---

# 19. P14 - React Live Application

## Goal

Create a polished operational interface for scanning signs and demonstrating
driving-assistance decisions.

## Required Views

1. Live camera.
2. Image and video analysis.
3. Coursework batch runner.
4. Event history.
5. System diagnostics.
6. Settings.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P14.1 | Define the dashboard information architecture | `[x]` | Codex | Three-rail operational dashboard |
| P14.2 | Build source selector | `[x]` | Codex | Camera, image, video, and batch controls |
| P14.3 | Build full live-video surface | `[x]` | Codex | Webcam and uploaded-media surfaces |
| P14.4 | Draw masks, boxes, labels, and track IDs | `[x]` | Codex | Letterbox-correct SVG/HTML overlay renderer |
| P14.5 | Build current-sign information panel | `[x]` | Codex | Sign details panel |
| P14.6 | Display OCR text and detected language | `[x]` | Codex | OCR facts |
| P14.7 | Display ADAS recommendation | `[x]` | Codex | Advisory action panel |
| P14.8 | Build simulated vehicle-state display | `[x]` | Codex | Target speed and action state |
| P14.9 | Build event timeline | `[x]` | Codex | Bounded event history |
| P14.10 | Display FPS, latency, device, and model health | `[x]` | Codex | Live diagnostics include detector device and classifier provider details; deep diagnostics covered by E2E |
| P14.11 | Add EN/BM/ZH audio selection and mute | `[x]` | Codex | Language and mute controls; playback is P16 |
| P14.12 | Build batch results table and gallery | `[x]` | Codex | Multi-image thumbnails and results |
| P14.13 | Add presenter mode | `[x]` | Codex | One-click expanded workspace |
| P14.14 | Implement responsive laptop/mobile layouts | `[x]` | Codex | Desktop and 390px mobile layouts |
| P14.15 | Add reconnecting and error states | `[x]` | Codex | Five-attempt WebSocket recovery, malformed-response handling, backend-offline recovery, and visible errors |
| P14.16 | Add Vitest component tests | `[x]` | Codex | Component and camera message parser tests pass |
| P14.17 | Add Playwright workflow tests | `[x]` | Codex | Ten desktop/mobile workflows pass |
| P14.18 | Verify no overlapping UI at target resolutions | `[x]` | Codex | QA screenshots under `outputs/` |

## Completion Gate

- [x] The first screen is the usable live application.
- [~] `run.ps1` starts the app; a no-terminal executable belongs to P19 packaging.
- [x] Multiple detections remain readable.
- [x] All important states have loading, empty, error, and recovery handling.
- [x] Desktop and mobile Playwright tests pass.

---

# 20. P15 - Phone-Camera Streaming

## Goal

Use a phone browser as a wireless camera while the RTX laptop performs
inference.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P15.1 | Define local hotspot/Wi-Fi setup | `[ ]` |  | Network guide |
| P15.2 | Generate local HTTPS certificate | `[ ]` |  | Certificate assets |
| P15.3 | Create QR-code connection flow | `[ ]` |  | QR component |
| P15.4 | Request phone camera permission | `[ ]` |  | Camera client |
| P15.5 | Select front/rear camera and resolution | `[ ]` |  | Camera controls |
| P15.6 | Encode and send binary frames | `[ ]` |  | WebSocket client |
| P15.7 | Add adaptive JPEG quality | `[ ]` |  | Bandwidth control |
| P15.8 | Drop stale frames under latency | `[ ]` |  | Queue policy |
| P15.9 | Return recognition events to phone | `[ ]` |  | Event display |
| P15.10 | Implement disconnection and reconnection | `[ ]` |  | Recovery flow |
| P15.11 | Test Android browser | `[ ]` |  | Test record |
| P15.12 | Test iPhone browser when available | `[ ]` |  | Test record |
| P15.13 | Run 30-minute offline soak test | `[ ]` |  | Soak report |

## Completion Gate

- [ ] Phone camera works with internet disabled.
- [ ] Connection can recover after temporary network loss.
- [ ] Frame queues do not grow without limit.
- [ ] Average latency remains suitable for live demonstration.
- [ ] Laptop webcam remains an immediate fallback.

---

# 21. P16 - Offline Multilingual Audio

## Goal

Announce useful sign warnings in selectable English, Bahasa Melayu, and
Mandarin without internet access.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P16.1 | Write warning phrase templates | `[ ]` |  | Phrase catalogue |
| P16.2 | Review English phrases | `[ ]` |  | Approved text |
| P16.3 | Review Bahasa Melayu phrases | `[ ]` |  | Approved text |
| P16.4 | Review Mandarin phrases | `[ ]` |  | Approved text |
| P16.5 | Record or generate offline audio assets | `[ ]` |  | Audio files |
| P16.6 | Normalize volume and format | `[ ]` |  | Processed audio |
| P16.7 | Add parameterized speed warnings | `[ ]` |  | Speed audio |
| P16.8 | Add parameterized restriction warnings | `[ ]` |  | Restriction audio |
| P16.9 | Implement language selection | `[ ]` |  | Audio service |
| P16.10 | Implement priority and interruption policy | `[ ]` |  | Playback policy |
| P16.11 | Implement track cooldown | `[ ]` |  | Deduplication |
| P16.12 | Add generic unknown-sign caution | `[ ]` |  | Fallback audio |
| P16.13 | Test every supported audio key | `[ ]` |  | Audio test report |

## Completion Gate

- [ ] Every supported class has a valid audio mapping.
- [ ] Missing audio cannot crash inference.
- [ ] The same sign does not repeat continuously.
- [ ] Critical warnings can take priority over information messages.
- [ ] All audio works with networking disabled.

---

# 22. P17 - Coursework Batch Compatibility

## Goal

Meet the official assessment interface and runtime requirements.

## Required Command

```powershell
roadsign-assist batch --input inputFiles.txt --output outputs\coursework
```

## Required Output Per Image

- Input path.
- Detected sign count.
- Instance masks.
- Bounding boxes.
- Cropped signs.
- Coursework sign ID.
- Semantic sign meaning.
- OCR text and parameters.
- Confidence.
- Runtime.
- Failure or uncertainty reason.
- Annotated output image.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P17.1 | Implement exact `inputFiles.txt` parsing | `[ ]` |  | Batch loader |
| P17.2 | Support relative and absolute paths | `[ ]` |  | Path tests |
| P17.3 | Process multiple signs per image | `[ ]` |  | Batch inference |
| P17.4 | Save masks and bounding boxes | `[ ]` |  | Segmentation outputs |
| P17.5 | Save predicted coursework IDs | `[ ]` |  | Prediction table |
| P17.6 | Save semantic meanings and OCR | `[ ]` |  | Semantic table |
| P17.7 | Save per-image runtime | `[ ]` |  | Runtime table |
| P17.8 | Produce CSV and JSON summaries | `[ ]` |  | Summary files |
| P17.9 | Produce annotated image gallery | `[ ]` |  | Visual evidence |
| P17.10 | Rename and relocate test images | `[ ]` |  | Leakage test |
| P17.11 | Run all 84 official images | `[ ]` |  | Acceptance output |
| P17.12 | Benchmark target lab CPU | `[ ]` |  | Runtime report |

## Completion Gate

- [ ] All 84 images complete without a crash.
- [ ] Every detected sign is segmented.
- [ ] Verified coursework IDs are predicted from image pixels.
- [ ] Every image finishes in less than two seconds on the target lab machine.
- [ ] Renaming files does not change predictions.

---

# 23. P18 - Evaluation And Optimization

## Goal

Measure the complete system honestly and optimize it without hiding accuracy
losses.

## Required Evaluation Groups

- Overall segmentation.
- Small and distant signs.
- Semantic classification.
- Safety-critical classes.
- Unknown signs.
- Malay/English OCR.
- Chinese OCR.
- Numeric OCR.
- End-to-end semantic recognition.
- GPU live performance.
- CPU coursework performance.
- Difficult environmental conditions.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P18.1 | Implement segmentation metrics | `[ ]` |  | mAP/IoU/recall |
| P18.2 | Implement classification metrics | `[ ]` |  | Accuracy/F1/report |
| P18.3 | Implement calibration metrics | `[ ]` |  | ECE/reliability |
| P18.4 | Implement unknown-detection metrics | `[ ]` |  | AUROC/FPR |
| P18.5 | Implement OCR metrics | `[ ]` |  | CER/WER |
| P18.6 | Implement end-to-end event metrics | `[ ]` |  | Event accuracy |
| P18.7 | Implement condition-based analysis | `[ ]` |  | Failure slices |
| P18.8 | Benchmark PyTorch and ONNX Runtime | `[ ]` |  | Runtime comparison |
| P18.9 | Evaluate TensorRT feasibility | `[ ]` |  | Decision record |
| P18.10 | Add asynchronous frame capture | `[ ]` |  | Optimized pipeline |
| P18.11 | Add bounded latest-frame processing | `[ ]` |  | Stable memory use |
| P18.12 | Profile CPU, GPU, memory, and I/O | `[ ]` |  | Profile report |
| P18.13 | Optimize bottlenecks | `[ ]` |  | Performance changes |
| P18.14 | Re-run accuracy after every optimization | `[ ]` |  | Regression report |
| P18.15 | Produce final model cards | `[ ]` |  | Model documentation |

## TensorRT Decision Rule

TensorRT will be included only if it:

- Improves p95 inference latency by at least 20%.
- Does not create a meaningful accuracy regression.
- Can be packaged reliably on the presentation laptop.
- Does not endanger the CPU fallback or reproducible ONNX path.

## Completion Gate

- [ ] Every claimed metric is produced by a reproducible command.
- [ ] Test data is not used for threshold selection.
- [ ] Optimization does not silently reduce accuracy.
- [ ] Failure cases are saved and categorized.
- [ ] GPU and CPU results are reported separately.

---

# 24. P19 - Windows Packaging

## Goal

Provide a one-click offline application for presentation day.

## Tasks

| ID | Task | Status | Owner | Deliverable or evidence |
|---|---|---:|---|---|
| P19.1 | Build frontend production assets | `[ ]` |  | Static web bundle |
| P19.2 | Serve frontend from FastAPI | `[ ]` |  | Integrated app |
| P19.3 | Create application launcher | `[ ]` |  | Launcher module |
| P19.4 | Add single-instance protection | `[ ]` |  | Process guard |
| P19.5 | Add automatic browser opening | `[ ]` |  | Launch flow |
| P19.6 | Package backend with PyInstaller | `[ ]` |  | One-folder build |
| P19.7 | Bundle models and audio | `[ ]` |  | Offline assets |
| P19.8 | Bundle HTTPS certificates | `[ ]` |  | Phone-camera support |
| P19.9 | Add model and asset integrity checks | `[ ]` |  | Startup validation |
| P19.10 | Add automatic GPU/CPU selection | `[ ]` |  | Runtime selection |
| P19.11 | Add restart/recovery control | `[ ]` |  | Recovery flow |
| P19.12 | Create Inno Setup installer | `[ ]` |  | Installer |
| P19.13 | Test installation on a clean Windows account | `[ ]` |  | Installation report |
| P19.14 | Test with internet disabled | `[ ]` |  | Offline report |
| P19.15 | Test uninstall and reinstall | `[ ]` |  | Packaging report |

## Completion Gate

- [ ] `RoadSignAssist.exe` launches the complete system.
- [ ] No terminal commands are needed for the normal presentation.
- [ ] The app starts within 15 seconds on the presentation laptop.
- [ ] The app works with networking disabled.
- [ ] The installer works on a clean Windows account.

---

# 25. P20 - Final Technical Acceptance

## Functional Acceptance

- [ ] Laptop webcam input works.
- [ ] USB camera input works.
- [ ] Phone-camera input works.
- [ ] Image upload works.
- [ ] Video upload works.
- [ ] `inputFiles.txt` batch processing works.
- [ ] Multiple signs can be processed in one frame.
- [ ] Masks and bounding boxes are visible.
- [ ] Exact semantic sign meanings are displayed.
- [ ] Malay/English OCR works.
- [ ] Chinese OCR works.
- [ ] Numeric values are extracted.
- [ ] Unknown signs are rejected safely.
- [ ] ADAS recommendations are generated.
- [ ] Vehicle response simulation works.
- [ ] English audio works.
- [ ] Bahasa Melayu audio works.
- [ ] Mandarin audio works.

## Performance Acceptance

- [ ] 84/84 coursework images complete.
- [ ] Coursework runtime is below two seconds per image on the lab CPU.
- [ ] Detector recall is at least 90%.
- [ ] Small-sign recall is at least 80%.
- [ ] Mean mask IoU is at least 0.75.
- [ ] Semantic macro-F1 is at least 85%.
- [ ] Safety-critical recall is at least 90%.
- [ ] Unknown-sign AUROC is at least 0.85.
- [ ] GPU live interface runs at least 15 FPS.
- [ ] Stable warning is produced within one second.
- [ ] Laptop-camera one-hour soak test passes.
- [ ] Phone-camera 30-minute soak test passes.

## Safety And Integrity Acceptance

- [ ] Predictions do not depend on filenames or source folders.
- [ ] Low-confidence signs never force a dangerous action.
- [ ] Every ADAS event includes evidence and confidence.
- [ ] Every semantic action comes from deterministic reviewed rules.
- [ ] No cloud service is required during inference.
- [ ] No real vehicle hardware is controlled.
- [ ] Dataset licences and provenance are complete.
- [ ] Faces and number plates are anonymized where required.

---

# 26. Recommended Four-Person Technical Allocation

| Member | Primary ownership | Secondary ownership |
|---|---|---|
| Member 1 | Ontology, coursework mapping, classical baseline | Data provenance and evaluation |
| Member 2 | Segmentation annotation and YOLO training | Dataset QA and optimization |
| Member 3 | Semantic classification, OCR, calibration | Unknown detection and model export |
| Member 4 | Backend, frontend, phone camera, audio, packaging | Live performance and integration |

All members must participate in:

- Annotation review.
- Field testing.
- Failure analysis.
- Final acceptance testing.
- Independent verification of safety-critical mappings.

## 26.1 Assignment Tracker

| Task ID | Assigned member | Start date | Due date | Review member | Status |
|---|---|---|---|---|---:|
|  |  |  |  |  | `[ ]` |
|  |  |  |  |  | `[ ]` |
|  |  |  |  |  | `[ ]` |
|  |  |  |  |  | `[ ]` |
|  |  |  |  |  | `[ ]` |

---

# 27. Weekly Milestone Tracker

The sequence below is a recommended minimum. Additional weeks should be used for
data collection and model improvement rather than expanding unsupported claims.

| Week | Intended outcome | Status | Actual outcome/evidence |
|---:|---|---:|---|
| 1 | Reset, repository, environment, and CI | `[x]` | External backup audit, reproducible environment, CI, and DVC verified |
| 2 | Ontology and coursework mapping | `[~]` | 81-entry draft and 84-image single-review mapping; independent review pending |
| 3 | Data provenance, collection protocol, and CVAT setup | `[~]` | Public provenance/protocol complete; CVAT service and local collection pending |
| 4 | First annotation release and split pipeline | `[~]` | Experimental SAM/source-box release and leakage-safe splits; review pending |
| 5 | Clean classical baseline | `[x]` | Six SVM/RF feature comparisons and 84-image batch artifacts |
| 6 | First YOLO segmentation models | `[x]` | YOLO26n-seg and YOLO26s-seg trained |
| 7 | Segmentation improvement and ONNX export | `[~]` | 640/960 YOLO26s comparison, parity, recall slices, and GPU/CPU benchmarks complete; recall target remains unmet |
| 8 | Semantic classifiers and unknown rejection | `[~]` | Two CNNs, calibration, confidence rejection, and embedding-distance prototype gate; reviewed OOD AUROC pending |
| 9 | Multilingual OCR and parameter parsing | `[~]` | Offline OCR, parsing, unit/range safety complete; real-road set pending |
| 10 | Tracking, fusion, catalogue, and ADAS rules | `[~]` | Motion-aware tracking and deterministic advisory rules; field/safety review pending |
| 11 | FastAPI backend and React live application | `[x]` | Endpoints, WebSocket, dashboard, desktop/mobile E2E and Chrome QA pass |
| 12 | Phone camera, audio, and vehicle simulator | `[ ]` |  |
| 13 | Coursework compatibility and CPU benchmark | `[ ]` |  |
| 14 | Optimization, packaging, and offline testing | `[ ]` |  |
| 15 | Field tests, soak tests, and failure corrections | `[ ]` |  |
| 16 | Final acceptance and demonstration rehearsal | `[ ]` |  |

---

# 28. Dataset Release Tracker

| Version | Date | Real instances | Synthetic instances | Supported classes | Annotation QA | DVC revision | Notes |
|---|---|---:|---:|---:|---:|---|---|
| `v0.1` |  |  |  |  |  |  | Initial import |
| `v0.2` |  |  |  |  |  |  | First annotations |
| `v1.0` |  |  |  |  |  |  | First training release |
| `v1.1` |  |  |  |  |  |  | Failure-case expansion |
| `v2.0` |  |  |  |  |  |  | Final evaluated release |

---

# 29. Model Experiment Tracker

| Experiment ID | Model | Data version | Parameters | Primary metric | Result | Runtime | Artifact | Status |
|---|---|---|---|---|---:|---:|---|---:|
| DET-01 | YOLO26n-seg | EMTD draft 510 | 20 epochs, 512 px | Test mask mAP50 | 0.487; recall 0.440 | ONNX 23.3 ms/model image on RTX 4050 | `models/exported/experimental/emtd_segmenter_n20.onnx` | `[~]` |
| DET-01-CPU | YOLO26n-seg ONNX | EMTD test 63 | 512 px, conf 0.25 | CPU wall latency | 272 ms mean; 419 ms p95 | CPUExecutionProvider | `outputs/evaluation/emtd_segmenter_n20/cpu_runtime.json` | `[x]` |
| DET-02 | YOLO26s-seg | EMTD draft 510 | 30 epochs, 640 px | Test mask mAP50 | 0.598; recall 0.573 | ONNX 80.2 ms/model image on RTX 4050 | `models/exported/experimental/emtd_segmenter_s30.onnx` | `[~]` |
| DET-02-CPU | YOLO26s-seg ONNX | EMTD test 63 | 640 px, conf 0.25 | CPU wall latency | 522 ms mean; 1,041 ms p95 | CPUExecutionProvider | `outputs/evaluation/emtd_segmenter_s30/cpu_runtime.json` | `[x]` |
| DET-03 | YOLO26s-seg 960 | EMTD draft 510 | 20 epochs, 960 px | Small-sign box recall at IoU 0.50 | 0.687; mask mAP50 0.708 | PyTorch 64.9 ms; ONNX CUDA 1,906 ms; CPU p95 853 ms | `models/exported/experimental/emtd_segmenter_s960_20.onnx` | `[~]` |
| DET-04 | YOLO26 P2 seg |  |  | Small recall |  |  |  | `[ ]` |
| CLS-01 | MobileNetV3-Large | EMTD 1,064 crops | 15 epochs, 192 px | Test macro-F1 | 0.675 | 17.0 MB ONNX | `models/exported/experimental/emtd_classifier_mobilenet15.onnx` | `[~]` |
| CLS-02 | EfficientNetV2-S | EMTD 1,064 crops | 15 epochs, 192 px | Test macro-F1 | 0.706; selective accuracy 0.818 | 80.7 MB ONNX | `models/exported/experimental/emtd_classifier_efficientnet15.onnx` | `[~]` |
| CLS-02E | EfficientNetV2-S logits+embedding | EMTD 1,064 crops | Train prototypes from train split; tune distance on validation | Test selective accuracy | 0.829 at 0.745 coverage; distance threshold 0.3775 | 80.7 MB ONNX plus 1.4 MB calibration | `models/exported/experimental/emtd_classifier_efficientnet15_embedding.onnx` | `[~]` |
| CLS-03 | SVM deep embeddings |  |  | Macro-F1 |  |  |  | `[ ]` |
| BASE-01 | SVM and Random Forest | EMTD 1,064 crops | HOG, HOG+HSV, all handcrafted | Test macro-F1 | Best RF all-handcrafted 0.570 | Six Joblib models | `outputs/evaluation/baseline_classifiers/comparison.csv` | `[x]` |
| OCR-01 | PP-OCRv6-small Latin | Synthetic smoke 3 | Local CPU, conditional | CER | 0.000 | 189 ms warm mean across all scripts | `models/ocr` | `[~]` |
| OCR-02 | PP-OCRv6-small Chinese | Synthetic smoke 2 | Local CPU, conditional | CER | 0.000 | Included above | `models/ocr` | `[~]` |

## 29.1 Production Model Registry

| Role | Selected model | Version | Format | Input | Output | Threshold file | Status |
|---|---|---|---|---|---|---|---:|
| GPU segmenter | YOLO26s-seg experimental | s30 | ONNX | 640 full frame | Masks/boxes | Conf 0.25; validation candidate 0.10 | `[~]` |
| CPU segmenter | YOLO26s-seg experimental | s30 | ONNX | 640 full frame | Masks/boxes | Conf 0.25; 522 ms mean | `[~]` |
| GPU classifier | EfficientNetV2-S embedding-gated experimental | 15 epochs + prototype gate | ONNX | 192 crop | 35 semantic labels or `unknown_sign` | Confidence 0.72; cosine distance 0.3775 | `[~]` |
| CPU classifier |  |  | ONNX | Sign crop | Semantic class |  | `[ ]` |
| Latin OCR | PP-OCRv6-small | 3.7.0 | Paddle | Text crop | Text/numeric | Local checksum manifest | `[~]` |
| Chinese OCR | PP-OCRv6-small | 3.7.0 | Paddle | Text crop | Chinese text | Local checksum manifest | `[~]` |

---

# 30. Test Run Tracker

| Test run | Date | Commit | Data version | Model versions | Device | Result | Evidence |
|---|---|---|---|---|---|---|---|
| Unit suite | 2026-06-26 | Uncommitted greenfield tree | EMTD draft | Experimental | CPU/GPU | 70 passed | pytest |
| API integration | 2026-06-25 | Uncommitted greenfield tree | EMTD draft | Experimental | CPU/GPU | Pass | FastAPI tests and live HTTP |
| Frontend E2E | 2026-06-26 | Uncommitted greenfield tree | Mock contracts | N/A | Chromium desktop/mobile | 10 passed | Playwright |
| Coursework 84 | 2026-06-25 | Uncommitted greenfield tree | External acceptance | YOLO26s hybrid experimental | RTX 4050 laptop | 84/84 events; mean 787 ms, max 1,413 ms; draft semantic exact 15.5% | `outputs/evaluation/coursework_experimental_s30` |
| Detector CPU | 2026-06-25 | Uncommitted greenfield tree | EMTD test 63 | YOLO26s-seg ONNX | CPUExecutionProvider | 522 ms mean, 1,041 ms p95, 1,285 ms max | `outputs/evaluation/emtd_segmenter_s30/cpu_runtime.json` |
| Detector small-sign slice | 2026-06-25 | Uncommitted greenfield tree | EMTD test 63 | YOLO26s 640/960 | RTX 4050 | 640: 153/227, 67.4%; 960: 156/227, 68.7%; 640 retained for live use | `outputs/evaluation/emtd_segmenter_s30/recall_slices.json` |
| Laptop live | 2026-06-25 | Uncommitted greenfield tree | Held-out EMTD image | YOLO26s experimental ONNX | RTX 4050 | 10-point mask; speed 30 recognized; advisory `SET_TARGET_SPEED`; 1,147 ms HTTP | `outputs/evaluation/live_api_s30_speed30.json` |
| Live embedding profile | 2026-06-26 | Uncommitted greenfield tree | Held-out EMTD image | YOLO26s + EfficientNet embedding ONNX + PP-OCRv6 | RTX 4050 CUDAExecutionProvider | Health loaded detector/classifier/OCR; API evidence includes `embedding:<label>:<distance>` and `classifier_rejection:confidence` | `GET /api/v1/health`; `POST /api/v1/infer/image` smoke |
| Phone live |  |  |  |  | RTX 4050 |  |  |
| Offline test |  |  |  |  | RTX 4050 |  |  |
| One-hour soak |  |  |  |  | RTX 4050 |  |  |

---

# 31. Failure Case Tracker

| ID | Date | Input/track | Failure type | Expected | Actual | Root cause | Fix task | Retest | Status |
|---|---|---|---|---|---|---|---|---|---:|
| F-001 |  |  |  |  |  |  |  |  | `[ ]` |
| F-002 | 2026-06-25 | Initial dashboard | Incorrect status label | Deep pipeline label before first frame | Classical baseline | Label used only frame result, not health model mode | Use result mode or health mode | Chrome reload shows Semantic AI pipeline | `[x]` |
| F-003 | 2026-06-25 | ONNX CPU benchmark | Offline package auto-install attempt | Use installed runtime without network/pip | Ultralytics attempted `pip install onnxruntime` | Auto-install default enabled | Set `YOLO_AUTOINSTALL=false` before CLI imports | Latest server starts without install attempt | `[x]` |
| F-004 | 2026-06-26 | Dashboard first render | Misleading loading fallback | Unknown pipeline while health is loading | Classical baseline could appear before health resolved | Unknown mode fell through to baseline label | Add explicit `Checking pipeline` state and display backend runtime providers | In-app browser shows Semantic AI Pipeline, `cuda:0`, CUDA provider, no overflow, no console warnings | `[x]` |
| F-005 | 2026-06-26 | Health contract audit | Frontend/backend schema drift | Health includes computed readiness and runtime fields | `diagnostics.healthy` was typed in React but omitted from API JSON | Health used raw dataclass fields and untyped model-status dicts | Add explicit Pydantic response models and contract assertions | API tests pass and `/api/v1/health` schema includes healthy, device, providers, OCR status | `[x]` |
| F-006 | 2026-06-26 | OpenAPI contract audit | Untyped catalogue schema | Every React-facing HTTP endpoint has an explicit OpenAPI response schema | `/api/v1/catalogue` returned a strict model but OpenAPI exposed it as an unstructured dict | Missing response model on catalogue route | Add `SignCatalogue` response model, OpenAPI export script, and schema tests | `outputs/api/openapi.json` has 6 paths and typed response refs; API tests pass | `[x]` |
| F-007 | 2026-06-26 | Camera WebSocket audit | Fragile live-camera message handling | Bad camera frames and malformed responses do not kill the workflow | Backend recovery was covered weakly; React parsed camera messages inline | Missing bad-frame continuation test and parser boundary | Add backend recovery test and frontend camera message parser | Bad frame returns an error, next valid frame succeeds; malformed client message sets recoverable error | `[x]` |
| F-008 | 2026-06-26 | Upload boundary audit | Weak upload-limit and cleanup regression coverage | Oversized uploads are rejected and failed video uploads leave no temp files | Size/cleanup behavior existed but was not directly tested | Missing API regression tests | Add image-size, batch-count, and invalid-video cleanup tests | Oversized image returns 413, 101-image batch returns 400, invalid video temp file is removed | `[x]` |
| F-009 | 2026-06-26 | Dashboard E2E audit | Weak coverage of non-happy UI states | Offline and deep-runtime states are browser-tested | Playwright covered happy workflows but not backend-offline or deep diagnostics rendering | Missing E2E tests for recovery and runtime-provider display | Add offline-refresh and deep-diagnostics Playwright cases | Desktop/mobile Playwright suite passes 10 tests | `[x]` |

## Failure Categories

- Missed sign.
- Incorrect segmentation.
- Incorrect semantic class.
- Unknown sign accepted as known.
- Known sign rejected as unknown.
- Incorrect OCR.
- Incorrect numeric value.
- Incorrect ADAS action.
- Unstable tracking.
- Duplicate audio warning.
- Runtime violation.
- Camera or network failure.
- UI or packaging failure.

---

# 32. Risk Tracker

| Risk | Probability | Impact | Mitigation | Owner | Status |
|---|---|---|---|---|---:|
| Insufficient Malaysian data per class | High | High | Reduce supported class list or collect more real data |  | `[ ]` |
| Public labels are inconsistent | High | High | Manual remapping and independent review |  | `[ ]` |
| Chinese sign data is too limited | Medium | High | Targeted collection and OCR-specific augmentation |  | `[ ]` |
| Small signs are missed | High | High | Higher-resolution/P2 experiments and targeted data |  | `[ ]` |
| CPU runtime exceeds two seconds | Medium | High | YOLO26s current p95 1.04 s; retain YOLO26n fallback and test lab CPU |  | `[~]` |
| Phone HTTPS setup fails | Medium | Medium | Prepare certificate early and retain webcam fallback |  | `[ ]` |
| OCR slows live inference | High | Medium | Run only on stable text tracks and cache results |  | `[ ]` |
| Model confidence is misleading | Medium | High | Temperature calibration plus embedding-distance rejection; OOD set still required | Codex | `[~]` |
| Dataset leakage inflates accuracy | Medium | High | Grouped splits and duplicate refusal checks |  | `[ ]` |
| Packaging misses a model or DLL | Medium | High | Integrity check and clean-machine test |  | `[ ]` |
| Internet-dependent component remains | Low | High | Network-disabled acceptance test |  | `[ ]` |
| Audio repeats excessively | Medium | Medium | Track-based cooldown and priority queue |  | `[ ]` |

---

# 33. Decision Log

Record changes that affect architecture, data, class coverage, safety rules, or
acceptance targets.

| Date | Decision | Reason | Alternatives rejected | Approved by |
|---|---|---|---|---|
| 2026-06-24 | Restart as a greenfield project | Existing system recognizes color groups rather than semantic sign meanings | Continue extending legacy code | Project owner |
| 2026-06-24 | Preserve only official coursework inputs during reset | User requested a complete technical restart | Preserve legacy models and labels | Project owner |
| 2026-06-24 | Use detector + classifier + OCR architecture | Better support for pictograms, text, and variable values | Single multi-class detector | Project owner |
| 2026-06-24 | Use Git and DVC | Reproducible code, datasets, and models | Untracked data directories | Project owner |
| 2026-06-24 | Target 60-80 validated classes | Broad but measurable Malaysian coverage | Unsupported claim of all possible signs | Project owner |
| 2026-06-24 | Keep inference fully offline | Reliable presentation and privacy | Cloud OCR or cloud APIs | Project owner |

---

# 34. Change Log

| Date | Version | Change | Author |
|---|---|---|---|
| 2026-06-24 | `0.1` | Initial greenfield technical development plan and tracker | Codex |
| 2026-06-25 | `0.2` | Added DVC-backed EMTD release, SAM draft masks, YOLO26 segmentation, CNN comparison, offline PP-OCRv6, hybrid fallback, deep live profile, and measured acceptance results | Codex |
| 2026-06-25 | `0.3` | Verified deterministic DVC split reproduction and live ONNX polygon-mask, OCR, and ADAS output | Codex |
| 2026-06-25 | `0.4` | Added reset checksum audit, six classical classifier experiments, motion-aware global tracking, strict numeric ADAS safety, detector threshold/CPU evaluations, offline auto-install guard, and corrected live pipeline status | Codex |
| 2026-06-25 | `0.5` | Trained and selected experimental YOLO26s-seg for the live profile: test mask mAP50 0.598, mask recall 0.573, ONNX parity pass, CPU p95 1.04 s, and 84/84 coursework images under two seconds | Codex |
| 2026-06-25 | `0.6` | Completed 960 px small-sign experiment and normalized-area recall slices; 1.3-point small-sign gain did not justify its 1.91 s ONNX CUDA inference, so 640 px remains the live default | Codex |
| 2026-06-26 | `0.7` | Added EfficientNet logits+embedding ONNX export, cosine prototype unknown rejection, true classifier-provider health reporting, adjacent-video leakage regression test, and refreshed DVC/model verification | Codex |
| 2026-06-26 | `0.8` | Polished P14 diagnostics: fixed loading-state pipeline fallback, displayed detector/classifier runtime details, added a regression assertion, rebuilt the React app, and re-ran browser/Playwright verification | Codex |
| 2026-06-26 | `0.9` | Hardened P13/P14 API contracts with explicit health/model/batch/video response models, restored `diagnostics.healthy` in health JSON, aligned frontend batch nullability, and raised Python/API tests to 64 passing | Codex |
| 2026-06-26 | `0.10` | Added typed catalogue OpenAPI response, reproducible `scripts/export_openapi.py`, OpenAPI schema contract tests, and raised Python/API tests to 66 passing | Codex |
| 2026-06-26 | `0.11` | Hardened live-camera reliability with WebSocket bad-frame recovery tests, defensive React camera-message parsing, parser unit tests, and raised Python/API tests to 67 passing | Codex |
| 2026-06-26 | `0.12` | Hardened upload boundaries with oversized-image, oversized-batch, and invalid-video cleanup regression tests, raising Python/API tests to 70 passing | Codex |
| 2026-06-26 | `0.13` | Expanded P14 Playwright coverage for backend-offline recovery and deep runtime/provider diagnostics, raising desktop/mobile E2E coverage to 10 passing tests | Codex |

---

# 35. Final Demonstration Technical Checklist

## Before Presentation Day

- [ ] Install final build on the presentation laptop.
- [ ] Verify GPU and CPU inference.
- [ ] Verify all model and audio checksums.
- [ ] Disable internet and run full application.
- [ ] Test laptop webcam.
- [ ] Test USB camera if used.
- [ ] Test phone QR connection.
- [ ] Trust the prepared local certificate on the phone.
- [ ] Test English audio.
- [ ] Test Bahasa Melayu audio.
- [ ] Test Mandarin audio.
- [ ] Prepare physical example signs.
- [ ] Prepare a difficult prerecorded Malaysian road video.
- [ ] Prepare an offline copy of the 84 coursework images.
- [ ] Run the final 84-image acceptance test.
- [ ] Save final benchmark results locally.
- [ ] Rehearse recovery from camera or WebSocket failure.

## Recommended Live Sequence

1. Launch `RoadSignAssist.exe`.
2. Show the system health and offline status.
3. Demonstrate the classical color and shape segmentation foundation.
4. Switch to semantic AI mode.
5. Scan standard pictorial signs.
6. Scan a numerical sign such as a speed limit.
7. Scan Malay, English, and Chinese text signs.
8. Show stable tracking and no repeated warning.
9. Show the ADAS event and simulated vehicle reaction.
10. Show an unfamiliar sign returning `unknown_sign`.
11. Connect the phone camera using the QR code.
12. Run the prerecorded difficult road scene.
13. Run the compulsory `inputFiles.txt` batch.
14. Display the generated accuracy and runtime results.

## Presentation Fallbacks

- [ ] Laptop webcam if phone connection fails.
- [ ] Prerecorded route if live lighting is poor.
- [ ] CPU model if CUDA initialization fails.
- [ ] Silent visual warnings if audio device fails.
- [ ] Saved acceptance results if a lab machine is unavailable.

---

# 36. Definition Of Done

The project is technically complete only when:

1. All P0-P20 completion gates are satisfied or explicitly accepted with a
   documented limitation.
2. The clean environment can be reproduced from Git and DVC.
3. The supported class list is backed by reviewed data.
4. All claimed metrics are produced on frozen held-out data.
5. The 84 coursework images run successfully within the required time.
6. The live application functions fully offline.
7. Laptop and phone camera demonstrations have passed soak testing.
8. Unknown signs are handled safely.
9. ADAS outputs are deterministic, reviewed, and advisory.
10. A one-click Windows build passes clean-machine installation testing.

---

# 37. Technical References

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26)
- [Ultralytics multi-object tracking](https://docs.ultralytics.com/modes/track)
- [PaddleOCR OCR pipeline](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html)
- [ONNX Runtime CUDA execution provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [uv project locking and synchronization](https://docs.astral.sh/uv/concepts/projects/sync/)
- [DVC data pipelines](https://dvc.org/doc/start/data-pipelines/data-pipelines)
- [CVAT documentation](https://docs.cvat.ai/)
