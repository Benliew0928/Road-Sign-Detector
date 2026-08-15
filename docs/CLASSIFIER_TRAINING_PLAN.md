# Clean classifier training report

## Outcome

The classifier plan was executed on 2026-08-13 against the frozen
`classifier_no_controlled_variants_20260812` release: 2,976 images, 78 labels,
zero controlled-background variants, zero coursework images, and the original
2,049 / 465 / 462 train, validation, and test split.

The selected candidate is `clean_b2_effnetv2s_224_e40_b32_s2513`. It has been
promoted to `models/exported/runtime/sign_classifier.*` and is now the classifier used
by both website inference profiles. The release remains
`coverage_gaps_block_final`; promotion means selected runtime candidate, not a
clean-final claim.

## Trainer changes completed before training

- Exposed learning rate, weight decay, label smoothing, seed, workers,
  confidence threshold, validation threshold tuning, and locked test evaluation
  as command options.
- Moved every candidate to its own run-specific checkpoint, report, ONNX,
  labels, and calibration directory.
- Added unique-run validation and an explicit `--overwrite` escape hatch.
- Added persistent Windows data workers to prevent per-epoch respawn overhead.
- Recorded dataset ID, Git commit, Python/PyTorch/CUDA/GPU environment, full
  configuration, best epoch, history, per-class metrics, largest confusion
  pairs, duration, ONNX size, and latency.
- Added percentage accuracy, number correct, denominator, and Wilson 95%
  confidence interval while retaining raw values.
- Kept tuning validation-only and added a separate locked test command that
  also writes a row-level prediction CSV.
- Added a promotion command that requires locked test results and passed ONNX
  parity before copying the runtime bundle.
- Added tests for no-overwrite, threshold selection, parity, and
  validation-only run comparison.

## Experiment results

All rows used AdamW, weight decay `1e-4`, label smoothing `0.05`, cosine
scheduling, weighted sampling, in-memory train-only augmentation, ImageNet
transfer learning, and validation macro-F1 selection. The smoke run is not
counted as a full experiment.

| Run | Architecture | px | LR | Seed | Workers | Best epoch | Val accuracy | Val macro-F1 | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `smoke_clean_mnv3_20260813` | MobileNetV3-Large | 224 | 3e-4 | 2513 | 4 | 1 | 53.12% | 50.50% | Smoke; parity passed |
| `clean_b1_mnv3_224_e40_s2513` | MobileNetV3-Large | 224 | 3e-4 | 2513 | 4 | 32 | 95.27% | 95.53% | Baseline B1 |
| `clean_b2_effnetv2s_224_e40_b32_s2513` | EfficientNetV2-S | 224 | 3e-4 | 2513 | 4 | 31 | 97.20% | **97.38%** | Selected predeclared seed |
| `clean_t1_effnetv2s_224_e40_lr1e3_s2513` | EfficientNetV2-S | 224 | 1e-3 | 2513 | 4 | 36 | 96.56% | 96.69% | LR challenge |
| `clean_t2_effnetv2s_224_e40_lr1e4_s2513` | EfficientNetV2-S | 224 | 1e-4 | 2513 | 4 | 25 | 97.20% | 97.17% | LR challenge |
| `clean_s2_effnetv2s_224_e40_b32_s1337` | EfficientNetV2-S | 224 | 3e-4 | 1337 | 4 | 33 | 96.99% | 97.16% | Seed repeat |
| `clean_s3_effnetv2s_224_e40_b32_s2026` | EfficientNetV2-S | 224 | 3e-4 | 2026 | 4 | 36 | 96.77% | 96.84% | Seed repeat |
| `clean_t3s1_effnetv2s_256_e40_lr3e4_s2513_w0` | EfficientNetV2-S | 256 | 3e-4 | 2513 | 0 | 20 | 96.56% | 96.64% | Image-size group |
| `clean_t3s2_effnetv2s_256_e40_lr3e4_s1337` | EfficientNetV2-S | 256 | 3e-4 | 1337 | 0 | 35 | 97.20% | 97.43% | Image-size group |
| `clean_t3s3_effnetv2s_256_e40_lr3e4_s2026` | EfficientNetV2-S | 256 | 3e-4 | 2026 | 0 | 22 | 96.99% | 97.17% | Image-size group |

An initial B2 batch-16 trial was stopped after epoch 9 because Windows worker
respawn overhead made it needlessly slow. It had already demonstrated that the
architecture was competitive (best validation macro-F1 95.80%); no test data
was read, and it was not included as a completed candidate. Batch 32 fit the
RTX 4050 and was used for the completed B2 run.

The default `3e-4` rate beat both `1e-4` and `1e-3`. The matched 224 px seed
group scored 0.9738, 0.9716, and 0.9684 macro-F1 (mean 0.9713, sample SD
0.0027). The matched 256 px group scored 0.9664, 0.9743, and 0.9717 (mean
0.9708, sample SD 0.0040). Because their means differ by only 0.0005, the plan's
tie-breakers apply: 224 px has slightly better mean selective accuracy (0.9810
vs 0.9802), mean coverage (0.9799 vs 0.9792), lower input cost, and lower seed
variance. The predeclared seed-2513 224 px run therefore remains selected.

A 256 px worker-4 pilot reached 0.9792, but it was not compared as part of the
matched seed group because the repeat runs used worker zero after Windows
worker processes were interrupted. Both matched groups use identical worker
settings within the group. The final full-run count exceeded the original
six-to-eight target only to produce matched seed evidence after this runtime
change. No blind grid or label-smoothing sweep was run; the selected validation
ECE was already 0.0218, and more searches would risk overfitting rare classes.
The selected curve peaked at epoch 31 and did not justify extending past 40.

## Locked test result

The selected configuration was locked before running:

```powershell
.\.venv\Scripts\roadsign-assist.exe evaluate-classifier-candidate `
  --run clean_b2_effnetv2s_224_e40_b32_s2513 `
  --target-selective-accuracy 0.98
```

The threshold was selected from validation predictions only, maximizing
coverage while meeting 98% validation selective accuracy.

| Metric | Result |
| --- | ---: |
| Correct | 437 / 462 |
| Raw accuracy | 94.59% |
| Wilson 95% CI | 92.13%-96.31% |
| Macro-F1, all 78 labels | 92.91% |
| Macro-F1, observed labels | 92.91% |
| Confidence threshold | 0.4770 |
| Selective accuracy | 94.99% |
| Selective coverage | 99.35% |
| Accepted-correct rate | 94.37% |
| ECE | 0.0300 |
| Calibrated temperature | 0.8074 |
| ONNX mean latency | 17.81 ms/image |
| ONNX size | 77.21 MiB |
| ONNX/PyTorch parity | Passed |

The ten largest directed confusion pairs were:

| Expected | Predicted | Count |
| --- | --- | ---: |
| `divided_road_begins` | `roadway_diverges` | 2 |
| `pass_either_side` | `permitted_u_turn` | 2 |
| `chevron_right` | `chevron_left` | 1 |
| `divided_road_begins` | `obstruction_ahead` | 1 |
| `general_caution` | `camera_enforcement` | 1 |
| `height_restriction` | `no_u_turn` | 1 |
| `keep_left` | `chevron_right` | 1 |
| `keep_left` | `keep_right` | 1 |
| `merge_left` | `merge_right` | 1 |
| `merge_right` | `merge_left` | 1 |

The most important weak result is `steep_descent`: recall and F1 were both
zero on its two test samples. `divided_road_begins` also had 25% recall on four
samples. These small supports make their uncertainty large but do not excuse
the failures; they remain collection and review priorities.

## Held-out assignment evaluation

All 84 eligible rows were evaluated and none were skipped:

| Metric | Result |
| --- | ---: |
| Raw accuracy | 70.24% |
| Observed-label macro-F1 | 64.03% |
| Selective accuracy | 71.25% |
| Selective coverage | 95.24% |
| Accepted-correct rate | 67.86% |
| Mean latency | 33.47 ms/image |

This external result is materially lower than the clean crop test result. The
external images were not moved into training or used to revise the selected
configuration. It is evidence that crop/domain robustness still needs real
data work.

## Promotion and website integration

Promotion was performed with:

```powershell
.\.venv\Scripts\roadsign-assist.exe promote-classifier `
  --run clean_b2_effnetv2s_224_e40_b32_s2513
```

`configs/inference/default.yaml` and `configs/inference/experimental.yaml` now
load `models/exported/runtime/sign_classifier.onnx`, labels, and calibration. API and
web health expose the release as `coverage_gaps_block_final`.

The runtime bundle is tracked by `models/exported/runtime.dvc` and its four
files were pushed to the shared DVC remote. Run-specific candidates remain
under ignored `models/candidates/`; checkpoints and experiment bundles were not
published. `dvc status` reports the tracked artifacts and pipelines up to date.

## Remaining limitation and next data release

The current frozen release is complete for this training cycle but cannot
become clean-final until real-data gaps are closed and a new versioned release
is frozen. Current must-have gaps are:

| Class | Current | Minimum | Gap |
| --- | ---: | ---: | ---: |
| `no_left_or_right_turn` | 7 | 50 | 43 |
| `no_straight_or_left` | 9 | 50 | 41 |
| `steep_descent` | 10 | 50 | 40 |
| `sound_horn` | 11 | 50 | 39 |
| `turn_left_or_right` | 11 | 50 | 39 |
| `residential_area_ahead` | 49 | 50 | 1 |
| `side_road_right` | 49 | 50 | 1 |

New images must go through the contributor inbox, provenance review,
deduplication, annotation, and a new frozen dataset. Do not mutate the current
release or tune against the assignment manifest.
