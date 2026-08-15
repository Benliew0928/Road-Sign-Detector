# Classifier model-run inventory

## Scope and current runtime

This is the durable inventory for the clean-classifier training cycle completed
on 2026-08-13. It supplements the decision rationale and error analysis in
[`CLASSIFIER_TRAINING_PLAN.md`](CLASSIFIER_TRAINING_PLAN.md).

All completed runs used the frozen
`classifier_no_controlled_variants_20260812` release: 2,976 images, 78 labels,
and fixed 2,049 / 465 / 462 train, validation, and test partitions. The source
release contains no controlled-background variants or coursework images.

## Clean-dataset provenance

The canonical row-level provenance record is
[`data/annotations/classifier_review_decisions.csv`](../data/annotations/classifier_review_decisions.csv).
It records the source URL, licence notes, source group, original label,
accepted/rejected decision, reviewer evidence, crop hash, and dedupe key for
every candidate. Dataset-wide release properties are recorded in
`data/processed/classifier_no_controlled_variants_20260812/dataset_metadata.json`,
and licence citations are in [`DATA_SOURCE_CITATION_LEDGER.md`](DATA_SOURCE_CITATION_LEDGER.md).

The approved manifest contains 2,978 accepted records. Two exact duplicate
pairs were collapsed during the freeze, leaving the 2,976 unique image files
used for training. The table below allocates each unique frozen image to its
reviewed source dataset; it therefore sums to the release total rather than to
the pre-deduplication candidate count.

| Reviewed source dataset | Unique frozen images | Notes |
| --- | ---: | --- |
| EMTD processed classification crop | 761 | Zenodo EMTD source, mapped and P5-reviewed crops. |
| Roboflow Malaysia Road Sign v1 | 937 | Reviewed crops; one cross-manifest exact duplicate removed. |
| Roboflow Dr Samsudin Malaysia road-sign dataset | 327 | Reviewed crops. |
| German Traffic Sign Recognition Benchmark (GTSRB) | 398 | Exact reviewed class mappings only. |
| Tsinghua-Tencent 100K (TT100K) | 354 | Exact reviewed class mappings only. |
| Malaysia Road Sign Dataset v1 via Roboflow | 50 | Exact `roadway_diverges` mining run. |
| TT100K alternate Hugging Face mirror | 8 | Reviewed, distinct source records. |
| Official-style reference diagrams | 92 | Reference-derived crops; not controlled-background variants. |
| Real-road photo visual matches | 36 | Individually reviewed public road photos. |
| Dataset-legend reference icons | 6 | Individually reviewed reference assets. |
| Public web reference diagrams | 3 | Individually reviewed reference assets. |
| Malaysian reference diagram | 1 | Individually reviewed reference asset. |
| Commercial product-photo visual matches | 3 | Four accepted manifest rows collapse to three unique bytes. |

This provenance does **not** mean every source has equal deployment value. The
next release should prioritize consented/local real-road imagery for the seven
remaining must-have gaps, while preserving the same row-level provenance and
group-safe split rules.

The website uses the promoted runtime from
`clean_b2_effnetv2s_224_e40_b32_s2513`:

| Runtime property | Value |
| --- | --- |
| Architecture | EfficientNetV2-S |
| Input | 224 x 224 RGB |
| Validation selection | macro-F1, epoch 31 |
| Locked-test accuracy | 437 / 462 (94.59%) |
| Locked-test macro-F1 | 92.91% across all 78 labels |
| Confidence threshold | 0.4770, chosen on validation only |
| ONNX parity | Passed |
| ONNX mean latency | 17.81 ms/image |
| Release status | `coverage_gaps_block_final` |

`coverage_gaps_block_final` is intentional. The classifier is the selected
runtime candidate, but it is not claimed to be a clean-final production model
while coverage gaps and real-world robustness work remain.

## Completed runs

All non-smoke runs used ImageNet transfer learning, AdamW, weight decay
`1e-4`, label smoothing `0.05`, cosine learning-rate scheduling, weighted
sampling, and train-only in-memory augmentation. Validation was used for model
selection; only the promoted run accessed the locked test set.

| Run | Architecture | Px | Batch | LR | Seed | Workers | Best epoch | Validation accuracy | Validation macro-F1 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `smoke_clean_mnv3_20260813` | MobileNetV3-Large | 224 | 32 | 3e-4 | 2513 | 4 | 1 | 53.12% | 50.50% | Smoke and ONNX-parity check only |
| `clean_b1_mnv3_224_e40_s2513` | MobileNetV3-Large | 224 | 32 | 3e-4 | 2513 | 4 | 32 | 95.27% | 95.53% | Baseline B1 |
| `clean_b2_effnetv2s_224_e40_b32_s2513` | EfficientNetV2-S | 224 | 32 | 3e-4 | 2513 | 4 | 31 | 97.20% | 97.38% | **Selected and promoted** |
| `clean_s2_effnetv2s_224_e40_b32_s1337` | EfficientNetV2-S | 224 | 32 | 3e-4 | 1337 | 4 | 33 | 96.99% | 97.16% | 224 px seed repeat |
| `clean_s3_effnetv2s_224_e40_b32_s2026` | EfficientNetV2-S | 224 | 32 | 3e-4 | 2026 | 4 | 36 | 96.77% | 96.84% | 224 px seed repeat |
| `clean_t1_effnetv2s_224_e40_lr1e3_s2513` | EfficientNetV2-S | 224 | 32 | 1e-3 | 2513 | 4 | 36 | 96.56% | 96.69% | Learning-rate challenge |
| `clean_t2_effnetv2s_224_e40_lr1e4_s2513` | EfficientNetV2-S | 224 | 32 | 1e-4 | 2513 | 4 | 25 | 97.20% | 97.17% | Learning-rate challenge |
| `clean_t3_effnetv2s_256_e40_lr3e4_s2513` | EfficientNetV2-S | 256 | 32 | 3e-4 | 2513 | 4 | 22 | 97.85% | 97.92% | Worker-4 image-size pilot; excluded from matched stability comparison |
| `clean_t3s1_effnetv2s_256_e40_lr3e4_s2513_w0` | EfficientNetV2-S | 256 | 32 | 3e-4 | 2513 | 0 | 20 | 96.56% | 96.64% | 256 px matched seed group |
| `clean_t3s2_effnetv2s_256_e40_lr3e4_s1337` | EfficientNetV2-S | 256 | 32 | 3e-4 | 1337 | 0 | 35 | 97.20% | 97.43% | 256 px matched seed group |
| `clean_t3s3_effnetv2s_256_e40_lr3e4_s2026` | EfficientNetV2-S | 256 | 32 | 3e-4 | 2026 | 0 | 22 | 96.99% | 97.17% | 256 px matched seed group |

One early EfficientNetV2-S batch-16 attempt,
`clean_b2_effnetv2s_224_e40_s2513`, stopped after epoch 9 because Windows
worker respawn overhead made it too slow. It has no completed metrics report,
was never test-evaluated, and was not considered for selection.

## Selection evidence

The comparable 224 px group had macro-F1 values of 0.9738, 0.9716, and
0.9684 (mean 0.9713, sample SD 0.0027). The comparable 256 px worker-zero
group had 0.9664, 0.9743, and 0.9717 (mean 0.9708, sample SD 0.0040).

The means differ by only 0.0005. The 224 px selection therefore used the
predeclared tie-breakers: slightly higher mean selective accuracy and coverage,
lower seed variance, and lower input cost. The worker-four 256 px pilot is
recorded above for reproducibility but cannot be mixed with the worker-zero
comparison group.

The selected run's independent held-out assignment evaluation was 59 / 84
(70.24% raw accuracy, 64.03% observed-label macro-F1). Those images were never
used to select a configuration or tune a threshold; the result is evidence of
the remaining crop/domain gap.

## Artifact locations and version control

| Artifact | Location | Stored in Git? | Notes |
| --- | --- | --- | --- |
| Per-run ONNX bundles | `models/candidates/<run>/` | No | Local, ignored candidate artifacts. Preserve or archive separately if needed. |
| Per-run metrics | `outputs/training/<run>/metrics.json` | No | Local evidence for each completed run. |
| Website runtime bundle | `models/exported/runtime/` | No, DVC-managed | Four files: ONNX, labels, calibration, and runtime manifest. |
| Website runtime pointer | `models/exported/runtime.dvc` | Yes | Commit this pointer so teammates can restore the exact runtime. |
| Training decision record | `docs/CLASSIFIER_TRAINING_PLAN.md` | Yes | Dataset, test result, external evaluation, and limitations. |

The runtime bundle was pushed to the shared DVC remote and restored into a
fresh verification directory with matching hashes. A normal `git push` carries
only code, configuration, documentation, and the DVC pointer; it does not
upload the 77 MiB ONNX file. Teammates run `dvc pull` after pulling Git to
restore the runtime locally.

## Reproducibility guardrails

- Do not overwrite or train into an existing candidate run name unless an
  explicit recovery uses `--overwrite`.
- Keep the current dataset freeze unchanged. New images enter a separate intake
  and review process, then a newly versioned release.
- Keep validation-only selection and run the locked test exactly once per
  selected candidate.
- Promote only a candidate with a locked-test report and passed ONNX/PyTorch
  parity.
