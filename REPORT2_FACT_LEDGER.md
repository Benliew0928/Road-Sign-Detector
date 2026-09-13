# Report 2 fact ledger

This ledger supports the Report 2 draft and marks the evidence boundary used in the document.

| Report claim | Value/status | Evidence artefact |
|---|---:|---|
| Classifier release | 3,368 crops; 78 classes; train 2,441; validation 465; locked test 462 | `data/manifests/classifier_production_78_v3_20260829.csv` |
| Detector release | 9,926 frames; 12,896 boxes; train 6,948; validation 1,489; test 1,489; 120 no-sign frames | `data/manifests/detector_production_assignment_v1_20260829.csv` |
| Coursework evaluation | 84 images; 13 Maximum-speed images | `data/manifests/coursework_manifest.csv` |
| Selected classifier | EfficientNetV2-M, 320 px; validation macro-F1 0.9767748 | `outputs/training/phase_b_v3_selection.json` |
| Classifier locked-test accuracy | 0.9502165 (439 / 462) | `outputs/training/pb_v3_effnetv2m_320_s2513/metrics.json` |
| Classifier locked-test macro-F1 | 0.9410565 | `outputs/training/pb_v3_effnetv2m_320_s2513/metrics.json` |
| Classifier threshold | 0.6960521 | `outputs/training/pb_v3_effnetv2m_320_s2513/metrics.json` |
| Classifier ONNX parity | Passed | `outputs/training/pb_v3_effnetv2m_320_s2513/metrics.json` |
| Classifier ONNX mean latency | 30.16 ms | `outputs/training/pb_v3_effnetv2m_320_s2513/metrics.json` |
| Crop padding | 0.06 of box width/height | `src/roadsign_assist/inference/engine.py` |
| OCR confidence threshold | 0.65 | `src/roadsign_assist/semantics/rules.py` |
| Valid Maximum-speed range | 5–160 km/h | `src/roadsign_assist/semantics/rules.py` |
| Detector selection and locked metrics | Pending Phase D completion | Required final evidence: saved Phase D selection and locked-test outputs |
| `inputFiles.txt` batch reader | Pending implementation and executable evidence | Required final evidence: source, acceptance logs, CSV/JSON results |
| 84-image semantic/numeric result | Pending final primary-association evaluation | Required final evidence: frozen-configuration CSV/JSON audit |
| Laboratory runtime | Pending laboratory benchmark | Required final evidence: per-image timing log and environment record |

The 61/84 development diagnostic is deliberately excluded from final accuracy reporting: it used a permissive event-presence rule and preceded the final detector selection.
