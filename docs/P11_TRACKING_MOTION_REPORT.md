# P11 Tracking Motion Report

This report is generated from synthetic motion, occlusion, blur, and cooldown
checks. No real road footage was collected by Codex; owner field footage should
still be used for final demonstration confidence.

Evidence: `C:/MiniProject/outputs/evaluation/tracking_motion/summary.json`

| Scenario | Result | ID switches | First stable frame |
|---|---:|---:|---:|
| stationary_visible_sign | pass | 0 | 2 |
| moving_camera_sparse_optical_flow | pass | 0 | 2 |
| short_occlusion_reappearance | pass | 0 | 1 |
| motion_blur_camera_translation | pass | 0 | 2 |
| duplicate_warning_cooldown | pass | 0 | 0 |

Overall result: pass

Implementation notes:

- Inference tracking is configurable and attempts BoT-SORT/GMC when the optional
  tracker dependency is available.
- The fallback tracker now uses sparse optical-flow camera translation
  compensation, global assignment, stale-track expiry, short occlusion rebinding,
  semantic score fusion, stable-frame gating, OCR caching, and per-track warning
  cooldowns.
- Real moving-camera footage remains owner-test evidence, not a claim made by
  this synthetic report.
