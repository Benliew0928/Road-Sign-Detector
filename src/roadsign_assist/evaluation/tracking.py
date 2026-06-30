from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.inference.models import BoundingBoxModel, DetectionModel
from roadsign_assist.paths import project_path
from roadsign_assist.semantics.rules import SemanticRuleEngine
from roadsign_assist.tracking.iou_tracker import IoUTracker, TrackState


def evaluate_tracking_motion(
    output_root: str | Path = "outputs/evaluation/tracking_motion",
    report_path: str | Path = "docs/P11_TRACKING_MOTION_REPORT.md",
) -> dict[str, Any]:
    output_dir = project_path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios: list[dict[str, Any]] = [
        _stationary_visibility(),
        _moving_camera_translation(),
        _short_occlusion_rebind(),
        _motion_blur_translation(),
        _duplicate_warning_cooldown(),
    ]
    report: dict[str, Any] = {
        "tracker": "auto-ready BoT-SORT/GMC with sparseOptFlow IoU fallback",
        "real_footage_collected": False,
        "scenarios": scenarios,
    }
    report["passed"] = all(bool(scenario["passed"]) for scenario in scenarios)
    report["summary"] = {
        "scenario_count": len(scenarios),
        "passed_count": sum(1 for scenario in scenarios if scenario["passed"]),
        "total_id_switches": sum(int(scenario.get("id_switches", 0)) for scenario in scenarios),
        "maximum_stable_frame": max(
            int(scenario.get("first_stable_frame", 0) or 0) for scenario in scenarios
        ),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    project_path(report_path).write_text(_render_markdown(report, summary_path), encoding="utf-8")
    return report


def _stationary_visibility() -> dict[str, Any]:
    tracker = IoUTracker(min_stable_frames=3, gmc_method="sparseOptFlow")
    track_ids: list[int] = []
    stable_frames: list[int] = []
    for frame_index in range(8):
        assignments = tracker.update(
            [_detection("stationary", 88, 58)],
            image=_synthetic_frame(),
        )
        track = assignments[0][1]
        track_ids.append(track.track_id)
        if tracker.is_stable(track):
            stable_frames.append(frame_index)
    return _scenario_result(
        name="stationary_visible_sign",
        track_ids=track_ids,
        stable_frames=stable_frames,
        expected_first_stable_frame=2,
    )


def _moving_camera_translation() -> dict[str, Any]:
    tracker = IoUTracker(
        match_iou=0.55,
        max_center_distance=0.45,
        min_stable_frames=3,
        gmc_method="sparseOptFlow",
    )
    track_ids: list[int] = []
    stable_frames: list[int] = []
    for frame_index, dx in enumerate([0, 18, 36, 54, 72, 90]):
        image = _synthetic_frame(dx=dx)
        assignments = tracker.update(
            [_detection("camera-pan", 80 + dx, 62)],
            image=image,
        )
        track = assignments[0][1]
        track_ids.append(track.track_id)
        if tracker.is_stable(track):
            stable_frames.append(frame_index)
    return _scenario_result(
        name="moving_camera_sparse_optical_flow",
        track_ids=track_ids,
        stable_frames=stable_frames,
        expected_first_stable_frame=2,
    )


def _short_occlusion_rebind() -> dict[str, Any]:
    tracker = IoUTracker(
        match_iou=0.45,
        max_center_distance=1.0,
        max_missed_frames=3,
        min_stable_frames=2,
        gmc_method="sparseOptFlow",
    )
    track_ids: list[int] = []
    stable_frames: list[int] = []
    positions: list[float | None] = [70, 86, None, 118, 134]
    for frame_index, x in enumerate(positions):
        detections = [] if x is None else [_detection("occlusion", x, 64)]
        assignments = tracker.update(detections, image=_synthetic_frame(dx=frame_index * 8))
        if not assignments:
            continue
        track = assignments[0][1]
        track_ids.append(track.track_id)
        if tracker.is_stable(track):
            stable_frames.append(frame_index)
    return _scenario_result(
        name="short_occlusion_reappearance",
        track_ids=track_ids,
        stable_frames=stable_frames,
        expected_first_stable_frame=1,
    )


def _motion_blur_translation() -> dict[str, Any]:
    tracker = IoUTracker(
        match_iou=0.35,
        max_center_distance=1.2,
        min_stable_frames=3,
        gmc_method="sparseOptFlow",
    )
    track_ids: list[int] = []
    stable_frames: list[int] = []
    for frame_index, dx in enumerate([0, 12, 24, 36, 48, 60]):
        image = cast(UInt8Image, cv2.GaussianBlur(_synthetic_frame(dx=dx), (9, 9), 0))
        assignments = tracker.update(
            [_detection("blur", 92 + dx, 66)],
            image=image,
        )
        track = assignments[0][1]
        track_ids.append(track.track_id)
        if tracker.is_stable(track):
            stable_frames.append(frame_index)
    return _scenario_result(
        name="motion_blur_camera_translation",
        track_ids=track_ids,
        stable_frames=stable_frames,
        expected_first_stable_frame=2,
    )


def _duplicate_warning_cooldown() -> dict[str, Any]:
    engine = SemanticRuleEngine(duplicate_warning_seconds=8.0)
    track = TrackState(track_id=1, bbox=BoundingBoxModel(x1=0, y1=0, x2=20, y2=20), hits=3)
    decisions = [
        engine.should_announce(track, stable=False, now=0.0),
        engine.should_announce(track, stable=True, now=1.0),
        engine.should_announce(track, stable=True, now=5.0),
        engine.should_announce(track, stable=True, now=10.0),
    ]
    expected = [False, True, False, True]
    return {
        "name": "duplicate_warning_cooldown",
        "passed": decisions == expected,
        "decisions": decisions,
        "expected": expected,
        "id_switches": 0,
        "first_stable_frame": 0,
    }


def _scenario_result(
    *,
    name: str,
    track_ids: list[int],
    stable_frames: list[int],
    expected_first_stable_frame: int,
) -> dict[str, Any]:
    id_switches = sum(1 for first, second in pairwise(track_ids) if first != second)
    first_stable_frame = stable_frames[0] if stable_frames else None
    return {
        "name": name,
        "passed": bool(
            track_ids
            and id_switches == 0
            and first_stable_frame is not None
            and first_stable_frame <= expected_first_stable_frame
        ),
        "track_ids": track_ids,
        "id_switches": id_switches,
        "stable_frames": stable_frames,
        "first_stable_frame": first_stable_frame,
        "expected_first_stable_frame_or_earlier": expected_first_stable_frame,
    }


def _synthetic_frame(*, dx: int = 0, dy: int = 0) -> UInt8Image:
    rng = np.random.default_rng(2513)
    canvas = np.zeros((180, 260, 3), dtype=np.uint8)
    for _ in range(220):
        x = int(rng.integers(8, 252))
        y = int(rng.integers(8, 172))
        color = int(rng.integers(130, 255))
        cv2.circle(canvas, (x, y), 1, (color, color, color), -1)
    cv2.rectangle(canvas, (80, 62), (122, 104), (0, 0, 220), 2)
    transform = np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cast(
        UInt8Image,
        cv2.warpAffine(canvas, transform, (260, 180), borderMode=cv2.BORDER_REFLECT),
    )


def _detection(name: str, x: float, y: float) -> DetectionModel:
    return DetectionModel(
        detection_id=name,
        bbox=BoundingBoxModel(x1=x, y1=y, x2=x + 42, y2=y + 42),
        confidence=0.90,
        detector="synthetic",
    )


def _render_markdown(report: dict[str, Any], summary_path: Path) -> str:
    rows = "\n".join(
        "| {name} | {passed} | {id_switches} | {first_stable_frame} |".format(
            name=scenario["name"],
            passed="pass" if scenario["passed"] else "fail",
            id_switches=scenario.get("id_switches", 0),
            first_stable_frame=scenario.get("first_stable_frame", ""),
        )
        for scenario in report["scenarios"]
    )
    return f"""# P11 Tracking Motion Report

This report is generated from synthetic motion, occlusion, blur, and cooldown
checks. No real road footage was collected by Codex; owner field footage should
still be used for final demonstration confidence.

Evidence: `{summary_path.as_posix()}`

| Scenario | Result | ID switches | First stable frame |
|---|---:|---:|---:|
{rows}

Overall result: {"pass" if report["passed"] else "fail"}

Implementation notes:

- Inference tracking is configurable and attempts BoT-SORT/GMC when the optional
  tracker dependency is available.
- The fallback tracker now uses sparse optical-flow camera translation
  compensation, global assignment, stale-track expiry, short occlusion rebinding,
  semantic score fusion, stable-frame gating, OCR caching, and per-track warning
  cooldowns.
- Real moving-camera footage remains owner-test evidence, not a claim made by
  this synthetic report.
"""
