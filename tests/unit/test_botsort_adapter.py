from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np

from roadsign_assist.inference.models import BoundingBoxModel, DetectionModel
from roadsign_assist.tracking.botsort_adapter import BotSortTracker
from roadsign_assist.tracking.factory import build_tracker
from roadsign_assist.tracking.iou_tracker import IoUTracker


def _detection(x: float) -> DetectionModel:
    return DetectionModel(
        detection_id=f"d-{x}",
        bbox=BoundingBoxModel(x1=x, y1=10, x2=x + 40, y2=50),
        confidence=0.9,
        detector="test",
    )


def _write_tracker_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "tracker_type: botsort",
                "track_high_thresh: 0.25",
                "track_low_thresh: 0.10",
                "new_track_thresh: 0.25",
                "track_buffer: 30",
                "match_thresh: 0.80",
                "fuse_score: true",
                "gmc_method: sparseOptFlow",
                "proximity_thresh: 0.50",
                "appearance_thresh: 0.80",
                "with_reid: false",
                "model: auto",
            ]
        ),
        encoding="utf-8",
    )


def test_botsort_adapter_maps_custom_detections_to_track_state(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    class FakeBOTSORT:
        def __init__(self, args: Any) -> None:
            self.args = args

        def update(self, results: Any, image: np.ndarray) -> np.ndarray:
            assert image.shape == (80, 120, 3)
            assert results.xyxy.shape == (1, 4)
            return np.asarray([[10, 10, 50, 50, 7, 0.9, 0, 0]], dtype=np.float32)

    module = types.ModuleType("ultralytics.trackers.bot_sort")
    cast(Any, module).BOTSORT = FakeBOTSORT
    ultralytics = types.ModuleType("ultralytics")
    trackers = types.ModuleType("ultralytics.trackers")
    cast(Any, ultralytics).trackers = trackers
    cast(Any, trackers).bot_sort = module
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.trackers", trackers)
    monkeypatch.setitem(sys.modules, "ultralytics.trackers.bot_sort", module)

    config_path = tmp_path / "botsort.yaml"
    _write_tracker_config(config_path)
    tracker = BotSortTracker(tracker_config_path=config_path)
    assignments = tracker.update([_detection(10)], image=np.zeros((80, 120, 3), dtype=np.uint8))

    assert tracker.availability.available is True
    assert assignments[0][1].track_id == 7
    assert assignments[0][1].bbox == BoundingBoxModel(x1=10, y1=10, x2=50, y2=50)


def test_tracker_factory_auto_falls_back_to_iou_when_botsort_is_unavailable() -> None:
    tracker = build_tracker(
        {
            "backend": "auto",
            "tracker_config_path": "missing.yaml",
            "gmc_method": "sparseOptFlow",
            "match_iou": 0.30,
            "max_center_distance": 2.0,
            "min_stable_frames": 3,
            "max_missed_frames": 12,
        }
    )

    assert isinstance(tracker, IoUTracker)
    assert tracker.name == "iou+sparseOptFlow-gmc"
