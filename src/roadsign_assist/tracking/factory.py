from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from roadsign_assist.tracking.base import SignTracker
from roadsign_assist.tracking.botsort_adapter import BotSortTracker
from roadsign_assist.tracking.iou_tracker import IoUTracker


def build_tracker(settings: Mapping[str, Any]) -> SignTracker:
    backend = str(settings.get("backend", "iou"))
    min_stable_frames = int(settings["min_stable_frames"])
    max_missed_frames = int(settings["max_missed_frames"])
    fallback = _build_iou_tracker(settings)

    if backend in {"auto", "botsort"}:
        botsort = BotSortTracker(
            tracker_config_path=str(
                settings.get("tracker_config_path", "configs/tracking/botsort.yaml")
            ),
            min_stable_frames=min_stable_frames,
            max_missed_frames=max_missed_frames,
        )
        if botsort.availability.available:
            return botsort
        if backend == "botsort":
            raise RuntimeError(botsort.availability.reason or "BoT-SORT tracker unavailable")
    return fallback


def _build_iou_tracker(settings: Mapping[str, Any]) -> IoUTracker:
    return IoUTracker(
        match_iou=float(settings.get("match_iou", 0.30)),
        max_center_distance=float(settings.get("max_center_distance", 2.0)),
        min_stable_frames=int(settings["min_stable_frames"]),
        max_missed_frames=int(settings["max_missed_frames"]),
        gmc_method=str(settings.get("gmc_method", "none")),
    )
