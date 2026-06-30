from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.inference.models import BoundingBoxModel, DetectionModel
from roadsign_assist.paths import project_path
from roadsign_assist.tracking.iou_tracker import TrackState


@dataclass(frozen=True)
class TrackerAvailability:
    available: bool
    reason: str | None = None


class _DetectionResults:
    def __init__(
        self,
        xyxy: np.ndarray,
        confidence: np.ndarray,
        classes: np.ndarray,
    ) -> None:
        self.xyxy = xyxy.astype(np.float32, copy=False)
        self.conf = confidence.astype(np.float32, copy=False)
        self.cls = classes.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    @property
    def xywh(self) -> np.ndarray:
        if len(self) == 0:
            return np.empty((0, 4), dtype=np.float32)
        xywh = self.xyxy.copy()
        xywh[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        xywh[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        xywh[:, 0] = self.xyxy[:, 0] + xywh[:, 2] / 2.0
        xywh[:, 1] = self.xyxy[:, 1] + xywh[:, 3] / 2.0
        return xywh

    def __getitem__(self, index: Any) -> _DetectionResults:
        xyxy = np.asarray(self.xyxy[index], dtype=np.float32)
        confidence = np.asarray(self.conf[index], dtype=np.float32)
        classes = np.asarray(self.cls[index], dtype=np.float32)
        if xyxy.ndim == 1:
            xyxy = xyxy.reshape(1, 4)
            confidence = confidence.reshape(1)
            classes = classes.reshape(1)
        return _DetectionResults(xyxy, confidence, classes)


class BotSortTracker:
    def __init__(
        self,
        *,
        tracker_config_path: str | Path,
        min_stable_frames: int = 3,
        max_missed_frames: int = 12,
    ) -> None:
        self.tracker_config_path = project_path(tracker_config_path)
        self.min_stable_frames = min_stable_frames
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, TrackState] = {}
        self._missed_ids: set[int] = set()
        self._tracker: Any | None = None
        self.availability = self._build_tracker()

    @property
    def name(self) -> str:
        return "botsort+gmc" if self.availability.available else "botsort-unavailable"

    @property
    def tracks(self) -> tuple[TrackState, ...]:
        return tuple(self._tracks.values())

    def _build_tracker(self) -> TrackerAvailability:
        try:
            os.environ.setdefault("YOLO_AUTOINSTALL", "false")
            from ultralytics.trackers.bot_sort import BOTSORT
        except Exception as exc:
            return TrackerAvailability(False, str(exc))

        try:
            with self.tracker_config_path.open("r", encoding="utf-8") as handle:
                settings = yaml.safe_load(handle) or {}
            self._tracker = BOTSORT(args=SimpleNamespace(**settings))
        except Exception as exc:
            return TrackerAvailability(False, str(exc))
        return TrackerAvailability(True)

    def update(
        self,
        detections: list[DetectionModel],
        image: UInt8Image | None = None,
    ) -> list[tuple[DetectionModel, TrackState]]:
        if self._tracker is None or not self.availability.available:
            raise RuntimeError(self.availability.reason or "BoT-SORT tracker is unavailable")
        if image is None:
            raise ValueError("BoT-SORT tracking requires the current video frame")

        results = _detections_to_results(detections)
        rows = np.asarray(self._tracker.update(results, image), dtype=np.float32)
        active_track_ids: set[int] = set()
        assignments: list[tuple[DetectionModel, TrackState]] = []

        for row in rows:
            if row.size < 8:
                continue
            detection_index = int(row[-1])
            if detection_index < 0 or detection_index >= len(detections):
                continue
            track_id = int(row[4])
            bbox = BoundingBoxModel(
                x1=float(row[0]),
                y1=float(row[1]),
                x2=float(row[2]),
                y2=float(row[3]),
            )
            track = self._tracks.get(track_id)
            if track is None:
                track = TrackState(track_id=track_id, bbox=bbox)
                self._tracks[track_id] = track
            else:
                _update_track_state(track, bbox)
            active_track_ids.add(track_id)
            assignments.append((detections[detection_index], track))

        for track_id, track in tuple(self._tracks.items()):
            track.age += 1
            if track_id in active_track_ids:
                continue
            track.missed += 1
            if track.missed > self.max_missed_frames:
                del self._tracks[track_id]
        self._missed_ids = set(self._tracks) - active_track_ids
        return assignments

    def is_stable(self, track: TrackState) -> bool:
        return track.hits >= self.min_stable_frames and track.missed == 0


def _detections_to_results(detections: list[DetectionModel]) -> _DetectionResults:
    if not detections:
        return _DetectionResults(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    boxes = np.asarray(
        [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
        dtype=np.float32,
    )
    confidence = np.asarray([d.confidence for d in detections], dtype=np.float32)
    classes = np.zeros((len(detections),), dtype=np.float32)
    return _DetectionResults(boxes, confidence, classes)


def _update_track_state(track: TrackState, bbox: BoundingBoxModel) -> None:
    elapsed_frames = track.missed + 1
    previous_center_x = (track.bbox.x1 + track.bbox.x2) / 2.0
    previous_center_y = (track.bbox.y1 + track.bbox.y2) / 2.0
    current_center_x = (bbox.x1 + bbox.x2) / 2.0
    current_center_y = (bbox.y1 + bbox.y2) / 2.0
    measured_velocity_x = (current_center_x - previous_center_x) / elapsed_frames
    measured_velocity_y = (current_center_y - previous_center_y) / elapsed_frames
    smoothing = 0.65 if track.hits > 1 else 0.0
    track.velocity_x = smoothing * track.velocity_x + (1.0 - smoothing) * measured_velocity_x
    track.velocity_y = smoothing * track.velocity_y + (1.0 - smoothing) * measured_velocity_y
    track.bbox = bbox
    track.hits += 1
    track.missed = 0
