from __future__ import annotations

# pyright: reportUnknownVariableType=false
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.inference.models import BoundingBoxModel, DetectionModel
from roadsign_assist.tracking.gmc import GlobalMotionCompensator


@dataclass
class TrackState:
    track_id: int
    bbox: BoundingBoxModel
    hits: int = 1
    missed: int = 0
    age: int = 1
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    semantic_scores: dict[str, float] = field(default_factory=dict)
    last_announced_at: float | None = None

    def predicted_bbox(
        self,
        *,
        camera_motion_x: float = 0.0,
        camera_motion_y: float = 0.0,
    ) -> BoundingBoxModel:
        steps = self.missed + 1
        dx = self.velocity_x * steps + camera_motion_x
        dy = self.velocity_y * steps + camera_motion_y
        return BoundingBoxModel(
            x1=self.bbox.x1 + dx,
            y1=self.bbox.y1 + dy,
            x2=self.bbox.x2 + dx,
            y2=self.bbox.y2 + dy,
        )


class IoUTracker:
    def __init__(
        self,
        *,
        match_iou: float = 0.30,
        max_center_distance: float = 2.0,
        max_missed_frames: int = 12,
        min_stable_frames: int = 3,
        gmc_method: str = "none",
    ) -> None:
        self.match_iou = match_iou
        self.max_center_distance = max_center_distance
        self.max_missed_frames = max_missed_frames
        self.min_stable_frames = min_stable_frames
        self.gmc_method = gmc_method
        self._gmc = GlobalMotionCompensator(method=gmc_method)
        self._next_id = 1
        self._tracks: dict[int, TrackState] = {}

    @property
    def name(self) -> str:
        return f"iou+{self.gmc_method}-gmc" if self.gmc_method != "none" else "iou"

    @property
    def tracks(self) -> tuple[TrackState, ...]:
        return tuple(self._tracks.values())

    def update(
        self,
        detections: list[DetectionModel],
        image: UInt8Image | None = None,
    ) -> list[tuple[DetectionModel, TrackState]]:
        track_ids = sorted(self._tracks)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignment_by_detection: dict[int, TrackState] = {}
        motion = self._gmc.estimate(image) if image is not None else self._gmc.identity()

        if track_ids and detections:
            costs = np.full((len(track_ids), len(detections)), 1e6, dtype=np.float64)
            for track_index, track_id in enumerate(track_ids):
                predicted = self._tracks[track_id].predicted_bbox(
                    camera_motion_x=motion.dx,
                    camera_motion_y=motion.dy,
                )
                for detection_index, detection in enumerate(detections):
                    iou = predicted.iou(detection.bbox)
                    center_distance = _normalized_center_distance(
                        predicted,
                        detection.bbox,
                    )
                    if iou < self.match_iou and center_distance > self.max_center_distance:
                        continue
                    center_score = max(
                        0.0,
                        1.0 - center_distance / self.max_center_distance,
                    )
                    costs[track_index, detection_index] = 1.0 - (0.75 * iou + 0.25 * center_score)
            row_indices_raw, column_indices_raw = linear_sum_assignment(costs)
            row_indices = np.asarray(row_indices_raw, dtype=np.int64)
            column_indices = np.asarray(column_indices_raw, dtype=np.int64)
            for row_value, column_value in zip(
                row_indices,
                column_indices,
                strict=True,
            ):
                row = int(row_value)
                column = int(column_value)
                if costs[row, column] >= 1e5:
                    continue
                track_id = track_ids[row]
                track = self._tracks[track_id]
                _update_track(
                    track,
                    detections[column].bbox,
                    camera_motion_x=motion.dx,
                    camera_motion_y=motion.dy,
                )
                matched_tracks.add(track_id)
                matched_detections.add(column)
                assignment_by_detection[column] = track

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track = TrackState(track_id=self._next_id, bbox=detection.bbox)
            self._tracks[track.track_id] = track
            self._next_id += 1
            assignment_by_detection[detection_index] = track

        for track_id in track_ids:
            track = self._tracks[track_id]
            track.age += 1
            if track_id not in matched_tracks:
                track.missed += 1
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.missed <= self.max_missed_frames
        }
        return [
            (detection, assignment_by_detection[index])
            for index, detection in enumerate(detections)
        ]

    def is_stable(self, track: TrackState) -> bool:
        return track.hits >= self.min_stable_frames and track.missed == 0


def _update_track(
    track: TrackState,
    bbox: BoundingBoxModel,
    *,
    camera_motion_x: float = 0.0,
    camera_motion_y: float = 0.0,
) -> None:
    elapsed_frames = track.missed + 1
    previous_center_x, previous_center_y = _center(track.bbox)
    current_center_x, current_center_y = _center(bbox)
    measured_velocity_x = (
        current_center_x - previous_center_x - camera_motion_x
    ) / elapsed_frames
    measured_velocity_y = (
        current_center_y - previous_center_y - camera_motion_y
    ) / elapsed_frames
    smoothing = 0.65 if track.hits > 1 else 0.0
    track.velocity_x = smoothing * track.velocity_x + (1.0 - smoothing) * measured_velocity_x
    track.velocity_y = smoothing * track.velocity_y + (1.0 - smoothing) * measured_velocity_y
    track.bbox = bbox
    track.hits += 1
    track.missed = 0


def _center(bbox: BoundingBoxModel) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def _normalized_center_distance(
    first: BoundingBoxModel,
    second: BoundingBoxModel,
) -> float:
    first_x, first_y = _center(first)
    second_x, second_y = _center(second)
    distance = ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5
    scale = max(
        1.0,
        (first.width**2 + first.height**2) ** 0.5,
        (second.width**2 + second.height**2) ** 0.5,
    )
    return distance / scale
