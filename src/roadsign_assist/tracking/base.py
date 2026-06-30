from __future__ import annotations

from typing import Protocol

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.inference.models import DetectionModel
from roadsign_assist.tracking.iou_tracker import TrackState


class SignTracker(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def tracks(self) -> tuple[TrackState, ...]: ...

    def update(
        self,
        detections: list[DetectionModel],
        image: UInt8Image | None = None,
    ) -> list[tuple[DetectionModel, TrackState]]: ...

    def is_stable(self, track: TrackState) -> bool: ...
