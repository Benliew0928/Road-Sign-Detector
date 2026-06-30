from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np

from roadsign_assist.baseline.models import UInt8Image


@dataclass(frozen=True)
class CameraMotion:
    dx: float = 0.0
    dy: float = 0.0
    confidence: float = 0.0
    method: str = "none"


class GlobalMotionCompensator:
    def __init__(self, *, method: str = "none", downscale: int = 2) -> None:
        self.method = method
        self.downscale = max(1, downscale)
        self._previous_gray: np.ndarray | None = None

    @staticmethod
    def identity() -> CameraMotion:
        return CameraMotion()

    def reset(self) -> None:
        self._previous_gray = None

    def estimate(self, image: UInt8Image) -> CameraMotion:
        if self.method in {"none", "None", ""}:
            return self.identity()
        gray = self._prepare_gray(image)
        if self._previous_gray is None:
            self._previous_gray = gray
            return CameraMotion(method=self.method)

        if self.method != "sparseOptFlow":
            self._previous_gray = gray
            return CameraMotion(method=self.method)

        motion = self._sparse_optical_flow(gray)
        self._previous_gray = gray
        return motion

    def _prepare_gray(self, image: UInt8Image) -> np.ndarray:
        gray = cast(np.ndarray, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) if image.ndim == 3 else image
        if self.downscale > 1:
            height, width = gray.shape[:2]
            gray = cast(
                np.ndarray,
                cv2.resize(
                    gray,
                    (max(1, width // self.downscale), max(1, height // self.downscale)),
                    interpolation=cv2.INTER_AREA,
                ),
            )
        return gray

    def _sparse_optical_flow(self, gray: np.ndarray) -> CameraMotion:
        assert self._previous_gray is not None
        previous_points = cast(
            np.ndarray | None,
            cv2.goodFeaturesToTrack(
                self._previous_gray,
                maxCorners=350,
                qualityLevel=0.01,
                minDistance=5,
                blockSize=5,
            ),
        )
        if previous_points is None or len(previous_points) < 6:
            return CameraMotion(method=self.method)

        current_points, status, _ = cast(
            tuple[np.ndarray | None, np.ndarray | None, Any],
            cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                gray,
                previous_points,
                np.empty_like(previous_points),
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            ),
        )
        if current_points is None or status is None:
            return CameraMotion(method=self.method)

        valid = status.reshape(-1) == 1
        previous = previous_points.reshape(-1, 2)[valid]
        current = current_points.reshape(-1, 2)[valid]
        if len(previous) < 6:
            return CameraMotion(method=self.method)

        transform, inliers = cast(
            tuple[np.ndarray | None, np.ndarray | None],
            cv2.estimateAffinePartial2D(
                previous,
                current,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.98,
            ),
        )
        if transform is not None:
            dx = float(transform[0, 2]) * self.downscale
            dy = float(transform[1, 2]) * self.downscale
            confidence = (
                float(np.count_nonzero(inliers)) / float(len(previous))
                if inliers is not None
                else 0.0
            )
        else:
            displacement = current - previous
            median = np.median(displacement, axis=0)
            dx = float(median[0]) * self.downscale
            dy = float(median[1]) * self.downscale
            confidence = min(1.0, float(len(previous)) / 80.0)

        max_reasonable = max(gray.shape[:2]) * self.downscale * 0.25
        if abs(dx) > max_reasonable or abs(dy) > max_reasonable:
            return CameraMotion(method=self.method)
        return CameraMotion(dx=dx, dy=dy, confidence=confidence, method=self.method)
