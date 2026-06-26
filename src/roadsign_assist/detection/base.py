from __future__ import annotations

from typing import Protocol

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.inference.models import DetectionModel


class SignDetector(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def available(self) -> bool: ...

    def warmup(self) -> bool: ...

    def detect(self, image: UInt8Image) -> list[DetectionModel]: ...
