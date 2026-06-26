from __future__ import annotations

from typing import Protocol

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.inference.models import ClassificationModel


class SignClassifier(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def available(self) -> bool: ...

    def warmup(self) -> bool: ...

    def classify(self, crop: UInt8Image) -> ClassificationModel: ...


class UnknownClassifier:
    @property
    def name(self) -> str:
        return "unavailable"

    @property
    def available(self) -> bool:
        return False

    def warmup(self) -> bool:
        return False

    def classify(self, crop: UInt8Image) -> ClassificationModel:
        del crop
        return ClassificationModel(
            semantic_sign_id="unknown_sign",
            confidence=0.0,
            accepted=False,
            model_name=self.name,
            top_k=[],
            unknown_score=1.0,
        )
