from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EmbeddingDecision:
    accepted: bool
    distance: float
    nearest_label: str
    unknown_score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingGate:
    labels: tuple[str, ...]
    prototypes: np.ndarray[Any, np.dtype[np.float32]]
    distance_threshold: float
    require_label_agreement: bool = True

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EmbeddingGate:
        labels = tuple(str(value) for value in payload.get("prototype_labels", []))
        prototypes = np.asarray(payload.get("prototypes", []), dtype=np.float32)
        if prototypes.ndim != 2 or prototypes.shape[0] != len(labels):
            raise ValueError("Embedding prototype labels and rows do not match")
        if not labels or prototypes.shape[1] == 0:
            raise ValueError("Embedding calibration has no prototypes")
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
        if np.any(norms <= 1e-8):
            raise ValueError("Embedding calibration contains a zero-length prototype")
        return cls(
            labels=labels,
            prototypes=(prototypes / norms).astype(np.float32),
            distance_threshold=float(payload["distance_threshold"]),
            require_label_agreement=bool(payload.get("require_label_agreement", True)),
        )

    def decide(self, embedding: np.ndarray[Any, Any], predicted_label: str) -> EmbeddingDecision:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.prototypes.shape[1]:
            raise ValueError("Classifier embedding dimension does not match calibration prototypes")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return EmbeddingDecision(
                accepted=False,
                distance=2.0,
                nearest_label="unknown_sign",
                unknown_score=1.0,
                reasons=("zero_embedding",),
            )
        normalized = vector / norm
        similarities = self.prototypes @ normalized
        nearest_index = int(np.argmax(similarities))
        nearest_label = self.labels[nearest_index]
        distance = float(np.clip(1.0 - similarities[nearest_index], 0.0, 2.0))
        reasons: list[str] = []
        if distance > self.distance_threshold:
            reasons.append("embedding_distance")
        if self.require_label_agreement and nearest_label != predicted_label:
            reasons.append("prototype_label_disagreement")
        distance_score = min(
            1.0,
            distance / max(2.0 * self.distance_threshold, 1e-6),
        )
        return EmbeddingDecision(
            accepted=not reasons,
            distance=distance,
            nearest_label=nearest_label,
            unknown_score=distance_score,
            reasons=tuple(reasons),
        )
