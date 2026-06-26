from __future__ import annotations

from roadsign_assist.inference.models import ClassificationModel
from roadsign_assist.tracking.iou_tracker import TrackState


def update_semantic_scores(
    track: TrackState,
    prediction: ClassificationModel,
    *,
    decay: float = 0.75,
) -> tuple[str, float]:
    track.semantic_scores = {label: score * decay for label, score in track.semantic_scores.items()}
    label = prediction.semantic_sign_id
    track.semantic_scores[label] = track.semantic_scores.get(label, 0.0) + prediction.confidence
    total = sum(track.semantic_scores.values())
    best_label, best_score = max(track.semantic_scores.items(), key=lambda item: item[1])
    return best_label, best_score / total if total else 0.0
