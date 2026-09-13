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
    track.semantic_weight = track.semantic_weight * decay + 1.0
    label = prediction.semantic_sign_id if prediction.accepted else "unknown_sign"
    contribution = prediction.confidence
    if label == "unknown_sign" and prediction.unknown_score is not None:
        contribution = prediction.unknown_score
    track.semantic_scores[label] = track.semantic_scores.get(label, 0.0) + contribution
    if not prediction.accepted:
        unknown_confidence = track.semantic_scores["unknown_sign"] / track.semantic_weight
        return "unknown_sign", min(1.0, max(0.0, unknown_confidence))
    best_label, best_score = max(track.semantic_scores.items(), key=lambda item: item[1])
    confidence = best_score / track.semantic_weight if track.semantic_weight else 0.0
    return best_label, min(1.0, max(0.0, confidence))
