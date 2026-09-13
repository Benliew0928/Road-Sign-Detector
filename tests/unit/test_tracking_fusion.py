from roadsign_assist.inference.models import BoundingBoxModel, ClassificationModel
from roadsign_assist.tracking.fusion import update_semantic_scores
from roadsign_assist.tracking.iou_tracker import TrackState


def _track() -> TrackState:
    return TrackState(
        track_id=1,
        bbox=BoundingBoxModel(x1=0, y1=0, x2=10, y2=10),
    )


def _prediction(
    label: str,
    confidence: float,
    *,
    accepted: bool = True,
    unknown_score: float | None = None,
) -> ClassificationModel:
    return ClassificationModel(
        semantic_sign_id=label if accepted else "unknown_sign",
        confidence=confidence,
        accepted=accepted,
        model_name="test",
        top_k=[(label, confidence)],
        unknown_score=unknown_score,
    )


def test_first_prediction_keeps_its_calibrated_confidence() -> None:
    label, confidence = update_semantic_scores(_track(), _prediction("stop", 0.70))

    assert label == "stop"
    assert abs(confidence - 0.70) < 1e-12


def test_repeated_predictions_do_not_inflate_confidence() -> None:
    track = _track()
    result = ("unknown_sign", 0.0)

    for _ in range(5):
        result = update_semantic_scores(track, _prediction("stop", 0.70))

    label, confidence = result
    assert label == "stop"
    assert abs(confidence - 0.70) < 1e-12


def test_conflicting_predictions_reduce_semantic_confidence() -> None:
    track = _track()
    update_semantic_scores(track, _prediction("stop", 0.90))

    label, confidence = update_semantic_scores(track, _prediction("give_way", 0.80))

    assert label == "give_way"
    assert confidence < 0.80


def test_rejected_prediction_uses_unknown_score() -> None:
    label, confidence = update_semantic_scores(
        _track(),
        _prediction("turn_left", 0.95, accepted=False, unknown_score=0.82),
    )

    assert label == "unknown_sign"
    assert abs(confidence - 0.82) < 1e-12


def test_rejection_immediately_fails_closed_after_an_accepted_prediction() -> None:
    track = _track()
    update_semantic_scores(track, _prediction("turn_left", 0.95))

    label, confidence = update_semantic_scores(
        track,
        _prediction("turn_left", 0.99, accepted=False, unknown_score=0.72),
    )

    assert label == "unknown_sign"
    assert confidence < 0.72
