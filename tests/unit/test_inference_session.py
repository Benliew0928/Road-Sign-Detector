from roadsign_assist.inference.engine import InferenceEngine


def test_new_session_has_independent_tracking_state() -> None:
    shared = InferenceEngine()
    first = shared.new_session()
    second = shared.new_session()
    assert first.detector is second.detector
    assert first.classifier is second.classifier
    assert first.tracker is not second.tracker
    assert first.frame_id == second.frame_id == 0
