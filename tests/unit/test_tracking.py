from roadsign_assist.inference.models import BoundingBoxModel, DetectionModel
from roadsign_assist.tracking.iou_tracker import IoUTracker


def _detection(x: float) -> DetectionModel:
    return DetectionModel(
        detection_id=f"d-{x}",
        bbox=BoundingBoxModel(x1=x, y1=10, x2=x + 40, y2=50),
        confidence=0.9,
        detector="test",
    )


def test_tracker_preserves_identity_and_reaches_stability() -> None:
    tracker = IoUTracker(min_stable_frames=3)
    first = tracker.update([_detection(10)])[0][1]
    second = tracker.update([_detection(12)])[0][1]
    third = tracker.update([_detection(14)])[0][1]
    assert first.track_id == second.track_id == third.track_id
    assert tracker.is_stable(third)


def test_tracker_preserves_multiple_independent_tracks() -> None:
    tracker = IoUTracker(min_stable_frames=2)
    first = tracker.update([_detection(10), _detection(180)])
    second = tracker.update([_detection(184), _detection(14)])
    assert first[0][1].track_id == second[1][1].track_id
    assert first[1][1].track_id == second[0][1].track_id
    assert len({assignment[1].track_id for assignment in second}) == 2


def test_tracker_rebinds_after_short_occlusion_and_expires_stale_track() -> None:
    tracker = IoUTracker(max_missed_frames=2)
    original = tracker.update([_detection(20)])[0][1]
    tracker.update([])
    rebound = tracker.update([_detection(24)])[0][1]
    assert rebound.track_id == original.track_id

    tracker.update([])
    tracker.update([])
    tracker.update([])
    replacement = tracker.update([_detection(24)])[0][1]
    assert replacement.track_id != original.track_id


def test_tracker_motion_prediction_handles_large_frame_to_frame_shift() -> None:
    tracker = IoUTracker(match_iou=0.30, max_center_distance=2.0)
    first = tracker.update([_detection(10)])[0][1]
    tracker.update([_detection(30)])
    tracker.update([])
    rebound = tracker.update([_detection(67)])[0][1]
    assert rebound.track_id == first.track_id
