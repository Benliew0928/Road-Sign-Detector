from __future__ import annotations

from typing import Any

from roadsign_assist.evaluation.recovery_incident import (
    classify_failure_locations,
    compare_direct_and_website,
    compare_model_status,
    normalized_thresholds,
)
from roadsign_assist.inference.models import BoundingBoxModel, FrameResultModel, InferenceMode


def _detection(identifier: str, confidence: float, box: list[float]) -> dict[str, object]:
    return {
        "detection_id": identifier,
        "bbox": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
        "mask": None,
        "confidence": confidence,
        "detector": "test",
        "color_hint": None,
        "shape_hint": None,
    }


def test_thresholds_include_deployed_value_once_and_are_sorted() -> None:
    assert normalized_thresholds(0.20, [0.20, 0.01, 0.10]) == [0.01, 0.10, 0.20]


def test_no_overlap_at_lowest_threshold_is_detector_representation_failure() -> None:
    locations = classify_failure_locations(
        expected_label="bend_left",
        human_box=BoundingBoxModel(x1=10, y1=10, x2=50, y2=50),
        deployed_threshold=0.37,
        sweep={
            "0.010000": [_detection("deep-0", 0.2, [60, 60, 90, 90])],
            "0.370000": [],
        },
        proposal_classifications={},
        tracking_results=[],
        website_comparison=None,
    )
    assert locations == ["detector_representation"]


def test_low_only_overlap_is_detector_calibration_failure() -> None:
    locations = classify_failure_locations(
        expected_label="bend_left",
        human_box=BoundingBoxModel(x1=10, y1=10, x2=50, y2=50),
        deployed_threshold=0.37,
        sweep={
            "0.010000": [_detection("deep-0", 0.2, [10, 10, 50, 50])],
            "0.370000": [],
        },
        proposal_classifications={},
        tracking_results=[],
        website_comparison=None,
    )
    assert locations == ["detector_calibration"]


def test_wrong_deployed_crop_and_website_mismatch_are_both_localized() -> None:
    detection = _detection("deep-0", 0.8, [10, 10, 50, 50])
    locations = classify_failure_locations(
        expected_label="bend_left",
        human_box=BoundingBoxModel(x1=10, y1=10, x2=50, y2=50),
        deployed_threshold=0.37,
        sweep={"0.010000": [detection], "0.370000": [detection]},
        proposal_classifications={
            "deep-0": {
                "prediction": {
                    "semantic_sign_id": "bend_right",
                    "accepted": True,
                }
            }
        },
        tracking_results=[{"events": []}],
        website_comparison={"matches": False},
    )
    assert locations == ["classifier", "runtime_integration"]


def test_correct_deployed_crop_without_temporal_event_is_tracking_failure() -> None:
    detection = _detection("deep-0", 0.8, [10, 10, 50, 50])
    locations = classify_failure_locations(
        expected_label="bend_left",
        human_box=BoundingBoxModel(x1=10, y1=10, x2=50, y2=50),
        deployed_threshold=0.37,
        sweep={"0.010000": [detection], "0.370000": [detection]},
        proposal_classifications={
            "deep-0": {
                "prediction": {
                    "semantic_sign_id": "bend_left",
                    "accepted": True,
                }
            }
        },
        tracking_results=[{"events": []}],
        website_comparison={"matches": True},
    )
    assert locations == ["tracking_advisory"]


def test_human_crop_failure_is_reported_alongside_detector_calibration() -> None:
    detection = _detection("deep-0", 0.2, [10, 10, 50, 50])
    wrong: dict[str, object] = {
        "prediction": {
            "semantic_sign_id": "bend_left",
            "accepted": True,
        }
    }
    locations = classify_failure_locations(
        expected_label="bend_right",
        human_box=BoundingBoxModel(x1=10, y1=10, x2=50, y2=50),
        deployed_threshold=0.37,
        sweep={"0.010000": [detection], "0.370000": []},
        proposal_classifications={"deep-0": wrong},
        tracking_results=[],
        website_comparison=None,
        human_classification=wrong,
    )
    assert locations == ["detector_calibration", "classifier"]


def test_direct_website_comparison_ignores_latency_but_checks_visible_output() -> None:
    direct = FrameResultModel(
        frame_id=0,
        width=100,
        height=100,
        mode=InferenceMode.DEEP,
        latency_ms=100.0,
        events=[],
    )
    website: dict[str, Any] = {
        **direct.model_dump(mode="json"),
        "latency_ms": 999.0,
    }
    assert compare_direct_and_website(direct, website)["matches"] is True


def test_model_status_comparison_detects_wrong_website_threshold() -> None:
    detector_profile: dict[str, object] = {
        "model_path": "detector.pt",
        "artifact_sha256": "abc",
        "image_size": 640,
        "confidence_threshold": 0.37,
    }
    direct: dict[str, object] = {
        "mode": "deep",
        "detector_profile": detector_profile,
        "classifier_profile": {
            "model_path": "classifier.onnx",
            "artifact_sha256": "def",
            "image_size": 320,
            "confidence_threshold": 0.69,
        },
    }
    website = {
        **direct,
        "detector_profile": {
            **detector_profile,
            "confidence_threshold": 0.25,
        },
    }

    comparison = compare_model_status(direct, website)

    assert comparison["matches"] is False
    fields = comparison["fields"]
    assert isinstance(fields, dict)
    assert fields["detector_confidence_threshold"]["matches"] is False
