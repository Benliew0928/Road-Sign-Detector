# pyright: reportPrivateUsage=false

from pathlib import Path

import numpy as np

from roadsign_assist.evaluation.detector import (
    _read_ground_truth_boxes,
    greedy_box_match_pairs,
    greedy_box_matches,
    percentile,
)


def test_percentile_uses_numeric_values() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0


def test_greedy_box_matching_keeps_predictions_unique() -> None:
    ground_truth = [
        np.asarray([0, 0, 10, 10], dtype=np.float64),
        np.asarray([20, 20, 30, 30], dtype=np.float64),
    ]
    predictions = np.asarray(
        [[0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 9, 9]],
        dtype=np.float64,
    )
    assert greedy_box_matches(ground_truth, predictions, 0.5) == {0, 1}
    assert greedy_box_match_pairs(np.asarray(ground_truth), predictions, 0.5) == [(1, 1), (0, 0)]


def test_detection_boxes_and_empty_negative_labels_are_supported(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    boxes = _read_ground_truth_boxes(label, 200, 100)
    assert len(boxes) == 1
    assert boxes[0].tolist() == [80.0, 30.0, 120.0, 70.0]

    label.write_text("", encoding="utf-8")
    assert _read_ground_truth_boxes(label, 200, 100) == []
