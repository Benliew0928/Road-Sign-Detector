import numpy as np

from roadsign_assist.evaluation.detector import greedy_box_matches, percentile


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
