import numpy as np
import pytest

from roadsign_assist.datasets.emtd_masks import (
    polygon_area,
    polygon_box_iou,
    validate_mask_polygon,
)


def test_mask_polygon_geometry_passes_reasonable_box() -> None:
    polygon = np.asarray(
        [[12, 12], [58, 12], [58, 58], [12, 58]],
        dtype=np.float32,
    )
    assert abs(polygon_area(polygon) - 2116.0) < 1e-6
    assert polygon_box_iou(polygon, (10, 10, 60, 60)) > 0.80
    metrics = validate_mask_polygon(
        polygon,
        (10, 10, 60, 60),
        100,
        100,
        minimum_box_iou=0.2,
        minimum_area_ratio=0.04,
        maximum_area_ratio=1.5,
    )
    assert metrics["area_ratio"] > 0.8


def test_mask_polygon_rejects_out_of_bounds_points() -> None:
    polygon = np.asarray(
        [[-1, 10], [50, 10], [50, 50], [10, 50]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="bounds"):
        validate_mask_polygon(
            polygon,
            (10, 10, 60, 60),
            100,
            100,
            minimum_box_iou=0.2,
            minimum_area_ratio=0.04,
            maximum_area_ratio=1.5,
        )
