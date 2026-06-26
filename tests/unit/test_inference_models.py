import pytest
from pydantic import ValidationError

from roadsign_assist.inference.models import BoundingBoxModel


def test_bounding_box_iou() -> None:
    first = BoundingBoxModel(x1=0, y1=0, x2=10, y2=10)
    second = BoundingBoxModel(x1=5, y1=5, x2=15, y2=15)
    assert abs(first.iou(second) - 25 / 175) < 1e-9


def test_invalid_bounding_box_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BoundingBoxModel(x1=10, y1=0, x2=5, y2=10)
