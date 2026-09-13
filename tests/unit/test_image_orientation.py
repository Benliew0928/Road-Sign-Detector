from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from roadsign_assist.inference.engine import decode_image


def _oriented_png(orientation: int) -> bytes:
    pixels = np.zeros((3, 4, 3), dtype=np.uint8)
    pixels[:2, :2] = (255, 0, 0)
    pixels[:2, 2:] = (0, 255, 0)
    pixels[2:, :2] = (0, 0, 255)
    pixels[2:, 2:] = (255, 255, 0)
    image = Image.fromarray(pixels, mode="RGB")
    exif = Image.Exif()
    exif[274] = orientation
    output = BytesIO()
    image.save(output, format="PNG", exif=exif)
    return output.getvalue()


@pytest.mark.parametrize(
    ("orientation", "expected_shape", "expected_top_left_bgr"),
    [
        (1, (3, 4), (0, 0, 255)),
        (3, (3, 4), (0, 255, 255)),
        (6, (4, 3), (255, 0, 0)),
        (8, (4, 3), (0, 255, 0)),
    ],
)
def test_decode_image_applies_exif_orientation(
    orientation: int,
    expected_shape: tuple[int, int],
    expected_top_left_bgr: tuple[int, int, int],
) -> None:
    decoded = decode_image(_oriented_png(orientation))

    assert decoded.shape[:2] == expected_shape
    assert tuple(int(value) for value in decoded[0, 0]) == expected_top_left_bgr


def test_decode_image_rejects_invalid_bytes() -> None:
    with pytest.raises(ValueError, match="not a supported image"):
        decode_image(b"not-an-image")
