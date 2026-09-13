from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from roadsign_assist.baseline.models import UInt8Image

PREPROCESSING_VERSION = "roadsign_raw_bgr_v1"
DECODE_POLICY = (
    "Pillow decode; EXIF transpose; RGB conversion; contiguous uint8 OpenCV BGR output"
)


def decode_image_bytes(data: bytes) -> UInt8Image:
    """Apply the canonical raw-image decode and orientation policy."""
    try:
        with Image.open(BytesIO(data)) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            rgb = np.asarray(oriented, dtype=np.uint8)
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
        ModuleNotFoundError,
    ) as exc:
        raise ValueError("The supplied bytes are not a supported image") from exc
    return cast(UInt8Image, np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))


def load_image_file(path: str | Path) -> UInt8Image:
    """Read a file as bytes before canonical decoding, matching the upload path."""
    return decode_image_bytes(Path(path).read_bytes())

