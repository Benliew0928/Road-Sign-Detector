from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roadsign_assist.classification.preprocessing import (
    build_evaluation_transform,
    preprocess_classifier_numpy,
)
from roadsign_assist.inference.preprocessing import decode_image_bytes, load_image_file


def _encoded_test_image() -> bytes:
    y, x = np.mgrid[:19, :31]
    rgb = np.stack(
        ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256),
        axis=-1,
    ).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def test_file_and_uploaded_bytes_decode_to_identical_bgr(tmp_path: Path) -> None:
    data = _encoded_test_image()
    path = tmp_path / "raw.png"
    path.write_bytes(data)

    uploaded = decode_image_bytes(data)
    direct_file = load_image_file(path)

    assert uploaded.flags.c_contiguous
    assert uploaded.dtype == np.uint8
    assert np.array_equal(uploaded, direct_file)


def test_onnx_numpy_preprocessing_matches_pytorch_evaluation_transform() -> None:
    bgr = decode_image_bytes(_encoded_test_image())
    onnx_tensor = preprocess_classifier_numpy(bgr, 24)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pytorch_tensor = build_evaluation_transform(24)(Image.fromarray(rgb)).numpy()[None]

    assert onnx_tensor.shape == (1, 3, 24, 24)
    assert np.allclose(onnx_tensor, pytorch_tensor, atol=1e-6)
