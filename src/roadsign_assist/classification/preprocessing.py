from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from typing import Any

import cv2
import numpy as np
from PIL import Image

from roadsign_assist.baseline.models import UInt8Image

CLASSIFIER_PREPROCESSING_VERSION = "classifier_imagenet_bilinear_v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_classifier_numpy(
    crop: UInt8Image,
    image_size: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Match the PIL/torchvision evaluation path for ONNX inference."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = Image.fromarray(rgb).resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )
    tensor = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    tensor = (tensor - mean) / std
    return np.transpose(tensor, (2, 0, 1))[None].astype(np.float32)


def build_evaluation_transform(image_size: int) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

