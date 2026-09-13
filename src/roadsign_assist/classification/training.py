from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import csv
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from roadsign_assist.paths import project_path

Architecture = Literal[
    "mobilenet_v3_large",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "convnext_tiny",
]

_PRETRAINED_WEIGHT_DOWNLOAD_ATTEMPTS = 4


def _build_with_pretrained_weight_retry(
    factory: Callable[..., Any],
    *,
    architecture: Architecture,
    weights: Any,
    pretrained: bool,
) -> Any:
    """Build a torchvision model, retrying transient pretrained-weight downloads."""
    if not pretrained:
        return factory(weights=None)
    for attempt in range(1, _PRETRAINED_WEIGHT_DOWNLOAD_ATTEMPTS + 1):
        try:
            return factory(weights=weights)
        except OSError as error:
            if attempt == _PRETRAINED_WEIGHT_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(
                    "Unable to download or load ImageNet weights for "
                    f"{architecture} after {attempt} attempts. Check the network connection "
                    "and rerun the same Phase-B stage."
                ) from error
            wait_seconds = attempt * 5
            print(
                "Pretrained weights for "
                f"{architecture} were temporarily unavailable ({error}); "
                f"retrying in {wait_seconds}s ({attempt}/{_PRETRAINED_WEIGHT_DOWNLOAD_ATTEMPTS})."
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class ClassifierTrainingConfig:
    train_manifest: Path
    validation_manifest: Path
    architecture: Architecture = "efficientnet_v2_s"
    image_size: int = 224
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    seed: int = 2513


def read_label_vocabulary(*manifests: Path) -> list[str]:
    labels: set[str] = set()
    for manifest in manifests:
        with project_path(manifest).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label = row.get("semantic_sign_id", "").strip()
                if label:
                    labels.add(label)
    if not labels:
        raise ValueError("No semantic labels were found in classifier manifests")
    return sorted(labels)


def build_torchvision_model(
    architecture: Architecture,
    class_count: int,
    *,
    pretrained: bool = True,
) -> Any:
    import torch.nn as nn
    from torchvision.models import (
        ConvNeXt_Tiny_Weights,
        EfficientNet_V2_M_Weights,
        EfficientNet_V2_S_Weights,
        MobileNet_V3_Large_Weights,
        convnext_tiny,
        efficientnet_v2_m,
        efficientnet_v2_s,
        mobilenet_v3_large,
    )

    if architecture == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = _build_with_pretrained_weight_retry(
            efficientnet_v2_s,
            architecture=architecture,
            weights=weights,
            pretrained=pretrained,
        )
        input_features = cast(int, model.classifier[-1].in_features)
        model.classifier[-1] = nn.Linear(input_features, class_count)
        return model
    if architecture == "efficientnet_v2_m":
        weights = EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
        model = _build_with_pretrained_weight_retry(
            efficientnet_v2_m,
            architecture=architecture,
            weights=weights,
            pretrained=pretrained,
        )
        input_features = cast(int, model.classifier[-1].in_features)
        model.classifier[-1] = nn.Linear(input_features, class_count)
        return model
    if architecture == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = _build_with_pretrained_weight_retry(
            convnext_tiny,
            architecture=architecture,
            weights=weights,
            pretrained=pretrained,
        )
        input_features = cast(int, model.classifier[-1].in_features)
        model.classifier[-1] = nn.Linear(input_features, class_count)
        return model
    if architecture == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = _build_with_pretrained_weight_retry(
            mobilenet_v3_large,
            architecture=architecture,
            weights=weights,
            pretrained=pretrained,
        )
        input_features = cast(int, model.classifier[-1].in_features)
        model.classifier[-1] = nn.Linear(input_features, class_count)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")


def classifier_embedding_and_logits(
    model: Any,
    architecture: Architecture,
    image: Any,
) -> tuple[Any, Any]:
    """Return logits plus the pre-classifier feature vector for supported backbones."""
    import torch

    features = model.features(image)
    pooled = model.avgpool(features)
    if architecture == "convnext_tiny":
        # torchvision ConvNeXt applies LayerNorm2d before flattening and its final
        # classifier layer is the linear head replaced in build_torchvision_model.
        normalized = model.classifier[0](pooled)
        embedding = torch.flatten(normalized, 1)
        logits = model.classifier[-1](embedding)
        return embedding, logits
    embedding = torch.flatten(pooled, 1)
    logits = model.classifier(embedding)
    return embedding, logits
