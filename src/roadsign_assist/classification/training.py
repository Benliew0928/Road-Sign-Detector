from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from roadsign_assist.paths import project_path

Architecture = Literal["mobilenet_v3_large", "efficientnet_v2_s"]


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
        EfficientNet_V2_S_Weights,
        MobileNet_V3_Large_Weights,
        efficientnet_v2_s,
        mobilenet_v3_large,
    )

    if architecture == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        input_features = cast(int, model.classifier[-1].in_features)
        model.classifier[-1] = nn.Linear(input_features, class_count)
        return model
    if architecture == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        input_features = cast(int, model.classifier[-1].in_features)
        model.classifier[-1] = nn.Linear(input_features, class_count)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")
