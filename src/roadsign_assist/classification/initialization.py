"""Initialize a new classifier from local model weights, never optimizer/run state."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def initialize_classifier_weights(
    model: Any,
    checkpoint_path: Path,
    *,
    labels: list[str],
    architecture: str,
    image_size: int,
    allow_label_expansion: bool = False,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_labels = list(checkpoint.get("labels", []))
    config = checkpoint.get("config", {})
    if config.get("architecture") != architecture or config.get("image_size") != image_size:
        raise ValueError("Initialization checkpoint architecture/image size mismatch")
    if checkpoint_labels == labels:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        initialization_mode = "weights_only_initialization_fresh_optimizer_and_schedule"
        added_labels: list[str] = []
    else:
        if not allow_label_expansion:
            raise ValueError("Initialization checkpoint label order does not match the release")
        if not checkpoint_labels or not set(checkpoint_labels).issubset(labels):
            raise ValueError(
                "Label-expansion initialization requires every checkpoint label to exist "
                "in the target label vocabulary"
            )
        source_state = checkpoint["state_dict"]
        target_state = model.state_dict()
        expanded_keys: list[str] = []
        incompatible_keys: list[str] = []
        for key, target_tensor in target_state.items():
            source_tensor = source_state.get(key)
            if source_tensor is None:
                incompatible_keys.append(key)
                continue
            if source_tensor.shape == target_tensor.shape:
                target_state[key] = source_tensor
                continue
            is_expanded_classifier_tensor = (
                source_tensor.ndim in {1, 2}
                and target_tensor.ndim == source_tensor.ndim
                and source_tensor.shape[0] == len(checkpoint_labels)
                and target_tensor.shape[0] == len(labels)
                and source_tensor.shape[1:] == target_tensor.shape[1:]
            )
            if not is_expanded_classifier_tensor:
                incompatible_keys.append(key)
                continue
            copied = target_tensor.clone()
            target_index = {label: index for index, label in enumerate(labels)}
            for source_index, label in enumerate(checkpoint_labels):
                copied[target_index[label]] = source_tensor[source_index]
            target_state[key] = copied
            expanded_keys.append(key)
        unexpected_source_keys = sorted(set(source_state) - set(target_state))
        if incompatible_keys or unexpected_source_keys or not expanded_keys:
            raise ValueError(
                "Checkpoint cannot be safely expanded: "
                f"incompatible={sorted(incompatible_keys)}, "
                f"unexpected={unexpected_source_keys}, expanded={sorted(expanded_keys)}"
            )
        model.load_state_dict(target_state, strict=True)
        initialization_mode = "label_aware_superset_expansion_shared_rows_copied"
        added_labels = sorted(set(labels) - set(checkpoint_labels))
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "mode": initialization_mode,
        "checkpoint_label_count": len(checkpoint_labels),
        "target_label_count": len(labels),
        "added_labels": added_labels,
    }
