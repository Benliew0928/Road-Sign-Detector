from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from roadsign_assist.classification.folder_training import CropFolderDataset
from roadsign_assist.classification.training import Architecture, build_torchvision_model
from roadsign_assist.paths import project_path


def _evaluation_transform(image_size: int) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _normalize_rows(values: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-8)).astype(np.float32)


def _softmax(logits: np.ndarray[Any, Any], temperature: float) -> np.ndarray[Any, Any]:
    calibrated = np.asarray(logits, dtype=np.float32) / max(temperature, 0.05)
    shifted = calibrated - calibrated.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _embedding_metrics(
    *,
    logits: np.ndarray[Any, Any],
    embeddings: np.ndarray[Any, Any],
    targets: np.ndarray[Any, Any],
    prototypes: np.ndarray[Any, Any],
    confidence_threshold: float,
    distance_threshold: float,
    temperature: float,
) -> dict[str, Any]:
    probabilities = _softmax(logits, temperature)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    similarities = embeddings @ prototypes.T
    nearest = similarities.argmax(axis=1)
    distance = 1.0 - similarities.max(axis=1)
    accepted = (
        (confidence >= confidence_threshold)
        & (distance <= distance_threshold)
        & (predictions == nearest)
    )
    correct = predictions == targets
    accepted_count = int(accepted.sum())
    return {
        "samples": len(targets),
        "accepted_samples": accepted_count,
        "coverage": float(accepted.mean()) if len(targets) else 0.0,
        "accuracy": float(correct.mean()) if len(targets) else 0.0,
        "accepted_accuracy": (float(correct[accepted].mean()) if accepted_count else None),
        "confidence_rejections": int((confidence < confidence_threshold).sum()),
        "distance_rejections": int((distance > distance_threshold).sum()),
        "prototype_disagreements": int((predictions != nearest).sum()),
        "distance_mean": float(distance.mean()) if len(distance) else None,
        "distance_p95": (float(np.percentile(distance, 95)) if len(distance) else None),
    }


def export_classifier_with_embeddings(
    *,
    checkpoint_path: str | Path,
    data_root: str | Path,
    model_output: str | Path,
    calibration_output: str | Path,
    report_output: str | Path,
    device: str = "auto",
    batch_size: int = 64,
    workers: int = 0,
    retention_quantile: float = 0.95,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    checkpoint_file = project_path(checkpoint_path)
    dataset_root = project_path(data_root)
    model_file = project_path(model_output)
    calibration_file = project_path(calibration_output)
    report_file = project_path(report_output)
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    labels = [str(value) for value in checkpoint["labels"]]
    config: dict[str, Any] = checkpoint["config"]
    architecture = cast(Architecture, str(config["architecture"]))
    image_size = int(config["image_size"])
    confidence_threshold = float(config.get("confidence_threshold", 0.72))
    if device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.isdigit():
        selected_device = f"cuda:{device}"
    else:
        selected_device = device
    backbone = build_torchvision_model(
        architecture,
        len(labels),
        pretrained=False,
    )
    backbone.load_state_dict(checkpoint["state_dict"])

    class ClassifierWithEmbedding(nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, image: Any) -> tuple[Any, Any]:
            features = self.model.features(image)
            pooled = self.model.avgpool(features)
            embedding = torch.flatten(pooled, 1)
            logits = self.model.classifier(embedding)
            return logits, functional.normalize(embedding, p=2, dim=1)

    model = ClassifierWithEmbedding(backbone).to(selected_device).eval()
    transform = _evaluation_transform(image_size)

    def collect(
        split: str,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        dataset = CropFolderDataset(dataset_root / split, labels, transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=selected_device.startswith("cuda"),
        )
        logits_parts: list[np.ndarray[Any, Any]] = []
        embedding_parts: list[np.ndarray[Any, Any]] = []
        target_parts: list[np.ndarray[Any, Any]] = []
        with torch.inference_mode():
            for images, targets in loader:
                logits, embeddings = model(images.to(selected_device, non_blocking=True))
                logits_parts.append(logits.cpu().numpy().astype(np.float32))
                embedding_parts.append(embeddings.cpu().numpy().astype(np.float32))
                target_parts.append(targets.numpy().astype(np.int64))
        if not target_parts:
            raise ValueError(f"Classifier split {split!r} is empty")
        return (
            np.concatenate(logits_parts),
            np.concatenate(embedding_parts),
            np.concatenate(target_parts),
        )

    train_logits, train_embeddings, train_targets = collect("train")
    validation_logits, validation_embeddings, validation_targets = collect("validation")
    test_logits, test_embeddings, test_targets = collect("test")
    del train_logits

    prototype_rows: list[np.ndarray[Any, Any]] = []
    for class_index, label in enumerate(labels):
        rows = train_embeddings[train_targets == class_index]
        if not len(rows):
            raise ValueError(f"Training split has no embeddings for class {label!r}")
        prototype_rows.append(rows.mean(axis=0))
    prototypes = _normalize_rows(np.stack(prototype_rows))

    validation_predictions = validation_logits.argmax(axis=1)
    true_distances = 1.0 - np.sum(
        validation_embeddings * prototypes[validation_targets],
        axis=1,
    )
    correctly_classified = validation_predictions == validation_targets
    calibration_distances = true_distances[correctly_classified]
    if not len(calibration_distances):
        raise ValueError("No correct validation predictions are available for calibration")
    distance_threshold = float(
        np.clip(
            np.quantile(calibration_distances, retention_quantile),
            0.01,
            1.0,
        )
    )

    temperature = 1.0
    prior_calibration = project_path(
        f"models/exported/experimental/{config['run_name']}.calibration.json"
    )
    if prior_calibration.is_file():
        payload = json.loads(prior_calibration.read_text(encoding="utf-8"))
        temperature = float(payload.get("temperature", 1.0))

    calibration_payload: dict[str, Any] = {
        "schema_version": "2.0",
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "experimental": True,
        "embedding_gate": {
            "distance_metric": "cosine",
            "distance_threshold": distance_threshold,
            "prototype_labels": labels,
            "prototypes": prototypes.tolist(),
            "require_label_agreement": True,
            "retention_quantile": retention_quantile,
            "calibration_split": "validation",
            "prototype_split": "train",
        },
    }

    model_file.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32).to(selected_device)
    torch.onnx.export(
        model,
        (sample,),
        model_file,
        input_names=["image"],
        output_names=["logits", "embedding"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    calibration_file.parent.mkdir(parents=True, exist_ok=True)
    calibration_file.write_text(
        json.dumps(calibration_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    import onnxruntime as ort

    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in ort.get_available_providers()
    ]
    session = ort.InferenceSession(
        str(model_file),
        providers=providers or ["CPUExecutionProvider"],
    )
    sample_batch = sample.detach().cpu().numpy()
    onnx_logits, onnx_embeddings = session.run(
        None,
        {session.get_inputs()[0].name: sample_batch},
    )
    with torch.inference_mode():
        torch_logits, torch_embeddings = model(sample)
    parity = {
        "logits_max_abs_difference": float(
            np.max(np.abs(onnx_logits - torch_logits.cpu().numpy()))
        ),
        "embedding_max_abs_difference": float(
            np.max(np.abs(onnx_embeddings - torch_embeddings.cpu().numpy()))
        ),
    }
    parity["passed"] = bool(
        parity["logits_max_abs_difference"] <= 0.02
        and parity["embedding_max_abs_difference"] <= 0.02
    )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "experimental": True,
        "checkpoint": str(checkpoint_file.relative_to(project_path("."))),
        "model": str(model_file.relative_to(project_path("."))),
        "calibration": str(calibration_file.relative_to(project_path("."))),
        "architecture": architecture,
        "image_size": image_size,
        "embedding_dimension": int(prototypes.shape[1]),
        "confidence_threshold": confidence_threshold,
        "distance_threshold": distance_threshold,
        "retention_quantile": retention_quantile,
        "validation": _embedding_metrics(
            logits=validation_logits,
            embeddings=validation_embeddings,
            targets=validation_targets,
            prototypes=prototypes,
            confidence_threshold=confidence_threshold,
            distance_threshold=distance_threshold,
            temperature=temperature,
        ),
        "test": _embedding_metrics(
            logits=test_logits,
            embeddings=test_embeddings,
            targets=test_targets,
            prototypes=prototypes,
            confidence_threshold=confidence_threshold,
            distance_threshold=distance_threshold,
            temperature=temperature,
        ),
        "onnx_parity": parity,
        "providers": session.get_providers(),
        "unknown_auroc": None,
        "unknown_auroc_reason": "No reviewed out-of-distribution set is available.",
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not parity["passed"]:
        raise RuntimeError("Embedding classifier ONNX parity exceeded tolerance")
    return report
