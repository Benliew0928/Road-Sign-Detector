from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from roadsign_assist.classification.training import (
    Architecture,
    build_torchvision_model,
)
from roadsign_assist.paths import project_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FolderClassifierTrainingConfig:
    data_root: Path
    architecture: Architecture = "mobilenet_v3_large"
    image_size: int = 224
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    workers: int = 4
    device: str = "auto"
    seed: int = 2513
    run_name: str = "malaysia_sign_classifier"
    confidence_threshold: float = 0.72
    allow_unreviewed_experiment: bool = False


class CropFolderDataset(Dataset[tuple[Any, int]]):
    def __init__(
        self,
        root: Path,
        labels: list[str],
        transform: Any,
    ) -> None:
        self.root = root
        self.labels = labels
        self.class_to_index = {label: index for index, label in enumerate(labels)}
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for label in labels:
            directory = root / label
            if not directory.exists():
                continue
            self.samples.extend(
                (path, self.class_to_index[label])
                for path in sorted(directory.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image), label


def validate_classifier_dataset(
    root: Path,
    *,
    allow_unreviewed_experiment: bool,
) -> dict[str, Any]:
    metadata_path = root / "dataset_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("coursework_images_included", -1)) != 0:
        raise ValueError("Coursework acceptance images cannot be used for classifier training")
    status = str(metadata.get("annotation_status", ""))
    if status != "approved" and not allow_unreviewed_experiment:
        raise ValueError(
            f"Classifier data status is {status!r}; "
            "an experimental override is required for unreviewed data"
        )
    return metadata


def _transforms(image_size: int) -> tuple[Any, Any]:
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.72, 1.0),
                ratio=(0.82, 1.18),
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25)],
                p=0.75,
            ),
            transforms.RandomPerspective(distortion_scale=0.16, p=0.35),
            transforms.RandomRotation(12),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.4))], p=0.20),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train, evaluate


def _expected_calibration_error(
    probabilities: np.ndarray[Any, np.dtype[np.float32]],
    labels: np.ndarray[Any, np.dtype[np.int64]],
    bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in pairwise(edges):
        selected = (confidence > lower) & (confidence <= upper)
        if not selected.any():
            continue
        accuracy = (predictions[selected] == labels[selected]).mean()
        error += float(selected.mean() * abs(accuracy - confidence[selected].mean()))
    return error


def compare_classifier_outputs(
    pytorch_logits: np.ndarray[Any, np.dtype[np.float32]],
    onnx_logits: np.ndarray[Any, np.dtype[np.float32]],
    *,
    temperature: float = 1.0,
    confidence_threshold: float = 0.72,
    maximum_probability_tolerance: float = 0.02,
    mean_probability_tolerance: float = 1e-3,
) -> dict[str, Any]:
    if pytorch_logits.shape != onnx_logits.shape:
        raise ValueError(
            "Classifier parity shape mismatch: "
            f"PyTorch {pytorch_logits.shape}, ONNX {onnx_logits.shape}"
        )
    if pytorch_logits.size == 0:
        raise ValueError("Classifier parity requires at least one output")
    absolute_logit_difference = np.abs(pytorch_logits - onnx_logits)
    safe_temperature = max(0.05, min(10.0, temperature))
    calibrated_pytorch = pytorch_logits / safe_temperature
    calibrated_onnx = onnx_logits / safe_temperature
    pytorch_shifted = calibrated_pytorch - calibrated_pytorch.max(axis=1, keepdims=True)
    onnx_shifted = calibrated_onnx - calibrated_onnx.max(axis=1, keepdims=True)
    pytorch_probabilities = np.exp(pytorch_shifted)
    pytorch_probabilities /= pytorch_probabilities.sum(axis=1, keepdims=True)
    onnx_probabilities = np.exp(onnx_shifted)
    onnx_probabilities /= onnx_probabilities.sum(axis=1, keepdims=True)
    absolute_probability_difference = np.abs(pytorch_probabilities - onnx_probabilities)
    maximum_probability_difference = float(absolute_probability_difference.max())
    mean_probability_difference = float(absolute_probability_difference.mean())
    top1_agreement = float((pytorch_logits.argmax(axis=1) == onnx_logits.argmax(axis=1)).mean())
    pytorch_accepted = pytorch_probabilities.max(axis=1) >= confidence_threshold
    onnx_accepted = onnx_probabilities.max(axis=1) >= confidence_threshold
    acceptance_agreement = float((pytorch_accepted == onnx_accepted).mean())
    return {
        "temperature": safe_temperature,
        "confidence_threshold": confidence_threshold,
        "maximum_probability_tolerance": maximum_probability_tolerance,
        "mean_probability_tolerance": mean_probability_tolerance,
        "passed": (
            maximum_probability_difference <= maximum_probability_tolerance
            and mean_probability_difference <= mean_probability_tolerance
            and top1_agreement == 1.0
            and acceptance_agreement == 1.0
        ),
        "maximum_absolute_logit_difference": float(absolute_logit_difference.max()),
        "mean_absolute_logit_difference": float(absolute_logit_difference.mean()),
        "maximum_absolute_probability_difference": maximum_probability_difference,
        "mean_absolute_probability_difference": mean_probability_difference,
        "top1_agreement": top1_agreement,
        "acceptance_agreement": acceptance_agreement,
    }


def _onnx_logits(onnx_path: Path, loader: Any) -> tuple[np.ndarray[Any, Any], list[str]]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    requested = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    session = ort.InferenceSession(str(onnx_path), providers=requested or available)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    parts: list[np.ndarray[Any, Any]] = []
    for images, _ in loader:
        parts.append(
            np.asarray(
                session.run(
                    [output_name],
                    {input_name: images.numpy().astype(np.float32)},
                )[0],
                dtype=np.float32,
            )
        )
    if not parts:
        raise ValueError("Classifier parity loader did not provide any samples")
    return np.concatenate(parts), list(session.get_providers())


def train_folder_classifier(config: FolderClassifierTrainingConfig) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from torch.utils.data import DataLoader, WeightedRandomSampler

    root = project_path(config.data_root)
    metadata = validate_classifier_dataset(
        root,
        allow_unreviewed_experiment=config.allow_unreviewed_experiment,
    )
    labels: list[str] = list(json.loads((root / "labels.json").read_text(encoding="utf-8")))
    if len(labels) < 2:
        raise ValueError("Classifier requires at least two labels")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_transform, evaluation_transform = _transforms(config.image_size)
    train_dataset = CropFolderDataset(root / "train", labels, train_transform)
    validation_dataset = CropFolderDataset(
        root / "validation",
        labels,
        evaluation_transform,
    )
    test_dataset = CropFolderDataset(root / "test", labels, evaluation_transform)
    if not train_dataset.samples or not validation_dataset.samples:
        raise ValueError("Classifier training and validation splits must both contain crops")
    trained_label_indices = {label for _, label in train_dataset.samples}
    missing_train_labels = [
        label for index, label in enumerate(labels) if index not in trained_label_indices
    ]
    if missing_train_labels and metadata.get("annotation_status") == "approved":
        raise ValueError(
            "Approved classifier data must include every label in the training split: "
            f"{missing_train_labels}"
        )

    counts = Counter(label for _, label in train_dataset.samples)
    sample_weights = [1.0 / counts[label] for _, label in train_dataset.samples]
    generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device_name = (
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.device == "auto"
        else config.device
    )
    device = torch.device(device_name)
    model = build_torchvision_model(config.architecture, len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs),
    )
    use_amp = device.type == "cuda"
    scaler = torch.GradScaler("cuda", enabled=use_amp)

    run_root = project_path("outputs/training") / config.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = project_path("models/checkpoints") / config.run_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / "best.pt"
    history: list[dict[str, float | int]] = []
    best_f1 = -1.0

    def collect(loader: Any) -> tuple[Any, Any, float]:
        model.eval()
        logits_parts: list[Any] = []
        labels_parts: list[Any] = []
        loss_sum = 0.0
        sample_count = 0
        with torch.inference_mode():
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, targets)
                logits_parts.append(logits.detach().cpu())
                labels_parts.append(targets.detach().cpu())
                loss_sum += float(loss.item()) * len(targets)
                sample_count += len(targets)
        if not logits_parts:
            return torch.empty((0, len(labels))), torch.empty((0,), dtype=torch.long), 0.0
        return (
            torch.cat(logits_parts),
            torch.cat(labels_parts),
            loss_sum / max(1, sample_count),
        )

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        train_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * len(targets)
            seen += len(targets)
        scheduler.step()

        validation_logits, validation_targets, validation_loss = collect(validation_loader)
        predictions = validation_logits.argmax(dim=1).numpy()
        targets_numpy = validation_targets.numpy()
        macro_f1 = float(
            f1_score(
                targets_numpy,
                predictions,
                average="macro",
            )
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, seen),
            "validation_loss": validation_loss,
            "validation_accuracy": float(accuracy_score(targets_numpy, predictions)),
            "validation_macro_f1": macro_f1,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record))
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": labels,
                    "config": asdict(config),
                    "metadata": metadata,
                    "validation_macro_f1": macro_f1,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    validation_logits, validation_targets, _ = collect(validation_loader)

    temperature = nn.Parameter(torch.ones(1, device=device))
    calibration_optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=50)
    calibration_logits = validation_logits.to(device)
    calibration_targets = validation_targets.to(device)

    def calibration_step() -> Any:
        calibration_optimizer.zero_grad()
        loss = nn.functional.cross_entropy(
            calibration_logits / temperature.clamp(0.05, 10.0),
            calibration_targets,
        )
        loss.backward()
        return loss

    calibration_optimizer.step(calibration_step)
    calibrated_temperature = float(temperature.detach().clamp(0.05, 10.0).item())

    test_logits, test_targets, test_loss = collect(test_loader)
    evaluation_logits = test_logits if len(test_targets) else validation_logits
    evaluation_targets = test_targets if len(test_targets) else validation_targets
    probabilities = (
        torch.softmax(
            evaluation_logits / calibrated_temperature,
            dim=1,
        )
        .numpy()
        .astype(np.float32)
    )
    target_array = evaluation_targets.numpy().astype(np.int64)
    predicted_array = probabilities.argmax(axis=1)
    maximum_confidence = probabilities.max(axis=1)
    accepted_mask = maximum_confidence >= config.confidence_threshold
    accepted_count = int(accepted_mask.sum())
    correct_mask = predicted_array == target_array
    observed_label_count = len(np.unique(target_array))
    metrics: dict[str, Any] = {
        "schema_version": "1.0",
        "experimental": metadata.get("annotation_status") != "approved",
        "architecture": config.architecture,
        "labels": len(labels),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "missing_train_labels": missing_train_labels,
        "evaluation_split": "test" if len(test_targets) else "validation",
        "evaluation_observed_labels": observed_label_count,
        "evaluation_label_coverage": observed_label_count / len(labels),
        "loss": test_loss,
        "accuracy": float(accuracy_score(target_array, predicted_array)),
        "confidence_threshold": config.confidence_threshold,
        "accepted_samples": accepted_count,
        "selective_coverage": float(accepted_mask.mean()),
        "selective_accuracy": (
            float(correct_mask[accepted_mask].mean()) if accepted_count else None
        ),
        "accepted_correct_rate": float((correct_mask & accepted_mask).mean()),
        "macro_f1_observed": float(
            f1_score(
                target_array,
                predicted_array,
                average="macro",
                labels=sorted(np.unique(target_array).tolist()),
                zero_division=0,  # pyright: ignore[reportArgumentType]
            )
        ),
        "macro_f1_all_labels": float(
            f1_score(
                target_array,
                predicted_array,
                average="macro",
                labels=list(range(len(labels))),
                zero_division=0,  # pyright: ignore[reportArgumentType]
            )
        ),
        "ece": _expected_calibration_error(probabilities, target_array),
        "temperature": calibrated_temperature,
        "confusion_matrix": confusion_matrix(
            target_array,
            predicted_array,
            labels=list(range(len(labels))),
        ).tolist(),
        "history": history,
        "unknown_auroc": None,
        "unknown_auroc_reason": "No reviewed out-of-distribution validation set is available.",
    }

    export_root = project_path("models/exported")
    if metrics["experimental"]:
        export_root /= "experimental"
    export_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    sample = torch.zeros(
        (1, 3, config.image_size, config.image_size),
        dtype=torch.float32,
        device=device,
    )
    artifact_stem = config.run_name if metrics["experimental"] else "sign_classifier"
    onnx_path = export_root / f"{artifact_stem}.onnx"
    torch.onnx.export(
        model,
        (sample,),
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    (export_root / f"{artifact_stem}.labels.json").write_text(
        json.dumps(labels, indent=2) + "\n",
        encoding="utf-8",
    )
    (export_root / f"{artifact_stem}.calibration.json").write_text(
        json.dumps(
            {
                "temperature": calibrated_temperature,
                "confidence_threshold": config.confidence_threshold,
                "experimental": metrics["experimental"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    onnx_logits, onnx_providers = _onnx_logits(
        onnx_path,
        test_loader if len(test_targets) else validation_loader,
    )
    parity = compare_classifier_outputs(
        evaluation_logits.numpy().astype(np.float32),
        onnx_logits,
        temperature=calibrated_temperature,
        confidence_threshold=config.confidence_threshold,
    )
    metrics["onnx_parity"] = {
        **parity,
        "providers": onnx_providers,
    }
    metrics["artifacts"] = {
        "checkpoint": str(checkpoint_path.relative_to(project_path("."))),
        "onnx": str(onnx_path.relative_to(project_path("."))),
        "labels": str(
            (export_root / f"{artifact_stem}.labels.json").relative_to(project_path("."))
        ),
        "calibration": str(
            (export_root / f"{artifact_stem}.calibration.json").relative_to(project_path("."))
        ),
    }
    (run_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    if not parity["passed"]:
        raise RuntimeError(
            "Classifier ONNX parity failed: "
            "maximum probability difference "
            f"{parity['maximum_absolute_probability_difference']:.6f}"
        )
    return metrics
