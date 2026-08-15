from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import csv
import json
import math
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

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
    tune_confidence_threshold: bool = False
    target_selective_accuracy: float = 0.80
    evaluate_test: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class FolderClassifierRunPaths:
    run_root: Path
    checkpoint_root: Path
    checkpoint_path: Path
    artifact_root: Path
    onnx_path: Path
    labels_path: Path
    calibration_path: Path


def _validate_run_name(run_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError(
            "Classifier run names may contain only letters, numbers, dots, underscores, "
            "and hyphens, and must begin with a letter or number"
        )
    return run_name


def classifier_run_paths(
    run_name: str,
    *,
    experimental: bool,
) -> FolderClassifierRunPaths:
    safe_name = _validate_run_name(run_name)
    # All tuning artifacts remain local candidates. Only the explicit promotion
    # command writes into the DVC-eligible runtime bundle.
    artifact_parent = project_path("models/candidates")
    artifact_root = artifact_parent / safe_name
    return FolderClassifierRunPaths(
        run_root=project_path("outputs/training") / safe_name,
        checkpoint_root=project_path("models/checkpoints") / safe_name,
        checkpoint_path=project_path("models/checkpoints") / safe_name / "best.pt",
        artifact_root=artifact_root,
        onnx_path=artifact_root / "sign_classifier.onnx",
        labels_path=artifact_root / "sign_classifier.labels.json",
        calibration_path=artifact_root / "sign_classifier.calibration.json",
    )


def prepare_classifier_run_paths(
    paths: FolderClassifierRunPaths,
    *,
    overwrite: bool,
) -> None:
    roots = (paths.run_root, paths.checkpoint_root, paths.artifact_root)
    existing = [path for path in roots if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Classifier run output already exists: {rendered}. "
            "Choose a unique --name or pass --overwrite deliberately."
        )
    for path in roots:
        path.mkdir(parents=True, exist_ok=True)


def _serializable_config(config: FolderClassifierTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["data_root"] = str(config.data_root)
    return payload


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_path("."),
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _runtime_environment(device_name: str) -> dict[str, Any]:
    import torch
    import torchvision

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": device_name,
        "gpu": (
            torch.cuda.get_device_name(torch.device(device_name))
            if torch.device(device_name).type == "cuda" and torch.cuda.is_available()
            else None
        ),
    }


def _wilson_confidence_interval(correct: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0, "lower_percent": 0.0, "upper_percent": 0.0}
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + (z * z / (2.0 * total))) / denominator
    margin = (
        z
        * math.sqrt(
            (proportion * (1.0 - proportion) / total)
            + (z * z / (4.0 * total * total))
        )
        / denominator
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return {
        "lower": lower,
        "upper": upper,
        "lower_percent": lower * 100.0,
        "upper_percent": upper * 100.0,
    }


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


def should_export_classifier_experimentally(
    metadata: dict[str, Any],
    *,
    allow_unreviewed_experiment: bool,
) -> bool:
    return allow_unreviewed_experiment or metadata.get("annotation_status") != "approved"


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
            transforms.Resize((image_size, image_size)),
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


def select_confidence_threshold(
    probabilities: np.ndarray[Any, np.dtype[np.float32]],
    targets: np.ndarray[Any, np.dtype[np.int64]],
    *,
    fallback_threshold: float,
    target_selective_accuracy: float,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    candidates = sorted({0.0, fallback_threshold, *confidence.tolist()})
    qualifying: list[dict[str, float | int]] = []
    evaluated: list[dict[str, float | int]] = []
    for threshold in candidates:
        accepted = confidence >= threshold
        accepted_count = int(accepted.sum())
        if accepted_count == 0:
            continue
        selective_accuracy = float(correct[accepted].mean())
        row: dict[str, float | int] = {
            "threshold": float(threshold),
            "accepted_samples": accepted_count,
            "coverage": accepted_count / len(targets),
            "selective_accuracy": selective_accuracy,
        }
        evaluated.append(row)
        if selective_accuracy >= target_selective_accuracy:
            qualifying.append(row)
    if qualifying:
        selected = max(
            qualifying,
            key=lambda row: (float(row["coverage"]), -float(row["threshold"])),
        )
        reason = (
            "maximum validation coverage meeting the target selective accuracy "
            f"of {target_selective_accuracy:.3f}"
        )
    else:
        selected = max(
            evaluated,
            key=lambda row: (float(row["selective_accuracy"]), float(row["coverage"])),
        )
        reason = (
            "no validation threshold met the target selective accuracy; selected the "
            "highest observed selective accuracy with coverage as the tie-breaker"
        )
    return {
        **selected,
        "target_selective_accuracy": target_selective_accuracy,
        "reason": reason,
        "candidate_count": len(evaluated),
    }


def _probability_metrics(
    probabilities: np.ndarray[Any, np.dtype[np.float32]],
    targets: np.ndarray[Any, np.dtype[np.int64]],
    labels: list[str],
    *,
    confidence_threshold: float,
    loss: float,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    accepted = confidence >= confidence_threshold
    correct = predictions == targets
    accepted_count = int(accepted.sum())
    correct_count = int(correct.sum())
    observed_indices = sorted(np.unique(targets).tolist())
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(labels))))
    precision_values, recall_values, class_f1_values, support_values = (
        precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(len(labels))),
        zero_division=0,  # pyright: ignore[reportArgumentType]
        )
    )
    precision = np.asarray(precision_values, dtype=np.float64)
    recall = np.asarray(recall_values, dtype=np.float64)
    class_f1 = np.asarray(class_f1_values, dtype=np.float64)
    support = np.asarray(support_values, dtype=np.int64)
    confusion_pairs: list[dict[str, Any]] = []
    for expected_index, row in enumerate(matrix):
        for predicted_index, count in enumerate(row):
            if expected_index != predicted_index and count:
                confusion_pairs.append(
                    {
                        "expected": labels[expected_index],
                        "predicted": labels[predicted_index],
                        "count": int(count),
                    }
                )
    confusion_pairs.sort(
        key=lambda row: (-int(row["count"]), str(row["expected"]), str(row["predicted"]))
    )
    accuracy = float(accuracy_score(targets, predictions))
    macro_f1_observed = float(
        f1_score(
            targets,
            predictions,
            average="macro",
            labels=observed_indices,
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
    )
    macro_f1_all = float(
        f1_score(
            targets,
            predictions,
            average="macro",
            labels=list(range(len(labels))),
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
    )
    return {
        "samples": len(targets),
        "loss": loss,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "accuracy_correct": correct_count,
        "accuracy_total": len(targets),
        "accuracy_95_ci": _wilson_confidence_interval(correct_count, len(targets)),
        "macro_f1_observed": macro_f1_observed,
        "macro_f1_observed_percent": macro_f1_observed * 100.0,
        "macro_f1_all_labels": macro_f1_all,
        "macro_f1_all_labels_percent": macro_f1_all * 100.0,
        "observed_labels": len(observed_indices),
        "label_coverage": len(observed_indices) / len(labels),
        "confidence_threshold": confidence_threshold,
        "accepted_samples": accepted_count,
        "selective_coverage": float(accepted.mean()),
        "selective_accuracy": float(correct[accepted].mean()) if accepted_count else None,
        "accepted_correct_rate": float((correct & accepted).mean()),
        "ece": _expected_calibration_error(probabilities, targets),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
        "largest_confusion_pairs": confusion_pairs[:10],
    }


def compare_classifier_outputs(
    pytorch_logits: np.ndarray[Any, np.dtype[np.float32]],
    onnx_logits: np.ndarray[Any, np.dtype[np.float32]],
    *,
    temperature: float = 1.0,
    confidence_threshold: float = 0.72,
    maximum_probability_tolerance: float = 0.02,
    mean_probability_tolerance: float = 1e-3,
    minimum_top1_agreement: float = 0.995,
    minimum_acceptance_agreement: float = 0.995,
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
        "minimum_top1_agreement": minimum_top1_agreement,
        "minimum_acceptance_agreement": minimum_acceptance_agreement,
        "passed": (
            maximum_probability_difference <= maximum_probability_tolerance
            and mean_probability_difference <= mean_probability_tolerance
            and top1_agreement >= minimum_top1_agreement
            and acceptance_agreement >= minimum_acceptance_agreement
        ),
        "maximum_absolute_logit_difference": float(absolute_logit_difference.max()),
        "mean_absolute_logit_difference": float(absolute_logit_difference.mean()),
        "maximum_absolute_probability_difference": maximum_probability_difference,
        "mean_absolute_probability_difference": mean_probability_difference,
        "top1_agreement": top1_agreement,
        "acceptance_agreement": acceptance_agreement,
    }


def _onnx_logits(
    onnx_path: Path,
    loader: Any,
) -> tuple[np.ndarray[Any, Any], list[str], float]:
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
    started = time.perf_counter()
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
    elapsed = time.perf_counter() - started
    return np.concatenate(parts), list(session.get_providers()), elapsed


def train_folder_classifier(config: FolderClassifierTrainingConfig) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, WeightedRandomSampler

    training_started = time.perf_counter()
    root = project_path(config.data_root)
    metadata = validate_classifier_dataset(
        root,
        allow_unreviewed_experiment=config.allow_unreviewed_experiment,
    )
    labels: list[str] = list(json.loads((root / "labels.json").read_text(encoding="utf-8")))
    if len(labels) < 2:
        raise ValueError("Classifier requires at least two labels")
    experimental_export = should_export_classifier_experimentally(
        metadata,
        allow_unreviewed_experiment=config.allow_unreviewed_experiment,
    )
    paths = classifier_run_paths(config.run_name, experimental=experimental_export)
    prepare_classifier_run_paths(paths, overwrite=config.overwrite)
    git_commit = _git_commit()

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
        persistent_workers=config.workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        persistent_workers=config.workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        persistent_workers=config.workers > 0,
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

    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0

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
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": labels,
                    "config": _serializable_config(config),
                    "metadata": metadata,
                    "validation_macro_f1": macro_f1,
                    "best_epoch": epoch,
                    "dataset_id": metadata.get("dataset_id"),
                    "git_commit": git_commit,
                },
                paths.checkpoint_path,
            )

    checkpoint = torch.load(paths.checkpoint_path, map_location=device, weights_only=False)
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

    validation_probabilities = (
        torch.softmax(validation_logits / calibrated_temperature, dim=1)
        .numpy()
        .astype(np.float32)
    )
    validation_target_array = validation_targets.numpy().astype(np.int64)
    threshold_selection: dict[str, Any] | None = None
    effective_confidence_threshold = config.confidence_threshold
    if config.tune_confidence_threshold:
        threshold_selection = select_confidence_threshold(
            validation_probabilities,
            validation_target_array,
            fallback_threshold=config.confidence_threshold,
            target_selective_accuracy=config.target_selective_accuracy,
        )
        effective_confidence_threshold = float(threshold_selection["threshold"])
    validation_metrics = _probability_metrics(
        validation_probabilities,
        validation_target_array,
        labels,
        confidence_threshold=effective_confidence_threshold,
        loss=float(history[best_epoch - 1]["validation_loss"]),
    )

    evaluation_split = "validation"
    evaluation_logits = validation_logits
    evaluation_targets = validation_targets
    evaluation_loss = float(history[best_epoch - 1]["validation_loss"])
    evaluation_loader = validation_loader
    if config.evaluate_test:
        test_logits, test_targets, test_loss = collect(test_loader)
        if not len(test_targets):
            raise ValueError("Explicit test evaluation requested, but the test split is empty")
        evaluation_split = "test"
        evaluation_logits = test_logits
        evaluation_targets = test_targets
        evaluation_loss = test_loss
        evaluation_loader = test_loader
    probabilities = (
        torch.softmax(evaluation_logits / calibrated_temperature, dim=1)
        .numpy()
        .astype(np.float32)
    )
    target_array = evaluation_targets.numpy().astype(np.int64)
    evaluation_metrics = _probability_metrics(
        probabilities,
        target_array,
        labels,
        confidence_threshold=effective_confidence_threshold,
        loss=evaluation_loss,
    )
    metrics: dict[str, Any] = {
        "schema_version": "2.0",
        "run_name": config.run_name,
        "experimental": experimental_export,
        "release_status": metadata.get("release_status", "unspecified"),
        "dataset_id": metadata.get("dataset_id", root.name),
        "git_commit": git_commit,
        "environment": _runtime_environment(device_name),
        "configuration": _serializable_config(config),
        "architecture": config.architecture,
        "labels": len(labels),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "missing_train_labels": missing_train_labels,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "temperature": calibrated_temperature,
        "confidence_threshold": effective_confidence_threshold,
        "threshold_selection": threshold_selection,
        "selection_split": "validation",
        "validation": validation_metrics,
        "evaluation_split": evaluation_split,
        "evaluation": evaluation_metrics,
        # Compatibility fields used by the comparison and safety-report commands.
        "evaluation_observed_labels": evaluation_metrics["observed_labels"],
        "evaluation_label_coverage": evaluation_metrics["label_coverage"],
        "loss": evaluation_metrics["loss"],
        "accuracy": evaluation_metrics["accuracy"],
        "accuracy_percent": evaluation_metrics["accuracy_percent"],
        "accuracy_correct": evaluation_metrics["accuracy_correct"],
        "accuracy_total": evaluation_metrics["accuracy_total"],
        "accuracy_95_ci": evaluation_metrics["accuracy_95_ci"],
        "accepted_samples": evaluation_metrics["accepted_samples"],
        "selective_coverage": evaluation_metrics["selective_coverage"],
        "selective_accuracy": evaluation_metrics["selective_accuracy"],
        "accepted_correct_rate": evaluation_metrics["accepted_correct_rate"],
        "macro_f1_observed": evaluation_metrics["macro_f1_observed"],
        "macro_f1_all_labels": evaluation_metrics["macro_f1_all_labels"],
        "macro_f1_all_labels_percent": evaluation_metrics["macro_f1_all_labels_percent"],
        "ece": evaluation_metrics["ece"],
        "per_class": evaluation_metrics["per_class"],
        "confusion_matrix": evaluation_metrics["confusion_matrix"],
        "largest_confusion_pairs": evaluation_metrics["largest_confusion_pairs"],
        "history": history,
        "training_seconds": time.perf_counter() - training_started,
        "unknown_auroc": None,
        "unknown_auroc_reason": "No reviewed out-of-distribution validation set is available.",
    }

    model.eval()
    sample = torch.zeros(
        (1, 3, config.image_size, config.image_size),
        dtype=torch.float32,
        device=device,
    )
    torch.onnx.export(
        model,
        (sample,),
        paths.onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    paths.labels_path.write_text(
        json.dumps(labels, indent=2) + "\n",
        encoding="utf-8",
    )
    paths.calibration_path.write_text(
        json.dumps(
            {
                "temperature": calibrated_temperature,
                "confidence_threshold": effective_confidence_threshold,
                "experimental": metrics["experimental"],
                "release_status": metadata.get("release_status", "unspecified"),
                "dataset_id": metadata.get("dataset_id", root.name),
                "source_run": config.run_name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    onnx_logits, onnx_providers, onnx_elapsed = _onnx_logits(
        paths.onnx_path,
        evaluation_loader,
    )
    parity = compare_classifier_outputs(
        evaluation_logits.numpy().astype(np.float32),
        onnx_logits,
        temperature=calibrated_temperature,
        confidence_threshold=effective_confidence_threshold,
    )
    metrics["onnx_parity"] = {
        **parity,
        "providers": onnx_providers,
    }
    metrics["onnx_size_bytes"] = paths.onnx_path.stat().st_size
    metrics["onnx_mean_latency_ms"] = onnx_elapsed * 1000.0 / len(target_array)
    metrics["artifacts"] = {
        "checkpoint": str(paths.checkpoint_path.relative_to(project_path("."))),
        "onnx": str(paths.onnx_path.relative_to(project_path("."))),
        "labels": str(paths.labels_path.relative_to(project_path("."))),
        "calibration": str(paths.calibration_path.relative_to(project_path("."))),
    }
    metrics["training_seconds"] = time.perf_counter() - training_started
    (paths.run_root / "metrics.json").write_text(
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


def promote_classifier_candidate(
    run_name: str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    safe_name = _validate_run_name(run_name)
    metrics_path = project_path("outputs/training") / safe_name / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("evaluation_split") != "test":
        raise ValueError(
            "A classifier can be promoted only after an explicit locked test evaluation"
        )
    parity = metrics.get("onnx_parity")
    if not isinstance(parity, dict) or parity.get("passed") is not True:
        raise ValueError("A classifier can be promoted only after ONNX parity passes")
    artifacts = metrics.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Classifier metrics do not contain an artifact manifest")
    source_paths = {
        key: project_path(str(artifacts[key]))
        for key in ("onnx", "labels", "calibration")
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Classifier promotion artifacts are missing: {missing}")

    export_root = project_path("models/exported/runtime")
    destinations = {
        "onnx": export_root / "sign_classifier.onnx",
        "labels": export_root / "sign_classifier.labels.json",
        "calibration": export_root / "sign_classifier.calibration.json",
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Runtime classifier already exists: {rendered}. Pass --overwrite deliberately."
        )
    export_root.mkdir(parents=True, exist_ok=True)
    for key, destination in destinations.items():
        shutil.copy2(source_paths[key], destination)

    release_status = str(metrics.get("release_status", "clean_candidate"))
    manifest = {
        "schema_version": "1.0",
        "source_run": safe_name,
        "dataset_id": metrics.get("dataset_id"),
        "git_commit": metrics.get("git_commit"),
        "release_status": release_status,
        "clean_final": release_status == "clean_final",
        "test_accuracy": metrics.get("accuracy"),
        "test_macro_f1_all_labels": metrics.get("macro_f1_all_labels"),
        "confidence_threshold": metrics.get("confidence_threshold"),
        "onnx_parity_passed": True,
        "metrics_path": str(metrics_path.relative_to(project_path("."))),
        "artifacts": {
            key: str(path.relative_to(project_path(".")))
            for key, path in destinations.items()
        },
    }
    manifest_path = export_root / "sign_classifier.runtime.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def evaluate_classifier_candidate(
    run_name: str,
    *,
    tune_confidence_threshold: bool = True,
    target_selective_accuracy: float = 0.98,
    overwrite: bool = False,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    safe_name = _validate_run_name(run_name)
    run_root = project_path("outputs/training") / safe_name
    metrics_path = run_root / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("evaluation_split") == "test" and not overwrite:
        raise FileExistsError(
            "This candidate already has a locked test evaluation. Pass --overwrite only "
            "to deliberately repeat it."
        )
    artifacts = metrics.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Classifier metrics do not contain an artifact manifest")
    onnx_path = project_path(str(artifacts["onnx"]))
    labels_path = project_path(str(artifacts["labels"]))
    calibration_path = project_path(str(artifacts["calibration"]))
    for path in (onnx_path, labels_path, calibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    configuration = metrics.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Classifier metrics do not contain the complete configuration")
    data_root = project_path(str(configuration["data_root"]))
    image_size = int(configuration["image_size"])
    batch_size = int(configuration["batch_size"])
    workers = int(configuration.get("workers", 0))
    labels: list[str] = json.loads(labels_path.read_text(encoding="utf-8"))
    calibration: dict[str, Any] = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    temperature = float(calibration.get("temperature", metrics.get("temperature", 1.0)))
    evaluation_transform = _transforms(image_size)[1]
    validation_dataset = CropFolderDataset(
        data_root / "validation", labels, evaluation_transform
    )
    test_dataset = CropFolderDataset(data_root / "test", labels, evaluation_transform)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_logits, _, validation_elapsed = _onnx_logits(onnx_path, validation_loader)
    validation_targets = np.asarray(
        [target for _, target in validation_dataset.samples], dtype=np.int64
    )
    validation_shifted = validation_logits / max(0.05, min(10.0, temperature))
    validation_shifted -= validation_shifted.max(axis=1, keepdims=True)
    validation_probabilities = np.exp(validation_shifted)
    validation_probabilities /= validation_probabilities.sum(axis=1, keepdims=True)
    confidence_threshold = float(calibration.get("confidence_threshold", 0.72))
    threshold_selection: dict[str, Any] | None = None
    if tune_confidence_threshold:
        threshold_selection = select_confidence_threshold(
            validation_probabilities.astype(np.float32),
            validation_targets,
            fallback_threshold=confidence_threshold,
            target_selective_accuracy=target_selective_accuracy,
        )
        confidence_threshold = float(threshold_selection["threshold"])

    test_logits, providers, test_elapsed = _onnx_logits(onnx_path, test_loader)
    test_targets = np.asarray([target for _, target in test_dataset.samples], dtype=np.int64)
    test_shifted = test_logits / max(0.05, min(10.0, temperature))
    test_shifted -= test_shifted.max(axis=1, keepdims=True)
    test_probabilities = np.exp(test_shifted)
    test_probabilities /= test_probabilities.sum(axis=1, keepdims=True)
    selected_probabilities = np.maximum(
        test_probabilities[np.arange(len(test_targets)), test_targets], 1e-12
    )
    test_loss = float(-np.log(selected_probabilities).mean())
    test_metrics = _probability_metrics(
        test_probabilities.astype(np.float32),
        test_targets,
        labels,
        confidence_threshold=confidence_threshold,
        loss=test_loss,
    )
    checkpoint_path = project_path(str(artifacts["checkpoint"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device_setting = str(configuration.get("device", "auto"))
    device_name = (
        "cuda"
        if device_setting == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_setting == "auto"
        else device_setting
    )
    device = torch.device(device_name)
    architecture = cast(Architecture, str(configuration["architecture"]))
    model = build_torchvision_model(architecture, len(labels), pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    pytorch_parts: list[np.ndarray[Any, Any]] = []
    with torch.inference_mode():
        for images, _ in test_loader:
            pytorch_parts.append(
                model(images.to(device, non_blocking=True))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    pytorch_logits = np.concatenate(pytorch_parts)
    parity = compare_classifier_outputs(
        pytorch_logits,
        test_logits,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
    )
    test_predictions = test_probabilities.argmax(axis=1)
    test_confidence = test_probabilities.max(axis=1)
    predictions_path = run_root / "locked_test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "path",
                "expected",
                "predicted",
                "confidence",
                "accepted",
                "correct",
            ),
        )
        writer.writeheader()
        for (path, target), predicted, confidence in zip(
            test_dataset.samples,
            test_predictions,
            test_confidence,
            strict=True,
        ):
            writer.writerow(
                {
                    "path": str(path.relative_to(project_path("."))),
                    "expected": labels[target],
                    "predicted": labels[int(predicted)],
                    "confidence": float(confidence),
                    "accepted": bool(confidence >= confidence_threshold),
                    "correct": bool(int(predicted) == target),
                }
            )
    artifacts["locked_test_predictions"] = str(
        predictions_path.relative_to(project_path("."))
    )
    metrics.update(
        {
            "confidence_threshold": confidence_threshold,
            "environment": _runtime_environment(device_name),
            "threshold_selection": threshold_selection,
            "evaluation_split": "test",
            "evaluation": test_metrics,
            "evaluation_observed_labels": test_metrics["observed_labels"],
            "evaluation_label_coverage": test_metrics["label_coverage"],
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "accuracy_percent": test_metrics["accuracy_percent"],
            "accuracy_correct": test_metrics["accuracy_correct"],
            "accuracy_total": test_metrics["accuracy_total"],
            "accuracy_95_ci": test_metrics["accuracy_95_ci"],
            "accepted_samples": test_metrics["accepted_samples"],
            "selective_coverage": test_metrics["selective_coverage"],
            "selective_accuracy": test_metrics["selective_accuracy"],
            "accepted_correct_rate": test_metrics["accepted_correct_rate"],
            "macro_f1_observed": test_metrics["macro_f1_observed"],
            "macro_f1_all_labels": test_metrics["macro_f1_all_labels"],
            "macro_f1_all_labels_percent": test_metrics["macro_f1_all_labels_percent"],
            "ece": test_metrics["ece"],
            "per_class": test_metrics["per_class"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "largest_confusion_pairs": test_metrics["largest_confusion_pairs"],
            "onnx_mean_latency_ms": test_elapsed * 1000.0 / len(test_targets),
            "onnx_parity": {**parity, "providers": providers, "split": "test"},
            "locked_test_evaluation": {
                "providers": providers,
                "validation_inference_seconds": validation_elapsed,
                "test_inference_seconds": test_elapsed,
                "threshold_selected_on": "validation",
            },
        }
    )
    calibration.update(
        {
            "confidence_threshold": confidence_threshold,
            "threshold_selection": threshold_selection,
        }
    )
    calibration_path.write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
