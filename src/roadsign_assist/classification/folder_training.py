from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import csv
import hashlib
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
from datetime import UTC, datetime
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
    resume: bool = False
    initialization_checkpoint: Path | None = None
    sampling_weights_path: Path | None = None
    use_separate_calibration: bool = False
    preserve_sign_extent: bool = False
    mild_geometry_augmentation: bool = False
    allow_label_expansion_initialization: bool = False
    freeze_backbone: bool = False
    backbone_learning_rate: float | None = None
    head_learning_rate: float | None = None


@dataclass(frozen=True)
class FolderClassifierRunPaths:
    run_root: Path
    checkpoint_root: Path
    checkpoint_path: Path
    latest_checkpoint_path: Path
    progress_path: Path
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
        latest_checkpoint_path=project_path("models/checkpoints") / safe_name / "latest.pt",
        progress_path=project_path("outputs/training") / safe_name / "progress.json",
        artifact_root=artifact_root,
        onnx_path=artifact_root / "sign_classifier.onnx",
        labels_path=artifact_root / "sign_classifier.labels.json",
        calibration_path=artifact_root / "sign_classifier.calibration.json",
    )


def prepare_classifier_run_paths(
    paths: FolderClassifierRunPaths,
    *,
    overwrite: bool,
    resume: bool = False,
) -> None:
    roots = (paths.run_root, paths.checkpoint_root, paths.artifact_root)
    existing = [path for path in roots if path.exists()]
    if existing and resume:
        if (paths.run_root / "metrics.json").is_file():
            raise FileExistsError(
                f"Classifier run {paths.run_root.name!r} is already complete; do not resume it."
            )
        if not paths.latest_checkpoint_path.is_file():
            raise FileNotFoundError(
                "Classifier run has existing output but no resumable latest checkpoint: "
                f"{paths.latest_checkpoint_path}"
            )
        return
    # The stage runner deliberately does not create a run directory before the
    # trainer starts.  Still, accepting an otherwise empty run directory makes
    # a first-run retry safe if PowerShell (or a user) created the directory
    # before an earlier launch failed.  It is not an overwrite: no checkpoint,
    # artifact, metric, or other file may be present.
    if (
        existing
        and all(path.is_dir() and not any(path.iterdir()) for path in existing)
        and not resume
        and not overwrite
    ):
        existing = []

    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Classifier run output already exists: {rendered}. "
            "Choose a unique --name or pass --overwrite deliberately."
        )
    for path in roots:
        path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a small status/report file for a second terminal to read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _serializable_config(config: FolderClassifierTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["data_root"] = str(config.data_root)
    if config.initialization_checkpoint is None:
        payload.pop("initialization_checkpoint", None)
    else:
        payload["initialization_checkpoint"] = str(config.initialization_checkpoint)
    if config.sampling_weights_path is None:
        payload.pop("sampling_weights_path", None)
    else:
        payload["sampling_weights_path"] = str(config.sampling_weights_path)
    if not config.use_separate_calibration:
        payload.pop("use_separate_calibration", None)
    if not config.preserve_sign_extent:
        payload.pop("preserve_sign_extent", None)
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


def _build_training_datasets(
    root: Path,
    labels: list[str],
    train_transform: Any,
    evaluation_transform: Any,
    *,
    include_test: bool,
) -> tuple[CropFolderDataset, CropFolderDataset, CropFolderDataset | None]:
    """Build only the splits explicitly authorized for this training run."""
    train_dataset = CropFolderDataset(root / "train", labels, train_transform)
    validation_dataset = CropFolderDataset(
        root / "validation",
        labels,
        evaluation_transform,
    )
    test_dataset = (
        CropFolderDataset(root / "test", labels, evaluation_transform)
        if include_test
        else None
    )
    return train_dataset, validation_dataset, test_dataset


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
    internal_exception = status == "approved_with_internal_academic_exception"
    if internal_exception:
        required = {
            "phase_b_authorized": True,
            "internal_academic_only": True,
            "internal_runtime_promotion_eligible": True,
            "external_data_or_model_release_allowed": False,
            "dvc_remote_push_allowed": False,
            "publication_prohibited": True,
        }
        invalid = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in required.items()
            if metadata.get(key) != expected
        }
        if invalid:
            raise ValueError(
                "Internal academic classifier release is missing required Phase-B safeguards: "
                f"{invalid}"
            )
    if status not in {"approved", "approved_with_internal_academic_exception"} and not allow_unreviewed_experiment:
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
    return allow_unreviewed_experiment or metadata.get("annotation_status") not in {
        "approved",
        "approved_with_internal_academic_exception",
    }


def _transforms(image_size: int) -> tuple[Any, Any]:
    from torchvision import transforms

    from roadsign_assist.classification.preprocessing import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        build_evaluation_transform,
    )

    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
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
    evaluate = build_evaluation_transform(image_size)
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
    selected = dict(selected)
    selection_boundary = float(selected["threshold"])
    accepted_at_boundary = confidence >= selection_boundary
    if accepted_at_boundary.any() and (~accepted_at_boundary).any():
        lowest_accepted = float(confidence[accepted_at_boundary].min())
        highest_rejected = float(confidence[~accepted_at_boundary].max())
        if lowest_accepted > highest_rejected:
            stable_threshold = (lowest_accepted + highest_rejected) / 2.0
            selected["threshold"] = stable_threshold
            selected["selection_boundary_threshold"] = selection_boundary
            selected["threshold_stability_margin"] = min(
                lowest_accepted - stable_threshold,
                stable_threshold - highest_rejected,
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


def repair_validation_only_classifier_export(run_name: str) -> dict[str, Any]:
    """Recheck a completed checkpoint and repair a threshold-boundary parity failure.

    This path reads validation only, never performs optimizer steps, and preserves the
    trained checkpoints and history. It is intended for a run whose ONNX logits and
    probabilities already satisfy the numerical parity limits but whose tuned threshold
    was placed directly on a validation confidence value.
    """
    import torch
    from torch.utils.data import DataLoader

    safe_name = _validate_run_name(run_name)
    paths = classifier_run_paths(safe_name, experimental=True)
    metrics_path = paths.run_root / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    if metrics.get("evaluation_split") != "validation":
        raise ValueError("Parity repair is restricted to validation-only classifier runs")
    if metrics.get("test_split_opened") is not False:
        raise ValueError("Parity repair refuses any run that opened the internal test split")
    prior_parity = cast(dict[str, Any], metrics.get("onnx_parity", {}))
    if prior_parity.get("passed") is True:
        return metrics
    numerical_limits_passed = (
        float(prior_parity.get("maximum_absolute_probability_difference", math.inf))
        <= float(prior_parity.get("maximum_probability_tolerance", 0.02))
        and float(prior_parity.get("mean_absolute_probability_difference", math.inf))
        <= float(prior_parity.get("mean_probability_tolerance", 1e-3))
        and float(prior_parity.get("top1_agreement", 0.0))
        >= float(prior_parity.get("minimum_top1_agreement", 0.995))
    )
    if not numerical_limits_passed:
        raise ValueError("Parity repair refuses a classifier with numerical output drift")

    configuration = cast(dict[str, Any], metrics["configuration"])
    data_root = project_path(str(configuration["data_root"]))
    labels = cast(list[str], json.loads((data_root / "labels.json").read_text(encoding="utf-8")))
    validation_dataset = CropFolderDataset(
        data_root / "validation",
        labels,
        _transforms(int(configuration["image_size"]))[1],
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(configuration["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torchvision_model(configuration["architecture"], len(labels)).to(device)
    checkpoint = torch.load(paths.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logits_parts: list[Any] = []
    target_parts: list[Any] = []
    with torch.inference_mode():
        for images, targets in validation_loader:
            logits_parts.append(model(images.to(device, non_blocking=True)).detach().cpu())
            target_parts.append(targets.detach().cpu())
    if not logits_parts:
        raise ValueError("Parity repair requires validation samples")
    pytorch_logits = torch.cat(logits_parts).numpy().astype(np.float32)
    target_array = torch.cat(target_parts).numpy().astype(np.int64)
    temperature = float(metrics["temperature"])
    probabilities = (
        torch.softmax(torch.from_numpy(pytorch_logits) / temperature, dim=1)
        .numpy()
        .astype(np.float32)
    )
    threshold_selection = select_confidence_threshold(
        probabilities,
        target_array,
        fallback_threshold=float(configuration["confidence_threshold"]),
        target_selective_accuracy=float(configuration["target_selective_accuracy"]),
    )
    stable_threshold = float(threshold_selection["threshold"])
    validation_metrics = _probability_metrics(
        probabilities,
        target_array,
        labels,
        confidence_threshold=stable_threshold,
        loss=float(cast(dict[str, Any], metrics["validation"])["loss"]),
    )
    onnx_logits, providers, elapsed = _onnx_logits(paths.onnx_path, validation_loader)
    parity = compare_classifier_outputs(
        pytorch_logits,
        onnx_logits,
        temperature=temperature,
        confidence_threshold=stable_threshold,
    )
    if not parity["passed"]:
        raise RuntimeError(
            "Classifier ONNX parity still fails after threshold stabilization: "
            f"acceptance agreement {parity['acceptance_agreement']:.6f}"
        )

    metrics["confidence_threshold"] = stable_threshold
    metrics["threshold_selection"] = threshold_selection
    metrics["validation"] = validation_metrics
    metrics["evaluation"] = validation_metrics
    compatibility_fields = (
        "loss",
        "accuracy",
        "accuracy_percent",
        "accuracy_correct",
        "accuracy_total",
        "accuracy_95_ci",
        "accepted_samples",
        "selective_coverage",
        "selective_accuracy",
        "accepted_correct_rate",
        "macro_f1_observed",
        "macro_f1_all_labels",
        "macro_f1_all_labels_percent",
        "ece",
        "per_class",
        "confusion_matrix",
        "largest_confusion_pairs",
    )
    for key in compatibility_fields:
        metrics[key] = validation_metrics[key]
    metrics["onnx_parity"] = {**parity, "providers": providers}
    metrics["onnx_mean_latency_ms"] = elapsed * 1000.0 / len(target_array)
    metrics["post_training_parity_repair"] = {
        "performed": True,
        "training_repeated": False,
        "split": "validation",
        "previous_threshold": prior_parity.get("confidence_threshold"),
        "stabilized_threshold": stable_threshold,
        "reason": "tuned threshold lay on a validation confidence boundary",
    }
    calibration = cast(
        dict[str, Any],
        json.loads(paths.calibration_path.read_text(encoding="utf-8")),
    )
    calibration["confidence_threshold"] = stable_threshold
    calibration["threshold_stabilized_for_onnx_parity"] = True
    _write_json(paths.calibration_path, calibration)
    _write_json(metrics_path, metrics)
    _write_json(
        paths.progress_path,
        {
            "schema_version": "1.1",
            "status": "completed",
            "run_name": safe_name,
            "epoch": int(configuration["epochs"]),
            "epochs": int(configuration["epochs"]),
            "best_epoch": int(metrics["best_epoch"]),
            "best_validation_macro_f1": float(metrics["best_validation_macro_f1"]),
            "post_training_parity_repair": True,
            "metrics": str(metrics_path.relative_to(project_path("."))),
        },
    )
    return metrics


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
    prepare_classifier_run_paths(
        paths,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    git_commit = _git_commit()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_transform, evaluation_transform = _transforms(config.image_size)
    if config.preserve_sign_extent:
        from torchvision import transforms
        if config.mild_geometry_augmentation:
            from roadsign_assist.classification.preprocessing import IMAGENET_MEAN, IMAGENET_STD

            train_transform = transforms.Compose([
                transforms.Resize(
                    (config.image_size, config.image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.18)],
                    p=0.65,
                ),
                transforms.RandomAffine(
                    degrees=7,
                    translate=(0.04, 0.04),
                    scale=(0.92, 1.08),
                    shear=3,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.RandomPerspective(distortion_scale=0.08, p=0.20, fill=0),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.9))], p=0.12),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15)], p=0.5),
                evaluation_transform,
            ])
    train_dataset, validation_dataset, test_dataset = _build_training_datasets(
        root,
        labels,
        train_transform,
        evaluation_transform,
        include_test=config.evaluate_test,
    )
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
    if config.sampling_weights_path is not None:
        weights_path = project_path(config.sampling_weights_path)
        weights = json.loads(weights_path.read_text(encoding="utf-8"))
        expected_keys = {str(path.relative_to(root).as_posix()) for path, _ in train_dataset.samples}
        if set(weights) != expected_keys:
            raise ValueError("Sampling weights must exactly cover the training files")
        sample_weights = [float(weights[path.relative_to(root).as_posix()]) for path, _ in train_dataset.samples]
        if any(not math.isfinite(weight) or weight <= 0 for weight in sample_weights):
            raise ValueError("Sampling weights must be finite and positive")
        _write_json(paths.run_root / "sampling.json", {
            "source": str(weights_path),
            "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "samples": len(sample_weights),
        })
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
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            persistent_workers=config.workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
        if test_dataset is not None
        else None
    )

    device_name = (
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.device == "auto"
        else config.device
    )
    device = torch.device(device_name)
    model = build_torchvision_model(
        config.architecture, len(labels), pretrained=config.initialization_checkpoint is None
    ).to(device)
    if config.initialization_checkpoint is not None:
        from roadsign_assist.classification.initialization import initialize_classifier_weights

        initialization = initialize_classifier_weights(
            model,
            project_path(config.initialization_checkpoint),
            labels=labels,
            architecture=config.architecture,
            image_size=config.image_size,
            allow_label_expansion=config.allow_label_expansion_initialization,
        )
        _write_json(paths.run_root / "initialization.json", initialization)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    named_parameters = list(model.named_parameters())
    if config.freeze_backbone:
        for name, parameter in named_parameters:
            parameter.requires_grad = name.startswith("classifier.")
    backbone_parameters = [
        parameter
        for name, parameter in named_parameters
        if parameter.requires_grad and not name.startswith("classifier.")
    ]
    head_parameters = [
        parameter
        for name, parameter in named_parameters
        if parameter.requires_grad and name.startswith("classifier.")
    ]
    if not head_parameters:
        raise ValueError("Classifier architecture exposed no classifier.* head parameters")
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": config.backbone_learning_rate or config.learning_rate,
                "group_name": "backbone",
            }
        )
    parameter_groups.append(
        {
            "params": head_parameters,
            "lr": config.head_learning_rate or config.learning_rate,
            "group_name": "classifier_head",
        }
    )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs),
    )
    use_amp = device.type == "cuda"
    scaler = torch.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0
    start_epoch = 1
    if config.resume:
        latest: dict[str, Any] = torch.load(
            paths.latest_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        saved_config = dict(latest.get("config", {}))
        expected_config = _serializable_config(config)
        saved_config.pop("resume", None)
        expected_config.pop("resume", None)
        if saved_config != expected_config:
            raise ValueError(
                "Refusing to resume a classifier run with a different configuration"
            )
        model.load_state_dict(latest["state_dict"])
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        scheduler.load_state_dict(latest["scheduler_state_dict"])
        scaler.load_state_dict(latest["scaler_state_dict"])
        history = list(latest["history"])
        best_f1 = float(latest["best_validation_macro_f1"])
        best_epoch = int(latest["best_epoch"])
        start_epoch = int(latest["next_epoch"])
        generator.set_state(latest["sampler_generator_state"])
        random.setstate(latest["python_random_state"])
        np.random.set_state(latest["numpy_random_state"])
        torch.set_rng_state(latest["torch_random_state"])
        if torch.cuda.is_available() and latest.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(latest["cuda_rng_state"])
        print(
            f"Resuming classifier run {config.run_name} from epoch {start_epoch} "
            f"after best epoch {best_epoch}."
        )

    gpu_total_gib = (
        torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if device.type == "cuda"
        else 0.0
    )

    def emit_live_progress(
        *,
        phase: str,
        epoch: int,
        batch: int,
        batches: int,
        running_loss: float,
    ) -> None:
        phase_fraction = batch / max(1, batches)
        epoch_fraction = (
            0.90 * phase_fraction
            if phase == "train"
            else 0.90 + (0.10 * phase_fraction)
        )
        completed_fraction = ((epoch - 1) + epoch_fraction) / max(1, config.epochs)
        elapsed_seconds = time.perf_counter() - training_started
        session_total_epochs = config.epochs - start_epoch + 1
        session_completed_epochs = (epoch - start_epoch) + epoch_fraction
        eta_seconds = (
            elapsed_seconds
            * (session_total_epochs - session_completed_epochs)
            / session_completed_epochs
            if session_completed_epochs > 0.0
            else None
        )
        payload: dict[str, Any] = {
            "schema_version": "1.1",
            "status": "running",
            "run_name": config.run_name,
            "phase": phase,
            "epoch": epoch,
            "epochs": config.epochs,
            "batch": batch,
            "batches": batches,
            "phase_percent": round(100.0 * phase_fraction, 2),
            "overall_percent": round(100.0 * completed_fraction, 2),
            "running_loss": round(running_loss, 6),
            "elapsed_seconds": round(elapsed_seconds, 1),
            "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
            "resume_supported_from_completed_epoch": max(0, epoch - 1),
            "internal_test_opened": False,
        }
        gpu_summary = ""
        if device.type == "cuda":
            gpu_allocated_gib = torch.cuda.memory_allocated(device) / (1024**3)
            payload["gpu_memory_allocated_gib"] = round(gpu_allocated_gib, 3)
            payload["gpu_memory_total_gib"] = round(gpu_total_gib, 3)
            gpu_summary = f" | GPU {gpu_allocated_gib:.2f}/{gpu_total_gib:.2f} GiB"
        _write_json(paths.progress_path, payload)
        eta_summary = f"{eta_seconds:.0f}s" if eta_seconds is not None else "calculating"
        print(
            f"{phase.upper()} epoch {epoch}/{config.epochs} batch {batch}/{batches} "
            f"| phase {payload['phase_percent']:.2f}% "
            f"| overall {payload['overall_percent']:.2f}% "
            f"| loss {running_loss:.5f}{gpu_summary} "
            f"| elapsed {elapsed_seconds:.0f}s "
            f"| ETA {eta_summary}",
            flush=True,
        )

    def collect(
        loader: Any,
        *,
        phase: str = "validation",
        epoch: int | None = None,
        report_progress: bool = False,
    ) -> tuple[Any, Any, float]:
        model.eval()
        logits_parts: list[Any] = []
        labels_parts: list[Any] = []
        loss_sum = 0.0
        sample_count = 0
        total_batches = len(loader)
        report_every = max(1, total_batches // 100)
        with torch.inference_mode():
            for batch_index, (images, targets) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, targets)
                logits_parts.append(logits.detach().cpu())
                labels_parts.append(targets.detach().cpu())
                loss_sum += float(loss.item()) * len(targets)
                sample_count += len(targets)
                if (
                    report_progress
                    and epoch is not None
                    and (
                        batch_index == 1
                        or batch_index == total_batches
                        or batch_index % report_every == 0
                    )
                ):
                    emit_live_progress(
                        phase=phase,
                        epoch=epoch,
                        batch=batch_index,
                        batches=total_batches,
                        running_loss=loss_sum / max(1, sample_count),
                    )
        if not logits_parts:
            return torch.empty((0, len(labels))), torch.empty((0,), dtype=torch.long), 0.0
        return (
            torch.cat(logits_parts),
            torch.cat(labels_parts),
            loss_sum / max(1, sample_count),
        )

    for epoch in range(start_epoch, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        train_loss = 0.0
        seen = 0
        total_train_batches = len(train_loader)
        report_every = max(1, total_train_batches // 100)
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
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
            if (
                batch_index == 1
                or batch_index == total_train_batches
                or batch_index % report_every == 0
            ):
                emit_live_progress(
                    phase="train",
                    epoch=epoch,
                    batch=batch_index,
                    batches=total_train_batches,
                    running_loss=train_loss / max(1, seen),
                )
        scheduler.step()

        validation_logits, validation_targets, validation_loss = collect(
            validation_loader,
            phase="validation",
            epoch=epoch,
            report_progress=True,
        )
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
        if device.type == "cuda":
            record["gpu_peak_memory_gib"] = round(
                torch.cuda.max_memory_allocated(device) / (1024**3), 3
            )
        history.append(record)
        print(
            "Epoch {epoch}/{total}: val_macro_f1={f1:.4f}, val_accuracy={accuracy:.4f}, "
            "val_loss={loss:.4f}, seconds={seconds:.1f}{memory}".format(
                epoch=epoch,
                total=config.epochs,
                f1=macro_f1,
                accuracy=float(record["validation_accuracy"]),
                loss=validation_loss,
                seconds=float(record["seconds"]),
                memory=(
                    f", gpu_peak_gib={record['gpu_peak_memory_gib']:.3f}"
                    if "gpu_peak_memory_gib" in record
                    else ""
                ),
            )
        )
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
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "labels": labels,
                "config": _serializable_config(config),
                "metadata": metadata,
                "history": history,
                "best_validation_macro_f1": best_f1,
                "best_epoch": best_epoch,
                "next_epoch": epoch + 1,
                "sampler_generator_state": generator.get_state(),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
            paths.latest_checkpoint_path,
        )
        _write_json(
            paths.progress_path,
            {
                "schema_version": "1.1",
                "status": "running",
                "run_name": config.run_name,
                "phase": "checkpoint_saved",
                "epoch": epoch,
                "epochs": config.epochs,
                "batch": 1,
                "batches": 1,
                "phase_percent": 100.0,
                "overall_percent": round(100.0 * epoch / config.epochs, 2),
                "running_loss": round(float(record["train_loss"]), 6),
                "elapsed_seconds": round(time.perf_counter() - training_started, 1),
                "eta_seconds": None,
                "resume_supported_from_completed_epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_f1,
                "last_epoch": record,
                "resume_supported": True,
            },
        )

    checkpoint = torch.load(paths.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    validation_logits, validation_targets, _ = collect(validation_loader)

    fitting_logits, fitting_targets = validation_logits, validation_targets
    if config.use_separate_calibration:
        calibration_dataset = CropFolderDataset(root / "calibration", labels, evaluation_transform)
        if not calibration_dataset.samples:
            raise ValueError("Separate calibration was requested but its folder is empty")
        calibration_loader = DataLoader(calibration_dataset, batch_size=config.batch_size,
                                        shuffle=False, num_workers=config.workers)
        fitting_logits, fitting_targets, _ = collect(calibration_loader)

    temperature = nn.Parameter(torch.ones(1, device=device))
    calibration_optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=50)
    calibration_logits = fitting_logits.to(device)
    calibration_targets = fitting_targets.to(device)

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
        threshold_probabilities = torch.softmax(fitting_logits / calibrated_temperature, dim=1).numpy().astype(np.float32)
        threshold_selection = select_confidence_threshold(
            threshold_probabilities,
            fitting_targets.numpy().astype(np.int64),
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
        if test_loader is None:
            raise AssertionError("Test evaluation was requested without a test loader")
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
        "internal_academic_only": bool(metadata.get("internal_academic_only", False)),
        "publication_prohibited": bool(metadata.get("publication_prohibited", False)),
        "dataset_id": metadata.get("dataset_id", root.name),
        "git_commit": git_commit,
        "environment": _runtime_environment(device_name),
        "configuration": _serializable_config(config),
        "architecture": config.architecture,
        "labels": len(labels),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset) if test_dataset is not None else 0,
        "test_split_opened": test_dataset is not None,
        "missing_train_labels": missing_train_labels,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "temperature": calibrated_temperature,
        "confidence_threshold": effective_confidence_threshold,
        "threshold_selection": threshold_selection,
        "selection_split": "validation",
        "calibration_split": "calibration" if config.use_separate_calibration else "validation",
        "calibration_samples": len(fitting_targets),
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
                "runtime_threshold_authoritative": True,
                "image_size": config.image_size,
                "experimental": metrics["experimental"],
                "release_status": metadata.get("release_status", "unspecified"),
                "dataset_id": metadata.get("dataset_id", root.name),
                "source_run": config.run_name,
                "internal_academic_only": bool(metadata.get("internal_academic_only", False)),
                "publication_prohibited": bool(metadata.get("publication_prohibited", False)),
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
    _write_json(paths.run_root / "metrics.json", metrics)
    _write_json(
        paths.progress_path,
        {
            "schema_version": "1.0",
            "status": "completed",
            "run_name": config.run_name,
            "epoch": config.epochs,
            "epochs": config.epochs,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "metrics": str((paths.run_root / "metrics.json").relative_to(project_path("."))),
        },
    )
    if not parity["passed"]:
        raise RuntimeError(
            "Classifier ONNX parity failed: "
            f"max_probability={parity['maximum_absolute_probability_difference']:.6f} "
            f"(limit {parity['maximum_probability_tolerance']:.6f}), "
            f"mean_probability={parity['mean_absolute_probability_difference']:.6f} "
            f"(limit {parity['mean_probability_tolerance']:.6f}), "
            f"top1_agreement={parity['top1_agreement']:.6f} "
            f"(minimum {parity['minimum_top1_agreement']:.6f}), "
            f"acceptance_agreement={parity['acceptance_agreement']:.6f} "
            f"(minimum {parity['minimum_acceptance_agreement']:.6f})"
        )
    return metrics


def promote_classifier_candidate(
    run_name: str,
    *,
    overwrite: bool = False,
    internal_only: bool = False,
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
    is_internal_v3 = metrics.get("dataset_id") == "classifier_production_78_v3_20260829"
    if is_internal_v3:
        if not internal_only:
            raise ValueError(
                "The v3 classifier is internal-academic-only. Pass the explicit internal-only "
                "promotion target; public/DVC promotion is prohibited."
            )
        if metrics.get("internal_academic_only") is not True or metrics.get("publication_prohibited") is not True:
            raise ValueError("The v3 classifier metrics are missing internal-use safeguards")
        if float(metrics.get("macro_f1_all_labels", 0.0)) < 0.929069704804999:
            raise ValueError("v3 classifier does not meet the current runtime locked-test macro-F1")
        if float(metrics.get("accuracy", 0.0)) < 0.9458874458874459:
            raise ValueError("v3 classifier does not meet the current runtime locked-test accuracy")
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
    dvc_pointer = project_path("models/exported/runtime.dvc")
    dvc_pointer_hash_before = (
        hashlib.sha256(dvc_pointer.read_bytes()).hexdigest() if dvc_pointer.is_file() else None
    )
    backup_id: str | None = None
    if existing:
        backup_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_root = project_path("models/exported/runtime_backups") / backup_id
        backup_root.mkdir(parents=True, exist_ok=False)
        for source in (*existing, export_root / "sign_classifier.runtime.json"):
            if source.is_file():
                shutil.copy2(source, backup_root / source.name)
        _write_json(
            backup_root / "backup.json",
            {
                "schema_version": "1.0",
                "backup_id": backup_id,
                "source_run": safe_name,
                "reason": "internal_local_runtime_promotion" if internal_only else "runtime_promotion",
            },
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
        "internal_academic_only": bool(metrics.get("internal_academic_only", False)),
        "publication_prohibited": bool(metrics.get("publication_prohibited", False)),
        "dvc_remote_push_allowed": False if is_internal_v3 else None,
        "local_runtime_only": internal_only,
        "backup_id": backup_id,
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
    _write_json(manifest_path, manifest)
    dvc_pointer_hash_after = (
        hashlib.sha256(dvc_pointer.read_bytes()).hexdigest() if dvc_pointer.is_file() else None
    )
    if dvc_pointer_hash_before != dvc_pointer_hash_after:
        raise RuntimeError("Promotion unexpectedly modified the shared runtime DVC pointer")
    return manifest


def rollback_classifier_runtime(backup_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"\d{8}T\d{6}Z", backup_id):
        raise ValueError("Runtime backup ID must be a UTC timestamp generated by promotion")
    backup_root = project_path("models/exported/runtime_backups") / backup_id
    backup_manifest = backup_root / "backup.json"
    if not backup_manifest.is_file():
        raise FileNotFoundError(backup_manifest)
    export_root = project_path("models/exported/runtime")
    required = (
        "sign_classifier.onnx",
        "sign_classifier.labels.json",
        "sign_classifier.calibration.json",
        "sign_classifier.runtime.json",
    )
    missing = [name for name in required if not (backup_root / name).is_file()]
    if missing:
        raise ValueError(f"Runtime backup is incomplete: {missing}")
    export_root.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(backup_root / name, export_root / name)
    return {
        "schema_version": "1.0",
        "restored_backup_id": backup_id,
        "runtime_root": str(export_root.relative_to(project_path("."))),
        "publication_prohibited": True,
    }


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
    if str(metrics.get("dataset_id")) == "classifier_production_78_v3_20260829":
        from roadsign_assist.classification.phase_b import locked_test_slice_report

        slice_report: dict[str, Any] | None = locked_test_slice_report(
            paths=[path for path, _ in test_dataset.samples],
            targets=test_targets,
            predictions=test_predictions,
            labels=labels,
        )
    else:
        slice_report = None
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
            "locked_test_slices": slice_report,
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
