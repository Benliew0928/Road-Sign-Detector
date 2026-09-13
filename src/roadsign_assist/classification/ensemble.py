"""Validation-gated equal-logit Phase-B ensemble export and locked evaluation."""

from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportAttributeAccessIssue=false
import json
from typing import Any, cast

import numpy as np

from roadsign_assist.classification.folder_training import (
    CropFolderDataset,
    _onnx_logits,
    _probability_metrics,
    _transforms,
    _write_json,
    compare_classifier_outputs,
    select_confidence_threshold,
)
from roadsign_assist.classification.phase_b import PHASE_B_RELEASE_ID, locked_test_slice_report
from roadsign_assist.classification.training import Architecture, build_torchvision_model
from roadsign_assist.paths import project_path


def _run_metrics(run_name: str) -> dict[str, Any]:
    path = project_path("outputs/training") / run_name / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    metrics: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("dataset_id") != PHASE_B_RELEASE_ID or metrics.get("experimental") is True:
        raise ValueError(f"Ensemble member {run_name} is not an approved Phase-B v3 run")
    if metrics.get("evaluation_split") != "validation":
        raise ValueError(f"Ensemble member {run_name} has already opened locked test data")
    return metrics


def _probabilities(logits: np.ndarray[Any, Any], temperature: float) -> np.ndarray[Any, Any]:
    shifted = np.asarray(logits, dtype=np.float32) / max(0.05, min(10.0, temperature))
    shifted -= shifted.max(axis=1, keepdims=True)
    value = np.exp(shifted)
    return value / value.sum(axis=1, keepdims=True)


def _fit_temperature(logits: np.ndarray[Any, Any], targets: np.ndarray[Any, Any], device: str) -> float:
    import torch

    value = torch.nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([value], lr=0.05, max_iter=50)
    tensors = torch.as_tensor(logits, device=device), torch.as_tensor(targets, device=device)

    def closure() -> Any:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(tensors[0] / value.clamp(0.05, 10.0), tensors[1])
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(value.detach().clamp(0.05, 10.0).item())


def _ensemble_model(metrics: list[dict[str, Any]], labels: list[str], device: str) -> Any:
    import torch
    import torch.nn as nn

    models: list[Any] = []
    for metric in metrics:
        configuration = metric["configuration"]
        name = str(configuration["architecture"])
        model = build_torchvision_model(cast(Architecture, name), len(labels), pretrained=False).to(device)
        checkpoint_path = project_path(str(metric["artifacts"]["checkpoint"]))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)

    class EqualLogitEnsemble(nn.Module):
        def __init__(self, members: list[Any]) -> None:
            super().__init__()
            self.members = nn.ModuleList(members)

        def forward(self, image: Any) -> Any:
            return sum(member(image) for member in self.members) / len(self.members)

    return EqualLogitEnsemble(models).to(device).eval()


def prepare_phase_b_ensemble(
    run_names: list[str],
    *,
    name: str,
    target_selective_accuracy: float = 0.98,
) -> dict[str, Any]:
    """Export a two-model ensemble using validation only; it never opens test."""
    import torch
    from torch.utils.data import DataLoader

    if len(run_names) != 2 or len(set(run_names)) != 2:
        raise ValueError("Phase-B ensemble requires exactly two distinct runs")
    metrics = [_run_metrics(run_name) for run_name in run_names]
    configurations = [dict(metric["configuration"]) for metric in metrics]
    image_sizes = {int(configuration["image_size"]) for configuration in configurations}
    datasets = {str(metric["dataset_id"]) for metric in metrics}
    if image_sizes != {320} or datasets != {PHASE_B_RELEASE_ID}:
        raise ValueError("Phase-B ensemble requires two 320 px models from the v3 release")
    families = {
        "convnext" if str(configuration["architecture"]).startswith("convnext") else "efficientnet"
        for configuration in configurations
    }
    if families != {"convnext", "efficientnet"}:
        raise ValueError("Phase-B ensemble requires one ConvNeXt and one EfficientNet member")
    labels = list(json.loads(project_path(str(metrics[0]["artifacts"]["labels"])).read_text(encoding="utf-8")))
    if labels != list(json.loads(project_path(str(metrics[1]["artifacts"]["labels"])).read_text(encoding="utf-8"))):
        raise ValueError("Phase-B ensemble members have different label vocabularies")
    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _ensemble_model(metrics, labels, selected_device)
    data_root = project_path(str(configurations[0]["data_root"]))
    loader = DataLoader(
        CropFolderDataset(data_root / "validation", labels, _transforms(320)[1]),
        batch_size=min(int(configuration["batch_size"]) for configuration in configurations),
        shuffle=False,
        num_workers=0,
    )
    torch_logits: list[np.ndarray[Any, Any]] = []
    targets: list[np.ndarray[Any, Any]] = []
    with torch.inference_mode():
        for images, batch_targets in loader:
            torch_logits.append(model(images.to(selected_device)).cpu().numpy().astype(np.float32))
            targets.append(batch_targets.numpy().astype(np.int64))
    validation_logits = np.concatenate(torch_logits)
    validation_targets = np.concatenate(targets)
    temperature = _fit_temperature(validation_logits, validation_targets, selected_device)
    probabilities = _probabilities(validation_logits, temperature)
    threshold = select_confidence_threshold(
        probabilities,
        validation_targets,
        fallback_threshold=0.72,
        target_selective_accuracy=target_selective_accuracy,
    )
    confidence_threshold = float(threshold["threshold"])
    validation = _probability_metrics(
        probabilities,
        validation_targets,
        labels,
        confidence_threshold=confidence_threshold,
        loss=float(-np.log(np.maximum(probabilities[np.arange(len(validation_targets)), validation_targets], 1e-12)).mean()),
    )
    run_root = project_path("outputs/training") / name
    artifact_root = project_path("models/candidates") / name
    if run_root.exists() or artifact_root.exists():
        raise FileExistsError(f"Ensemble output already exists for {name}")
    run_root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    onnx_path = artifact_root / "sign_classifier.onnx"
    labels_path = artifact_root / "sign_classifier.labels.json"
    calibration_path = artifact_root / "sign_classifier.calibration.json"
    sample = torch.zeros((1, 3, 320, 320), device=selected_device)
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
    labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    calibration = {
        "schema_version": "2.0",
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "runtime_threshold_authoritative": True,
        "image_size": 320,
        "dataset_id": PHASE_B_RELEASE_ID,
        "release_status": "production_candidate_with_policy_exceptions",
        "source_run": name,
        "internal_academic_only": True,
        "publication_prohibited": True,
        "ensemble_members": run_names,
    }
    _write_json(calibration_path, calibration)
    onnx_logits, providers, elapsed = _onnx_logits(onnx_path, loader)
    parity = compare_classifier_outputs(
        validation_logits,
        onnx_logits,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
    )
    best_single = max(float(metric["validation"]["macro_f1_all_labels"]) for metric in metrics)
    best_coverage = max(float(metric["validation"]["selective_coverage"]) for metric in metrics)
    qualified = bool(
        float(validation["macro_f1_all_labels"]) >= best_single + 0.0025
        and float(validation["selective_coverage"]) >= best_coverage - 0.01
    )
    report = {
        "schema_version": "1.0",
        "run_name": name,
        "ensemble": True,
        "ensemble_members": run_names,
        "dataset_id": PHASE_B_RELEASE_ID,
        "release_status": "production_candidate_with_policy_exceptions",
        "internal_academic_only": True,
        "publication_prohibited": True,
        "experimental": False,
        "configuration": {"data_root": str(data_root.relative_to(project_path("."))), "image_size": 320, "batch_size": loader.batch_size},
        "selection_split": "validation",
        "evaluation_split": "validation",
        "validation": validation,
        "evaluation": validation,
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "threshold_selection": threshold,
        "ensemble_qualification": {
            "qualified": qualified,
            "best_single_validation_macro_f1": best_single,
            "best_single_selective_coverage": best_coverage,
            "minimum_macro_f1_gain": 0.0025,
            "maximum_selective_coverage_loss": 0.01,
        },
        "onnx_parity": {**parity, "providers": providers},
        "onnx_mean_latency_ms": elapsed * 1000.0 / len(validation_targets),
        "artifacts": {
            "onnx": str(onnx_path.relative_to(project_path("."))),
            "labels": str(labels_path.relative_to(project_path("."))),
            "calibration": str(calibration_path.relative_to(project_path("."))),
        },
    }
    _write_json(run_root / "metrics.json", report)
    if not parity["passed"]:
        raise RuntimeError("Phase-B ensemble ONNX parity failed")
    return report


def evaluate_phase_b_ensemble(name: str, *, overwrite: bool = False) -> dict[str, Any]:
    """Open locked test exactly once for an already validation-qualified ensemble."""
    import torch
    from torch.utils.data import DataLoader

    run_root = project_path("outputs/training") / name
    metrics_path = run_root / "metrics.json"
    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("evaluation_split") == "test" and not overwrite:
        raise FileExistsError("This ensemble already has a locked test evaluation")
    if metrics.get("ensemble_qualification", {}).get("qualified") is not True:
        raise ValueError("Only a validation-qualified ensemble may open locked test data")
    labels = list(json.loads(project_path(str(metrics["artifacts"]["labels"])).read_text(encoding="utf-8")))
    data_root = project_path(str(metrics["configuration"]["data_root"]))
    loader = DataLoader(CropFolderDataset(data_root / "test", labels, _transforms(320)[1]), batch_size=12, shuffle=False)
    onnx_path = project_path(str(metrics["artifacts"]["onnx"]))
    logits, providers, elapsed = _onnx_logits(onnx_path, loader)
    targets = np.asarray([target for _, target in loader.dataset.samples], dtype=np.int64)
    temperature = float(metrics["temperature"])
    probabilities = _probabilities(logits, temperature)
    test = _probability_metrics(
        probabilities,
        targets,
        labels,
        confidence_threshold=float(metrics["confidence_threshold"]),
        loss=float(-np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)).mean()),
    )
    predictions = probabilities.argmax(axis=1)
    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    member_metrics = [_run_metrics(run_name) for run_name in metrics["ensemble_members"]]
    torch_model = _ensemble_model(member_metrics, labels, selected_device)
    pytorch_parts: list[np.ndarray[Any, Any]] = []
    with torch.inference_mode():
        for images, _ in loader:
            pytorch_parts.append(
                torch_model(images.to(selected_device)).detach().cpu().numpy().astype(np.float32)
            )
    parity = compare_classifier_outputs(
        np.concatenate(pytorch_parts),
        logits,
        temperature=temperature,
        confidence_threshold=float(metrics["confidence_threshold"]),
    )
    metrics.update(
        {
            "evaluation_split": "test",
            "evaluation": test,
            "accuracy": test["accuracy"],
            "macro_f1_all_labels": test["macro_f1_all_labels"],
            "accuracy_percent": test["accuracy_percent"],
            "locked_test_slices": locked_test_slice_report(
                paths=[path for path, _ in loader.dataset.samples],
                targets=targets,
                predictions=predictions,
                labels=labels,
            ),
            "locked_test_evaluation": {"providers": providers, "test_inference_seconds": elapsed},
            "onnx_parity": {**parity, "providers": providers, "split": "test"},
        }
    )
    _write_json(metrics_path, metrics)
    return metrics
