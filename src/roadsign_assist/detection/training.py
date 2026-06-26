from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from roadsign_assist.paths import project_path


@dataclass(frozen=True)
class SegmenterTrainingConfig:
    data_yaml: Path
    base_model: str = "yolo26n-seg.pt"
    image_size: int = 640
    epochs: int = 100
    batch_size: int = 16
    device: str = "0"
    workers: int = 4
    seed: int = 2513
    run_name: str = "malaysia_sign_segmenter"
    allow_unreviewed_experiment: bool = False


@dataclass(frozen=True)
class DetectorTrainingConfig:
    data_yaml: Path
    task: Literal["detect", "segment"] = "detect"
    base_model: str = "yolo26n.pt"
    image_size: int = 640
    epochs: int = 100
    batch_size: int = 16
    device: str = "0"
    workers: int = 4
    seed: int = 2513
    run_name: str = "malaysia_sign_detector"
    allow_unreviewed_experiment: bool = False


@dataclass(frozen=True)
class DetectorExportConfig:
    checkpoint: Path
    data_yaml: Path
    task: Literal["detect", "segment"] = "detect"
    image_size: int = 640
    device: str = "0"
    artifact_name: str = "sign_detector"
    parity_tolerance: float = 0.02
    allow_unreviewed_experiment: bool = False


def validate_training_data(
    data_yaml: Path,
    *,
    allow_unreviewed_experiment: bool,
) -> dict[str, Any]:
    import json

    resolved = project_path(data_yaml)
    metadata_path = resolved.parent / "dataset_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Dataset review metadata is required before training: {metadata_path}"
        )
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("coursework_images_included", -1)) != 0:
        raise ValueError("Coursework acceptance images cannot be used for training")
    status = str(metadata.get("annotation_status", ""))
    if status != "approved" and not allow_unreviewed_experiment:
        raise ValueError(
            f"Dataset annotation status is {status!r}; "
            "use allow_unreviewed_experiment only for an explicitly experimental run"
        )
    return metadata


def compare_metric_parity(
    pytorch_metrics: Mapping[str, Any],
    onnx_metrics: Mapping[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Compare shared scalar validation metrics between model runtimes."""
    differences: dict[str, float] = {}
    for key in sorted(pytorch_metrics.keys() & onnx_metrics.keys()):
        pytorch_value = pytorch_metrics[key]
        onnx_value = onnx_metrics[key]
        if not isinstance(pytorch_value, Real) or not isinstance(onnx_value, Real):
            continue
        differences[key] = abs(float(pytorch_value) - float(onnx_value))
    if not differences:
        raise ValueError("No shared scalar metrics were available for parity validation")
    maximum_difference = max(differences.values())
    return {
        "tolerance": tolerance,
        "passed": maximum_difference <= tolerance,
        "maximum_absolute_difference": maximum_difference,
        "absolute_differences": differences,
    }


def _validation_summary(metrics: Any) -> dict[str, Any]:
    results = {
        str(key): float(value) if isinstance(value, Real) else value
        for key, value in dict(metrics.results_dict).items()
    }
    speed = {
        str(key): float(value) if isinstance(value, Real) else value
        for key, value in dict(metrics.speed).items()
    }
    return {"metrics": results, "speed_ms": speed}


def train_segmenter(config: SegmenterTrainingConfig) -> Any:
    from ultralytics import YOLO

    data_yaml = project_path(config.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    validate_training_data(
        data_yaml,
        allow_unreviewed_experiment=config.allow_unreviewed_experiment,
    )
    model = YOLO(config.base_model)
    return model.train(
        data=str(data_yaml),
        imgsz=config.image_size,
        epochs=config.epochs,
        batch=config.batch_size,
        device=config.device,
        workers=config.workers,
        seed=config.seed,
        project=str(project_path("outputs/training")),
        name=config.run_name,
        exist_ok=True,
        plots=True,
        pretrained=True,
    )


def train_detector(config: DetectorTrainingConfig) -> Any:
    from ultralytics import YOLO

    data_yaml = project_path(config.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    validate_training_data(
        data_yaml,
        allow_unreviewed_experiment=config.allow_unreviewed_experiment,
    )
    model = YOLO(config.base_model, task=config.task)
    return model.train(
        data=str(data_yaml),
        imgsz=config.image_size,
        epochs=config.epochs,
        batch=config.batch_size,
        device=config.device,
        workers=config.workers,
        seed=config.seed,
        project=str(project_path("outputs/training")),
        name=config.run_name,
        exist_ok=True,
        plots=True,
        pretrained=True,
        patience=max(10, min(30, config.epochs // 3)),
        cache=False,
    )


def evaluate_and_export_detector(config: DetectorExportConfig) -> dict[str, Any]:
    """Evaluate a checkpoint on test data, export ONNX, and verify metric parity."""
    from ultralytics import YOLO

    data_yaml = project_path(config.data_yaml)
    checkpoint = project_path(config.checkpoint)
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    metadata = validate_training_data(
        data_yaml,
        allow_unreviewed_experiment=config.allow_unreviewed_experiment,
    )
    experimental = metadata.get("annotation_status") != "approved"
    evaluation_root = project_path("outputs/evaluation") / config.artifact_name
    evaluation_root.mkdir(parents=True, exist_ok=True)

    pytorch_model = YOLO(str(checkpoint), task=config.task)
    pytorch_validation = pytorch_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=config.image_size,
        device=config.device,
        project=str(evaluation_root),
        name="pytorch",
        exist_ok=True,
        plots=True,
    )
    exported = Path(
        pytorch_model.export(
            format="onnx",
            imgsz=config.image_size,
            dynamic=True,
            opset=17,
            simplify=False,
            half=False,
            device=config.device,
        )
    )
    if not exported.exists():
        raise FileNotFoundError(f"Ultralytics did not create the expected ONNX file: {exported}")

    onnx_model = YOLO(str(exported), task=config.task)
    onnx_validation = onnx_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=config.image_size,
        device=config.device,
        project=str(evaluation_root),
        name="onnx",
        exist_ok=True,
        plots=True,
    )
    pytorch_summary = _validation_summary(pytorch_validation)
    onnx_summary = _validation_summary(onnx_validation)
    parity = compare_metric_parity(
        pytorch_summary["metrics"],
        onnx_summary["metrics"],
        tolerance=config.parity_tolerance,
    )

    export_root = project_path("models/exported")
    if experimental:
        export_root /= "experimental"
    export_root.mkdir(parents=True, exist_ok=True)
    checkpoint_destination = export_root / f"{config.artifact_name}.pt"
    onnx_destination = export_root / f"{config.artifact_name}.onnx"
    shutil.copy2(checkpoint, checkpoint_destination)
    shutil.copy2(exported, onnx_destination)

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_name": config.artifact_name,
        "task": config.task,
        "experimental": experimental,
        "dataset_annotation_status": metadata.get("annotation_status"),
        "evaluation_split": "test",
        "image_size": config.image_size,
        "pytorch": pytorch_summary,
        "onnx": onnx_summary,
        "parity": parity,
        "artifacts": {
            "pytorch": str(checkpoint_destination.relative_to(project_path("."))),
            "onnx": str(onnx_destination.relative_to(project_path("."))),
        },
    }
    report_path = evaluation_root / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not parity["passed"]:
        raise RuntimeError(
            "ONNX metric parity failed: "
            f"maximum difference {parity['maximum_absolute_difference']:.6f} "
            f"exceeds {config.parity_tolerance:.6f}; see {report_path}"
        )
    return report
