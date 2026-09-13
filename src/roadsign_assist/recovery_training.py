"""Unattended, resumable V12 experimental-training orchestration.

This module deliberately has no Phase-E2, internal-test evaluation, canary, or
runtime-promotion operation.
"""

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast

import yaml

from roadsign_assist.datasets.recovery_v12 import (
    FREEZE,
    MANIFEST,
    RELEASE_ID,
    verify_freezes,
    verify_prepared_datasets,
)
from roadsign_assist.paths import project_path

DEFAULT_CONFIG = Path("configs/recovery/recovery_v12_training_matrix.yaml")
RUN_INDEX_ROOT = Path("outputs/training/recovery_v12")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_path(".").resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved = project_path(path).resolve()
    frozen_default = project_path(DEFAULT_CONFIG).resolve()
    if resolved != frozen_default:
        raise ValueError(
            f"Only the frozen V12 training matrix is allowed: {frozen_default}; got {resolved}"
        )
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Recovery training matrix must be a YAML object")
    config = cast(dict[str, Any], raw)
    required = {
        "release_id": RELEASE_ID,
        "manifest": MANIFEST.as_posix(),
        "freeze": FREEZE.as_posix(),
        "policy": "configs/recovery/available_data_experimental_training_20260906_v1.json",
        "detector_data": (
            "_archive/recovery_acquisition_v1/training_preparation/"
            f"{RELEASE_ID}/detector/data.yaml"
        ),
        "classifier_data": (
            "_archive/recovery_acquisition_v1/training_preparation/"
            f"{RELEASE_ID}/classifier"
        ),
        "evaluate_split": "validation",
        "open_internal_test": False,
        "phase_e2_allowed": False,
        "runtime_promotion_allowed": False,
        "deterministic": True,
    }
    invalid = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if invalid:
        raise ValueError(f"Unsafe or inconsistent recovery matrix: {invalid}")
    if not config.get("detectors") or not config.get("classifiers"):
        raise ValueError("Recovery matrix must declare detector and classifier candidates")
    candidates = [
        *cast(list[dict[str, Any]], config["detectors"]),
        *cast(list[dict[str, Any]], config["classifiers"]),
        cast(dict[str, Any], cast(Mapping[str, Any], config["smoke"])["detector"]),
        cast(dict[str, Any], cast(Mapping[str, Any], config["smoke"])["classifier"]),
    ]
    identifiers = [str(candidate.get("candidate_id", "")) for candidate in candidates]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("Recovery candidate IDs must be present and unique")
    if any(not isinstance(candidate.get("seed"), int) for candidate in candidates):
        raise ValueError("Every recovery candidate must declare an integer deterministic seed")
    return config


def _run_root(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_path(path)
    root = project_path(RUN_INDEX_ROOT).resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Run root must stay under {root}: {resolved}")
    return resolved


class _Tee:
    def __init__(self, first: TextIO, second: TextIO) -> None:
        self.first = first
        self.second = second

    def write(self, value: str) -> int:
        self.first.write(value)
        self.first.flush()
        self.second.write(value)
        self.second.flush()
        return len(value)

    def flush(self) -> None:
        self.first.flush()
        self.second.flush()


def _logging_handlers() -> list[logging.Handler]:
    """Return registered handlers once, including library-specific loggers."""

    handlers: list[logging.Handler] = []
    seen: set[int] = set()
    loggers = [logging.getLogger()]
    loggers.extend(
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    for logger in loggers:
        for handler in logger.handlers:
            if id(handler) not in seen:
                handlers.append(handler)
                seen.add(id(handler))
    return handlers


def _bind_logger_streams(
    standard_out: TextIO,
    standard_error: TextIO,
    tee_out: _Tee,
    tee_error: _Tee,
) -> list[tuple[logging.Handler, TextIO]]:
    bindings: list[tuple[logging.Handler, TextIO]] = []
    for handler in _logging_handlers():
        stream = getattr(handler, "stream", None)
        if stream is standard_out:
            bindings.append((handler, standard_out))
            handler.stream = tee_out  # type: ignore[attr-defined]
        elif stream is standard_error:
            bindings.append((handler, standard_error))
            handler.stream = tee_error  # type: ignore[attr-defined]
    return bindings


def _restore_logger_streams(
    bindings: Sequence[tuple[logging.Handler, TextIO]],
    standard_out: TextIO,
    standard_error: TextIO,
    tee_out: _Tee,
    tee_error: _Tee,
) -> None:
    for handler, stream in bindings:
        handler.stream = stream  # type: ignore[attr-defined]
    # A dependency may create a logger while stdout/stderr are redirected.
    # Rebind those new handlers before the per-stage log file closes.
    for handler in _logging_handlers():
        stream = getattr(handler, "stream", None)
        if stream is tee_out:
            handler.stream = standard_out  # type: ignore[attr-defined]
        elif stream is tee_error:
            handler.stream = standard_error  # type: ignore[attr-defined]


def _receipt_path(run_root: Path, stage: str) -> Path:
    return run_root / "receipts" / f"{stage}.json"


def _verify_outputs(outputs: Mapping[str, object]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for raw_path, raw_hash in outputs.items():
        path = project_path(str(raw_path))
        if not path.is_file():
            findings.append({"path": str(path), "status": "missing"})
            continue
        actual = _sha256(path)
        if actual != raw_hash:
            findings.append(
                {
                    "path": str(path),
                    "status": "hash_mismatch",
                    "expected": raw_hash,
                    "actual": actual,
                }
            )
    return findings


def _verified_receipt(run_root: Path, stage: str) -> dict[str, Any] | None:
    path = _receipt_path(run_root, stage)
    if not path.is_file():
        return None
    receipt = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if receipt.get("stage") != stage or receipt.get("status") != "completed":
        return None
    outputs = cast(Mapping[str, object], receipt.get("outputs", {}))
    findings = _verify_outputs(outputs)
    if findings:
        raise ValueError(f"Completed stage {stage} has invalid outputs: {findings[:3]}")
    return receipt


def _outputs(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        result[_relative(path)] = _sha256(path)
    return result


def _status(run_root: Path, **updates: object) -> dict[str, Any]:
    path = run_root / "status.json"
    current: dict[str, Any] = {}
    if path.is_file():
        current = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    current.update(updates)
    current["updated_at"] = _utc_now()
    _write_json(path, current)
    return current


def _run_stage(
    run_root: Path,
    stage: str,
    function: Callable[[], Sequence[Path]],
    *,
    always_run: bool = False,
) -> dict[str, Any]:
    prior = _verified_receipt(run_root, stage)
    if prior is not None and not always_run:
        print(f"[skip verified] {stage}", flush=True)
        return prior
    _status(run_root, state="running", current_stage=stage)
    log_path = run_root / "logs" / f"{stage}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"[stage] {stage}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        standard_out, standard_error = sys.stdout, sys.stderr
        tee_out = _Tee(standard_out, log_handle)
        tee_err = _Tee(standard_error, log_handle)
        bindings = _bind_logger_streams(standard_out, standard_error, tee_out, tee_err)
        try:
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                paths = list(function())
        finally:
            _restore_logger_streams(
                bindings, standard_out, standard_error, tee_out, tee_err
            )
    receipt = {
        "schema_version": "1.0",
        "stage": stage,
        "status": "completed",
        "completed_at": _utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "log": _relative(log_path),
        "outputs": _outputs(paths),
    }
    _write_json(_receipt_path(run_root, stage), receipt)
    return receipt


def _disk_free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def preflight(run_root: Path, config: Mapping[str, Any]) -> Path:
    import cv2
    import numpy
    import onnx
    import onnxruntime
    import sklearn
    import torch
    import torchvision
    import ultralytics
    from PIL import Image

    expected_python = project_path(".venv/Scripts/python.exe").resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise ValueError(
            f"V12 training must use {expected_python}; active Python is {actual_python}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU is required for the V12 unattended training matrix")
    free_gib = _disk_free_gib(project_path("."))
    minimum = float(config["minimum_free_gib"])
    if free_gib < minimum:
        raise RuntimeError(f"Preflight requires {minimum:.1f} GiB free; found {free_gib:.2f} GiB")
    freeze_result = verify_freezes(include_v12=True)
    if not freeze_result["passed"]:
        raise ValueError("V6-V12 freeze verification failed")
    dataset_result = verify_prepared_datasets()
    if not dataset_result["passed"]:
        raise ValueError(
            f"Prepared V12 dataset verification failed: {dataset_result['findings'][:3]}"
        )
    weights: dict[str, Any] = {}
    candidate_rows = list(cast(list[dict[str, Any]], config["detectors"]))
    candidate_rows.append(
        cast(dict[str, Any], cast(Mapping[str, Any], config["smoke"])["detector"])
    )
    for candidate in candidate_rows:
        model = project_path(str(candidate["model"]))
        if not model.is_file():
            raise FileNotFoundError(f"Pretrained detector weight is missing: {model}")
        actual = _sha256(model)
        if actual != candidate["model_sha256"]:
            raise ValueError(f"Pretrained detector weight changed: {model}")
        weights[_relative(model)] = actual
    manifest = project_path(MANIFEST)
    freeze = project_path(FREEZE)
    report = {
        "schema_version": "1.0",
        "checked_at": _utc_now(),
        "status": "passed",
        "release_id": RELEASE_ID,
        "python": platform.python_version(),
        "python_executable": str(actual_python),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "ultralytics": str(ultralytics.__version__),
        "numpy": str(numpy.__version__),
        "opencv": str(cv2.__version__),
        "onnx": str(onnx.__version__),
        "onnxruntime": str(onnxruntime.__version__),
        "pillow": str(Image.__version__),
        "scikit_learn": str(sklearn.__version__),
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": str(torch.version.cuda),
        "free_gib": free_gib,
        "minimum_free_gib": minimum,
        "manifest": _relative(manifest),
        "manifest_sha256": _sha256(manifest),
        "freeze": _relative(freeze),
        "freeze_sha256": _sha256(freeze),
        "matrix_config_sha256": _sha256(project_path(DEFAULT_CONFIG)),
        "pretrained_weights": weights,
        "freeze_verification": freeze_result,
        "prepared_dataset_verification": dataset_result,
        "internal_test_opened": False,
        "phase_e2_opened": False,
        "runtime_promotion_allowed": False,
    }
    output = run_root / "preflight.json"
    _write_json(output, report)
    return output


def _candidate_run_name(run_root: Path, kind: str, candidate_id: str) -> str:
    normalized = run_root.name.replace("-", "_").replace(".", "_")
    return f"rv12_{normalized}_{kind}_{candidate_id}"


def _detector_paths(run_root: Path, candidate_id: str) -> dict[str, Path]:
    root = run_root / "checkpoints" / "detectors" / candidate_id
    return {
        "root": root,
        "best": root / "weights" / "best.pt",
        "last": root / "weights" / "last.pt",
        "results": root / "results.csv",
        "completion": run_root / "manifests" / f"detector_{candidate_id}_completion.json",
    }


def train_detector(
    run_root: Path,
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    resume: bool,
) -> Sequence[Path]:
    from ultralytics import YOLO

    candidate_id = str(candidate["candidate_id"])
    paths = _detector_paths(run_root, candidate_id)
    data_yaml = project_path(str(config["detector_data"]))
    signature_payload = {
        "release_id": RELEASE_ID,
        "manifest_sha256": _sha256(project_path(MANIFEST)),
        "dataset_metadata_sha256": _sha256(data_yaml.parent / "dataset_metadata.json"),
        "candidate": dict(candidate),
        "model_sha256": _sha256(project_path(str(candidate["model"]))),
        "deterministic": True,
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    completion = paths["completion"]
    if completion.is_file():
        prior = cast(dict[str, Any], json.loads(completion.read_text(encoding="utf-8")))
        if prior.get("signature") != signature:
            raise ValueError(f"Completed detector signature changed: {candidate_id}")
        findings = _verify_outputs(cast(Mapping[str, object], prior["outputs"]))
        if findings:
            raise ValueError(
                f"Completed detector outputs changed: {candidate_id}: {findings[:3]}"
            )
        return [paths["best"], paths["last"], paths["results"], completion]
    if paths["root"].exists():
        if not resume:
            raise FileExistsError(f"Incomplete detector run exists; use resume: {paths['root']}")
        if not paths["last"].is_file():
            raise FileNotFoundError(f"Cannot resume detector without last.pt: {paths['last']}")
        print(f"[resume] detector {candidate_id} from {paths['last']}", flush=True)
        model = YOLO(str(paths["last"]), task="detect")
        model.train(resume=True)
    else:
        print(f"[start] detector {candidate_id}", flush=True)
        model = YOLO(str(project_path(str(candidate["model"]))), task="detect")
        arguments: dict[str, Any] = {
            "data": str(data_yaml),
            "imgsz": int(candidate["image_size"]),
            "epochs": int(candidate["epochs"]),
            "batch": int(candidate["batch"]),
            "device": str(config["device"]),
            "workers": int(config["workers"]),
            "seed": int(candidate["seed"]),
            "deterministic": True,
            "project": str(run_root / "checkpoints" / "detectors"),
            "name": candidate_id,
            "exist_ok": False,
            "plots": True,
            "pretrained": True,
            "patience": int(candidate["patience"]),
            "cache": False,
            "save": True,
            # Ultralytics rewrites last.pt after each epoch. Keeping every epoch
            # would need several extra GiB without improving crash recovery.
            "save_period": -1,
        }
        if "fraction" in candidate:
            arguments["fraction"] = float(candidate["fraction"])
        model.train(**arguments)
    for key in ("best", "last", "results"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"Detector did not produce {key}: {paths[key]}")
    _write_json(
        completion,
        {
            "schema_version": "1.0",
            "candidate_id": candidate_id,
            "status": "completed",
            "completed_at": _utc_now(),
            "signature": signature,
            "signature_payload": signature_payload,
            "outputs": _outputs([paths["best"], paths["last"], paths["results"]]),
        },
    )
    return [paths["best"], paths["last"], paths["results"], completion]


def _json_number(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def evaluate_detector(
    run_root: Path, config: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Sequence[Path]:
    from ultralytics import YOLO

    candidate_id = str(candidate["candidate_id"])
    paths = _detector_paths(run_root, candidate_id)
    checkpoint = paths["best"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    evaluation_root = run_root / "evaluation" / "detectors" / candidate_id
    raw_path = evaluation_root / "raw_predictions.jsonl"
    metrics_path = evaluation_root / "development_metrics.json"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    data_yaml = project_path(str(config["detector_data"]))
    model = YOLO(str(checkpoint), task="detect")
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=int(candidate["image_size"]),
        batch=int(candidate["batch"]),
        device=str(config["device"]),
        workers=int(config["workers"]),
        conf=0.001,
        iou=0.7,
        plots=True,
        save_json=True,
        project=str(evaluation_root),
        name="ultralytics_validation",
        exist_ok=True,
    )
    validation_images = sorted((data_yaml.parent / "images" / "validation").glob("*"))
    temporary = raw_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        predictions = model.predict(
            source=[str(path) for path in validation_images],
            stream=True,
            conf=0.001,
            iou=0.7,
            imgsz=int(candidate["image_size"]),
            device=str(config["device"]),
            verbose=False,
        )
        for image_path, result in zip(validation_images, predictions, strict=True):
            boxes = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.detach().cpu().tolist()
                confidence = result.boxes.conf.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                boxes = [
                    {
                        "xyxy": [float(item) for item in box],
                        "confidence": float(score),
                        "class_id": int(class_id),
                    }
                    for box, score, class_id in zip(xyxy, confidence, classes, strict=True)
                ]
            handle.write(
                json.dumps(
                    {
                        "image": _relative(image_path),
                        "image_sha256": image_path.stem,
                        "predictions": boxes,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(raw_path)
    report = {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "release_id": RELEASE_ID,
        "evaluation_split": "validation",
        "internal_test_evaluated": False,
        "phase_e2_evaluated": False,
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "configuration": dict(candidate),
        "metrics": {key: _json_number(value) for key, value in dict(metrics.results_dict).items()},
        "speed_ms": {key: _json_number(value) for key, value in dict(metrics.speed).items()},
        "raw_predictions": _relative(raw_path),
        "raw_predictions_sha256": _sha256(raw_path),
        "limitations": [
            "available-data development evidence only",
            "unknown physical-sign identity remains unmeasured",
            "not Phase E2 and not a real-camera accuracy or safety claim",
        ],
    }
    _write_json(metrics_path, report)
    return [raw_path, metrics_path]


def _create_smoke_classifier_data(run_root: Path, source_root: Path) -> Path:
    destination = run_root / "smoke_classifier_data"
    metadata_path = destination / "dataset_metadata.json"
    if metadata_path.is_file():
        return destination
    labels = cast(list[str], json.loads((source_root / "labels.json").read_text(encoding="utf-8")))
    for split, limit in (("train", 2), ("validation", 2)):
        for label in labels:
            candidates = sorted((source_root / split / label).glob("*.jpg"))[:limit]
            for source in candidates:
                target = destination / split / label / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    os.link(source, target)
    _write_json(destination / "labels.json", labels)
    _write_json(
        metadata_path,
        {
            "schema_version": "1.0",
            "dataset_id": RELEASE_ID + "_classifier_smoke",
            "annotation_status": "experimental_available_data_only",
            "coursework_images_included": 0,
            "internal_academic_only": True,
            "publication_prohibited": True,
            "runtime_promotion_allowed": False,
            "phase_e2_present": False,
        },
    )
    return destination


def _classifier_outputs(run_name: str) -> list[Path]:
    run = project_path("outputs/training") / run_name
    checkpoint = project_path("models/checkpoints") / run_name
    candidate = project_path("models/candidates") / run_name
    return [
        run / "metrics.json",
        run / "progress.json",
        checkpoint / "best.pt",
        checkpoint / "latest.pt",
        candidate / "sign_classifier.onnx",
        candidate / "sign_classifier.labels.json",
        candidate / "sign_classifier.calibration.json",
    ]


def train_classifier(
    run_root: Path,
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    resume: bool,
    smoke: bool,
) -> Sequence[Path]:
    from roadsign_assist.classification.folder_training import (
        FolderClassifierTrainingConfig,
        train_folder_classifier,
    )

    candidate_id = str(candidate["candidate_id"])
    run_name = _candidate_run_name(run_root, "classifier", candidate_id)
    outputs = _classifier_outputs(run_name)
    metrics_path = outputs[0]
    completion = run_root / "manifests" / f"classifier_{candidate_id}_completion.json"
    source_root = project_path(str(config["classifier_data"]))
    data_root = _create_smoke_classifier_data(run_root, source_root) if smoke else source_root
    common = cast(Mapping[str, Any], config.get("classifier_common", {}))
    training_config = FolderClassifierTrainingConfig(
        data_root=data_root,
        architecture=cast(Any, str(candidate["architecture"])),
        image_size=int(candidate["image_size"]),
        epochs=int(candidate["epochs"]),
        batch_size=int(candidate["batch"]),
        learning_rate=float(common.get("learning_rate", 0.0003)),
        weight_decay=float(common.get("weight_decay", 0.0001)),
        label_smoothing=float(common.get("label_smoothing", 0.05)),
        workers=int(config["workers"]),
        device="auto",
        seed=int(candidate["seed"]),
        run_name=run_name,
        allow_unreviewed_experiment=True,
        tune_confidence_threshold=bool(common.get("tune_confidence_threshold", True)),
        target_selective_accuracy=float(common.get("target_selective_accuracy", 0.98)),
        evaluate_test=False,
        overwrite=False,
        resume=resume and not metrics_path.is_file(),
    )
    signature_payload = {
        "release_id": RELEASE_ID,
        "manifest_sha256": _sha256(project_path(MANIFEST)),
        "data_root": _relative(data_root),
        "dataset_metadata_sha256": _sha256(data_root / "dataset_metadata.json"),
        "candidate": dict(candidate),
        "common": dict(common),
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    if metrics_path.is_file():
        metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
        if (
            metrics.get("evaluation_split") != "validation"
            or metrics.get("experimental") is not True
        ):
            raise ValueError(
                f"Classifier completion is not an experimental validation-only run: {run_name}"
            )
        parity = metrics.get("onnx_parity")
        if not isinstance(parity, dict) or parity.get("passed") is not True:
            raise ValueError(f"Classifier ONNX parity did not pass: {run_name}")
    else:
        train_folder_classifier(training_config)
    for path in outputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if completion.is_file():
        prior = cast(dict[str, Any], json.loads(completion.read_text(encoding="utf-8")))
        if prior.get("signature") != signature:
            raise ValueError(f"Completed classifier signature changed: {candidate_id}")
        findings = _verify_outputs(cast(Mapping[str, object], prior["outputs"]))
        if findings:
            raise ValueError(
                f"Completed classifier outputs changed: {candidate_id}: {findings[:3]}"
            )
    else:
        _write_json(
            completion,
            {
                "schema_version": "1.0",
                "candidate_id": candidate_id,
                "run_name": run_name,
                "status": "completed",
                "completed_at": _utc_now(),
                "signature": signature,
                "signature_payload": signature_payload,
                "outputs": _outputs(outputs),
            },
        )
    return [*outputs, completion]


def evaluate_classifier(
    run_root: Path,
    candidate: Mapping[str, Any],
) -> Sequence[Path]:
    import torch
    from torch.utils.data import DataLoader

    from roadsign_assist.classification.folder_training import CropFolderDataset
    from roadsign_assist.classification.preprocessing import build_evaluation_transform
    from roadsign_assist.classification.training import build_torchvision_model

    candidate_id = str(candidate["candidate_id"])
    run_name = _candidate_run_name(run_root, "classifier", candidate_id)
    metrics_source = project_path("outputs/training") / run_name / "metrics.json"
    metrics = cast(dict[str, Any], json.loads(metrics_source.read_text(encoding="utf-8")))
    if metrics.get("evaluation_split") != "validation":
        raise ValueError("Classifier raw evaluation may read validation only")
    configuration = cast(dict[str, Any], metrics["configuration"])
    data_root = project_path(str(configuration["data_root"]))
    labels = cast(list[str], json.loads((data_root / "labels.json").read_text(encoding="utf-8")))
    dataset = CropFolderDataset(
        data_root / "validation",
        labels,
        build_evaluation_transform(int(configuration["image_size"])),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torchvision_model(configuration["architecture"], len(labels)).to(device)
    checkpoint = project_path(str(cast(Mapping[str, Any], metrics["artifacts"])["checkpoint"]))
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    loader = DataLoader(
        dataset, batch_size=int(configuration["batch_size"]), shuffle=False, num_workers=0
    )
    evaluation_root = run_root / "evaluation" / "classifiers" / candidate_id
    raw_path = evaluation_root / "raw_predictions.jsonl"
    report_path = evaluation_root / "development_metrics.json"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    offset = 0
    temporary = raw_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device)).detach().cpu()
            probabilities = torch.softmax(logits / float(metrics["temperature"]), dim=1)
            for index in range(len(targets)):
                path, target_index = dataset.samples[offset + index]
                handle.write(
                    json.dumps(
                        {
                            "crop": _relative(path),
                            "target_index": int(target_index),
                            "target_label": labels[int(target_index)],
                            "logits": [float(value) for value in logits[index].tolist()],
                            "probabilities": [
                                float(value) for value in probabilities[index].tolist()
                            ],
                            "predicted_index": int(probabilities[index].argmax().item()),
                            "predicted_label": labels[int(probabilities[index].argmax().item())],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            offset += len(targets)
    temporary.replace(raw_path)
    report: dict[str, object] = {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "run_name": run_name,
        "release_id": RELEASE_ID,
        "evaluation_split": "validation",
        "internal_test_evaluated": False,
        "phase_e2_evaluated": False,
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "configuration": configuration,
        "validation": metrics["validation"],
        "raw_predictions": _relative(raw_path),
        "raw_predictions_sha256": _sha256(raw_path),
        "limitations": [
            "available-data development evidence only",
            "many supported classes are absent from the small real development slice",
            "not Phase E2 and not a safety or replacement claim",
        ],
    }
    _write_json(report_path, report)
    return [raw_path, report_path]


def summarize(run_root: Path, config: Mapping[str, Any], *, smoke: bool) -> Sequence[Path]:
    if smoke:
        detector_rows = [
            cast(Mapping[str, Any], cast(Mapping[str, Any], config["smoke"])["detector"])
        ]
        classifier_rows = [
            cast(Mapping[str, Any], cast(Mapping[str, Any], config["smoke"])["classifier"])
        ]
    else:
        detector_rows = cast(list[Mapping[str, Any]], config["detectors"])
        classifier_rows = cast(list[Mapping[str, Any]], config["classifiers"])
    detectors: list[dict[str, Any]] = []
    for candidate in detector_rows:
        path = (
            run_root
            / "evaluation"
            / "detectors"
            / str(candidate["candidate_id"])
            / "development_metrics.json"
        )
        report = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        metrics = cast(Mapping[str, Any], report["metrics"])
        detectors.append(
            {
                "candidate_id": candidate["candidate_id"],
                "map50_95": metrics.get("metrics/mAP50-95(B)"),
                "map50": metrics.get("metrics/mAP50(B)"),
                "precision": metrics.get("metrics/precision(B)"),
                "recall": metrics.get("metrics/recall(B)"),
                "report": _relative(path),
            }
        )
    classifiers: list[dict[str, Any]] = []
    for candidate in classifier_rows:
        path = (
            run_root
            / "evaluation"
            / "classifiers"
            / str(candidate["candidate_id"])
            / "development_metrics.json"
        )
        report = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        validation = cast(Mapping[str, Any], report["validation"])
        classifiers.append(
            {
                "candidate_id": candidate["candidate_id"],
                "macro_f1_all_labels": validation.get("macro_f1_all_labels"),
                "macro_f1_observed": validation.get("macro_f1_observed"),
                "accuracy": validation.get("accuracy"),
                "ece": validation.get("ece"),
                "observed_labels": validation.get("observed_labels"),
                "report": _relative(path),
            }
        )
    selected_detector = max(detectors, key=lambda row: float(row["map50_95"] or -1))
    selected_classifier = max(
        classifiers,
        key=lambda row: (
            float(row["macro_f1_all_labels"] or -1),
            float(row["accuracy"] or -1),
            -float(row["ece"] or 999),
        ),
    )
    remaining_gates = [
        "explicit selected-candidate internal-test opening",
        "separately locked Phase E2 target-device recordings",
        "real-camera accuracy and safety gates",
        "paired legacy comparison",
        "runtime performance and reliability gates",
        "shadow, canary, promotion, and rollback sign-off",
    ]
    report: dict[str, object] = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "run_id": run_root.name,
        "mode": "smoke" if smoke else "full",
        "release_id": RELEASE_ID,
        "evaluation_split": "validation",
        "internal_test_evaluated": False,
        "phase_e2_evaluated": False,
        "legacy_compared": False,
        "runtime_promotion_allowed": False,
        "detectors": detectors,
        "classifiers": classifiers,
        "development_selected_detector": selected_detector,
        "development_selected_classifier": selected_classifier,
        "result_scope": "available_data_experimental_development_only",
        "final_replacement_readiness": "blocked",
        "remaining_gates": remaining_gates,
    }
    json_path = run_root / "final" / "development_summary.json"
    markdown_path = run_root / "final" / "development_summary.md"
    _write_json(json_path, report)
    lines = [
        "# V12 experimental development summary",
        "",
        f"Run: `{run_root.name}`",
        "",
        f"Development-selected detector: `{selected_detector['candidate_id']}`",
        f"Development-selected classifier: `{selected_classifier['candidate_id']}`",
        "",
        "This is available-data experimental development evidence only. Internal test and Phase E2 were not opened. No safety, real-camera accuracy, legacy-replacement, canary, or runtime-promotion claim is authorized.",
        "",
        "## Remaining gates",
        "",
    ]
    lines.extend(f"- {gate}" for gate in remaining_gates)
    lines.append("")
    _write_text(markdown_path, "\n".join(lines))
    return [json_path, markdown_path]


def pipeline(run_root: Path, config: Mapping[str, Any], *, mode: str, resume: bool) -> None:
    smoke = mode == "smoke"
    run_root.mkdir(parents=True, exist_ok=resume)
    plan_path = run_root / "pipeline_plan.json"
    current_plan = {
        "schema_version": "1.0",
        "run_id": run_root.name,
        "mode": mode,
        "release_id": RELEASE_ID,
        "config": _relative(project_path(DEFAULT_CONFIG)),
        "config_sha256": _sha256(project_path(DEFAULT_CONFIG)),
        "manifest_sha256": _sha256(project_path(MANIFEST)),
        "freeze_sha256": _sha256(project_path(FREEZE)),
        "created_at": _utc_now(),
        "internal_test_allowed": False,
        "phase_e2_allowed": False,
        "runtime_promotion_allowed": False,
    }
    if plan_path.is_file():
        prior = cast(dict[str, Any], json.loads(plan_path.read_text(encoding="utf-8")))
        for key in (
            "run_id",
            "mode",
            "release_id",
            "config_sha256",
            "manifest_sha256",
            "freeze_sha256",
        ):
            if prior.get(key) != current_plan.get(key):
                raise ValueError(
                    f"Resume plan mismatch for {key}: {prior.get(key)!r} != {current_plan.get(key)!r}"
                )
    elif resume:
        raise FileNotFoundError(f"Cannot resume without pipeline plan: {plan_path}")
    else:
        _write_json(plan_path, current_plan)
    _status(
        run_root,
        schema_version="1.0",
        run_id=run_root.name,
        mode=mode,
        state="running",
        current_stage="preflight",
        internal_test_opened=False,
        phase_e2_opened=False,
        runtime_promotion_attempted=False,
    )
    try:
        # Re-run the complete freeze and dataset hash preflight after every
        # launch, including resume. A prior preflight receipt is verified first,
        # but is never sufficient proof that inputs are still unchanged now.
        _run_stage(
            run_root,
            "preflight",
            lambda: [preflight(run_root, config)],
            always_run=True,
        )
        if smoke:
            detector_rows = [
                cast(Mapping[str, Any], cast(Mapping[str, Any], config["smoke"])["detector"])
            ]
            classifier_rows = [
                cast(Mapping[str, Any], cast(Mapping[str, Any], config["smoke"])["classifier"])
            ]
        else:
            detector_rows = cast(list[Mapping[str, Any]], config["detectors"])
            classifier_rows = cast(list[Mapping[str, Any]], config["classifiers"])
        for candidate in detector_rows:
            candidate_id = str(candidate["candidate_id"])
            _run_stage(
                run_root,
                f"train_detector_{candidate_id}",
                lambda candidate=candidate: train_detector(
                    run_root, config, candidate, resume=resume
                ),
            )
            _run_stage(
                run_root,
                f"evaluate_detector_{candidate_id}",
                lambda candidate=candidate: evaluate_detector(run_root, config, candidate),
            )
        for candidate in classifier_rows:
            candidate_id = str(candidate["candidate_id"])
            _run_stage(
                run_root,
                f"train_classifier_{candidate_id}",
                lambda candidate=candidate: train_classifier(
                    run_root, config, candidate, resume=resume, smoke=smoke
                ),
            )
            _run_stage(
                run_root,
                f"evaluate_classifier_{candidate_id}",
                lambda candidate=candidate: evaluate_classifier(run_root, candidate),
            )
        _run_stage(run_root, "summarize", lambda: summarize(run_root, config, smoke=smoke))
        _status(run_root, state="completed", current_stage="completed", completed_at=_utc_now())
    except BaseException as error:
        failure = {
            "schema_version": "1.0",
            "failed_at": _utc_now(),
            "run_id": run_root.name,
            "mode": mode,
            "error_type": type(error).__name__,
            "error": str(error),
            "current_status": json.loads((run_root / "status.json").read_text(encoding="utf-8")),
            "resume_command": ".\\scripts\\run_recovery_training_v12.ps1 -Mode Resume",
        }
        _write_json(run_root / "failure.json", failure)
        _status(run_root, state="failed", failed_at=_utc_now(), error=str(error))
        raise


def summarize_existing(run_root: Path, config: Mapping[str, Any]) -> Sequence[Path]:
    plan = cast(
        dict[str, Any], json.loads((run_root / "pipeline_plan.json").read_text(encoding="utf-8"))
    )
    current_inputs = {
        "release_id": RELEASE_ID,
        "config_sha256": _sha256(project_path(DEFAULT_CONFIG)),
        "manifest_sha256": _sha256(project_path(MANIFEST)),
        "freeze_sha256": _sha256(project_path(FREEZE)),
    }
    mismatches = {
        key: {"recorded": plan.get(key), "current": value}
        for key, value in current_inputs.items()
        if plan.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Cannot summarize a run with changed frozen inputs: {mismatches}")
    freeze_result = verify_freezes(include_v12=True)
    if not freeze_result["passed"]:
        raise ValueError("Cannot summarize because V6-V12 freeze verification failed")

    smoke = plan.get("mode") == "smoke"
    if smoke:
        detector_rows = [
            cast(Mapping[str, Any], cast(Mapping[str, Any], config["smoke"])["detector"])
        ]
        classifier_rows = [
            cast(Mapping[str, Any], cast(Mapping[str, Any], config["smoke"])["classifier"])
        ]
    else:
        detector_rows = cast(list[Mapping[str, Any]], config["detectors"])
        classifier_rows = cast(list[Mapping[str, Any]], config["classifiers"])
    required_stages = [
        *(f"evaluate_detector_{row['candidate_id']}" for row in detector_rows),
        *(f"evaluate_classifier_{row['candidate_id']}" for row in classifier_rows),
    ]
    missing = [stage for stage in required_stages if _verified_receipt(run_root, stage) is None]
    if missing:
        raise ValueError(f"Cannot summarize before verified evaluations complete: {missing}")
    receipt = _run_stage(
        run_root,
        "summarize",
        lambda: summarize(run_root, config, smoke=smoke),
    )
    return [project_path(path) for path in cast(Mapping[str, object], receipt["outputs"])]


def main() -> int:
    parser = argparse.ArgumentParser(description="Recovery V12 unattended training driver")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--run-root", required=True)
    pipeline_parser = subparsers.add_parser("pipeline")
    pipeline_parser.add_argument("--run-root", required=True)
    pipeline_parser.add_argument("--mode", choices=("full", "smoke"), required=True)
    pipeline_parser.add_argument("--resume", action="store_true")
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    config = _load_config(Path(args.config))
    if args.command == "preflight":
        root = _run_root(args.run_root)
        root.mkdir(parents=True, exist_ok=False)
        output = preflight(root, config)
        print(json.dumps({"status": "passed", "report": _relative(output)}, indent=2))
        return 0
    if args.command == "pipeline":
        pipeline(_run_root(args.run_root), config, mode=args.mode, resume=args.resume)
        return 0
    if args.command == "summarize":
        outputs = summarize_existing(_run_root(args.run_root), config)
        print(json.dumps({"outputs": [_relative(path) for path in outputs]}, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
