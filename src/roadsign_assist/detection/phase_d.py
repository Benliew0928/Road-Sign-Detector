"""Guarded Phase-D detector training, validation selection, and locked testing."""

from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import csv
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml
from PIL import Image

from roadsign_assist.detection.training import validate_training_data
from roadsign_assist.evaluation.detector import greedy_box_match_pairs, percentile
from roadsign_assist.paths import project_path

PHASE_D_RELEASE_ID = "detector_production_assignment_v1_20260829"
PHASE_D_DATA_YAML = Path("data/processed/detector_production_assignment_v1_20260829/data.yaml")
PHASE_D_MANIFEST = Path("data/manifests/detector_production_assignment_v1_20260829.csv")
PHASE_D_SELECTION = Path("outputs/training/phase_d_selection.json")
PHASE_D_EVALUATION_ROOT = Path("outputs/evaluation/phase_d")
PHASE_D_CANDIDATE_ROOT = Path("models/candidates/phase_d")
PHASE_D_SEED = 2513
PHASE_D_EPOCHS = 100
PHASE_D_PATIENCE = 20
PHASE_D_WORKERS = 4
PHASE_D_MATCH_IOU = 0.50
PHASE_D_SMALL_AREA = 0.01
PHASE_D_VERY_SMALL_AREA = 0.001
PHASE_D_MANIFEST_SHA256 = "d39333958ff23a4fcc39af14a436b4c7ca58d0faaf89e8af255358e938304e65"
PHASE_D_DATA_YAML_SHA256 = "e9293f76559157c78bf8386ffd82e7a821c1f13abafbd221f1b9b4e1b836b2c8"
PHASE_D_METADATA_SHA256 = "3652224807ce780c9fee46f235ee3776258f3bd7374d3b93f86b68dc58d16020"
PHASE_D_LABEL_TREE_SHA256 = "04fac8447aa6371bfd548b931ca9782dd8f4e0c2aea4e07a3809c63c8312302f"
PHASE_D_WEIGHT_SHA256 = {
    "yolo26s.pt": "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
    "yolo26m.pt": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
}


@dataclass(frozen=True)
class PhaseDRunSpec:
    candidate_id: str
    model: str
    image_size: int
    batch_size: int
    gflops: float
    p2: bool = False


PHASE_D_RUN_SPECS = (
    PhaseDRunSpec("pd_v1_yolo26s_640_s2513", "yolo26s.pt", 640, 8, 22.8),
    PhaseDRunSpec("pd_v1_yolo26s_960_s2513", "yolo26s.pt", 960, 4, 51.4),
    PhaseDRunSpec("pd_v1_yolo26m_960_s2513", "yolo26m.pt", 960, 2, 169.6),
    PhaseDRunSpec("pd_v1_yolo26s_p2_960_s2513", "yolo26s-p2.pt", 960, 2, 62.5, True),
)

PHASE_D_EXCLUSIONS = {
    "pd_v1_yolo26s_p2_960_s2513": (
        "excluded_preflight: Ultralytics 8.4.75 provides the official YOLO26 P2 "
        "architecture but no official yolo26s-p2.pt pretrained asset; training from scratch "
        "is prohibited by the Phase-D plan"
    )
}

TRAINING_AUGMENTATION: dict[str, float | int | bool | str] = {
    "optimizer": "auto",
    "cos_lr": False,
    "close_mosaic": 10,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def phase_d_run_specs() -> tuple[PhaseDRunSpec, ...]:
    return PHASE_D_RUN_SPECS


def phase_d_active_run_specs() -> tuple[PhaseDRunSpec, ...]:
    return tuple(spec for spec in PHASE_D_RUN_SPECS if spec.candidate_id not in PHASE_D_EXCLUSIONS)


def phase_d_run_spec(candidate_id: str) -> PhaseDRunSpec:
    for spec in PHASE_D_RUN_SPECS:
        if spec.candidate_id == candidate_id:
            return spec
    allowed = ", ".join(spec.candidate_id for spec in PHASE_D_RUN_SPECS)
    raise ValueError(f"Unknown Phase-D candidate {candidate_id!r}; expected one of: {allowed}")


def _run_root(spec: PhaseDRunSpec) -> Path:
    return project_path("outputs/training") / spec.candidate_id


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(project_path(".").resolve())).replace("\\", "/")


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_path("."),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _implementation_hashes() -> dict[str, str]:
    paths = (
        Path("src/roadsign_assist/detection/phase_d.py"),
        Path("src/roadsign_assist/detection/training.py"),
        Path("src/roadsign_assist/evaluation/detector.py"),
        Path("src/roadsign_assist/cli.py"),
        Path("scripts/run_phase_d_detector.ps1"),
    )
    return {
        str(path).replace("\\", "/"): _sha256(project_path(path))
        for path in paths
        if project_path(path).is_file()
    }


def _environment_record() -> dict[str, Any]:
    import torch
    import ultralytics

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        ),
        "ultralytics": ultralytics.__version__,
    }


def validate_phase_d_release() -> dict[str, Any]:
    data_yaml = project_path(PHASE_D_DATA_YAML)
    manifest_path = project_path(PHASE_D_MANIFEST)
    metadata_path = data_yaml.parent / "dataset_metadata.json"
    expected_hashes = {
        manifest_path: PHASE_D_MANIFEST_SHA256,
        data_yaml: PHASE_D_DATA_YAML_SHA256,
        metadata_path: PHASE_D_METADATA_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Phase-D frozen artifact changed: {path} ({actual} != {expected})")

    metadata = validate_training_data(data_yaml, allow_unreviewed_experiment=False)
    required = {
        "dataset_id": PHASE_D_RELEASE_ID,
        "task": "detect",
        "annotation_status": "approved",
        "coursework_images_included": 0,
        "negative_test_images": 120,
        "publication_prohibited": True,
        "dvc_remote_push_allowed": False,
    }
    actual = {key: metadata.get(key) for key in required}
    if actual != required:
        raise ValueError(f"Phase-D metadata invariant failure: {actual}")

    counts = {"train": 0, "validation": 0, "test": 0}
    negative_counts = {"train": 0, "validation": 0, "test": 0}
    image_failures: list[str] = []
    missing_labels: list[str] = []
    label_tree_digest = hashlib.sha256()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = str(row["split"])
            counts[split] += 1
            if str(row["is_negative"]).casefold() == "true":
                negative_counts[split] += 1
            image_path = project_path(row["dataset_image_path"])
            label_path = project_path(row["dataset_label_path"])
            if not image_path.is_file() or _sha256(image_path) != row["sha256"]:
                image_failures.append(row["dataset_image_path"])
            if not label_path.is_file():
                missing_labels.append(row["dataset_label_path"])
            else:
                label_sha256 = _sha256(label_path)
                label_tree_digest.update(
                    (
                        row["dataset_label_path"].replace("\\", "/") + "\0" + label_sha256 + "\n"
                    ).encode("utf-8")
                )
    if counts != {"train": 6948, "validation": 1489, "test": 1489}:
        raise ValueError(f"Phase-D split counts changed: {counts}")
    if negative_counts != {"train": 0, "validation": 0, "test": 120}:
        raise ValueError(f"Phase-D negative split policy changed: {negative_counts}")
    if image_failures:
        raise ValueError(f"Phase-D image hash failures: {image_failures[:3]}")
    if missing_labels:
        raise ValueError(f"Phase-D labels are missing: {missing_labels[:3]}")
    label_tree_sha256 = label_tree_digest.hexdigest()
    if label_tree_sha256 != PHASE_D_LABEL_TREE_SHA256:
        raise ValueError(
            f"Phase-D label tree changed: {label_tree_sha256} != {PHASE_D_LABEL_TREE_SHA256}"
        )
    return {
        "dataset_id": PHASE_D_RELEASE_ID,
        "hashes": {str(path): expected for path, expected in expected_hashes.items()},
        "split_counts": counts,
        "negative_split_counts": negative_counts,
        "verified_image_hashes": sum(counts.values()),
        "label_tree_sha256": label_tree_sha256,
        "metadata": metadata,
    }


_ultralytics_read_only_patched = False


def _disable_ultralytics_image_repair() -> None:
    """Keep Ultralytics verification read-only for the byte-frozen release."""
    global _ultralytics_read_only_patched

    from ultralytics.data import utils as data_utils

    if _ultralytics_read_only_patched:
        return
    original_check_image = data_utils.check_image

    def check_image_read_only(im_file: str) -> tuple[str, tuple[int, int]]:
        path = Path(im_file)
        if path.suffix.casefold() in {".jpg", ".jpeg"}:
            with path.open("rb") as handle:
                handle.seek(-2, 2)
                missing_end_marker = handle.read() != b"\xff\xd9"
            if missing_end_marker:
                image = Image.open(path)
                image.verify()
                width, height = data_utils.exif_size(image)
                if width <= 9 or height <= 9:
                    raise AssertionError(f"image size {(height, width)} <10 pixels")
                return (
                    f"{path}: decodable JPEG has no EOI marker; accepted read-only without repair",
                    (height, width),
                )
        return original_check_image(im_file)

    data_utils.check_image = check_image_read_only
    _ultralytics_read_only_patched = True


def _resolve_pretrained_weight(spec: PhaseDRunSpec) -> Path:
    from ultralytics import YOLO

    local = project_path(spec.model)
    if not local.is_file():
        # Ultralytics resolves and downloads official assets into the current project directory.
        YOLO(spec.model, task="detect")
    if not local.is_file():
        raise FileNotFoundError(
            f"Official pretrained weights were not materialized for {spec.candidate_id}: {local}"
        )
    expected_sha256 = PHASE_D_WEIGHT_SHA256.get(spec.model)
    actual_sha256 = _sha256(local)
    if expected_sha256 is None or actual_sha256 != expected_sha256:
        raise ValueError(
            f"Pretrained weight hash mismatch for {spec.model}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return local


def _training_arguments(spec: PhaseDRunSpec, device: str) -> dict[str, Any]:
    return {
        "data": str(project_path(PHASE_D_DATA_YAML)),
        "task": "detect",
        "imgsz": spec.image_size,
        "epochs": PHASE_D_EPOCHS,
        "batch": spec.batch_size,
        "device": device,
        "workers": PHASE_D_WORKERS,
        "seed": PHASE_D_SEED,
        "deterministic": True,
        "project": str(project_path("outputs/training")),
        "name": spec.candidate_id,
        "exist_ok": True,
        "plots": True,
        "pretrained": True,
        "patience": PHASE_D_PATIENCE,
        "cache": False,
        "amp": True,
        "val": True,
        "split": "val",
        "single_cls": True,
        "rect": False,
        **TRAINING_AUGMENTATION,
    }


def _build_run_manifest(spec: PhaseDRunSpec, weight: Path, device: str) -> dict[str, Any]:
    git_status = _git_output("status", "--porcelain")
    git_diff = _git_output("diff", "--binary")
    dirty_fingerprint = hashlib.sha256((git_status + "\n" + git_diff).encode("utf-8")).hexdigest()
    signature_payload = {
        "candidate": asdict(spec),
        "dataset_hashes": {
            "manifest": PHASE_D_MANIFEST_SHA256,
            "data_yaml": PHASE_D_DATA_YAML_SHA256,
            "metadata": PHASE_D_METADATA_SHA256,
            "label_tree": PHASE_D_LABEL_TREE_SHA256,
        },
        "pretrained_weight_sha256": _sha256(weight),
        "training_arguments": _training_arguments(spec, device),
        "implementation_hashes": _implementation_hashes(),
        "environment": _environment_record(),
    }
    return {
        "schema_version": "1.0",
        "phase": "D",
        "dataset_id": PHASE_D_RELEASE_ID,
        "assignment_only": True,
        "redistribution_allowed": False,
        "shared_dvc_push_allowed": False,
        "created_at": _utc_now(),
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "git_dirty_fingerprint": dirty_fingerprint,
        **signature_payload,
        "run_signature": _json_sha256(signature_payload),
    }


def train_phase_d_detector(
    candidate_id: str,
    *,
    resume: bool = False,
    device: str = "0",
) -> dict[str, Any]:
    """Train or resume one fixed Phase-D candidate without silently reusing a run."""
    from ultralytics import YOLO

    validate_phase_d_release()
    _disable_ultralytics_image_repair()
    spec = phase_d_run_spec(candidate_id)
    if candidate_id in PHASE_D_EXCLUSIONS:
        raise ValueError(PHASE_D_EXCLUSIONS[candidate_id])
    weight = _resolve_pretrained_weight(spec)
    run_root = _run_root(spec)
    manifest_path = run_root / "run_manifest.json"
    complete_path = run_root / "training_complete.json"
    proposed = _build_run_manifest(spec, weight, device)

    if manifest_path.is_file():
        recorded = _read_json(manifest_path)
        if recorded.get("run_signature") != proposed.get("run_signature"):
            raise ValueError(
                f"Run signature mismatch for {candidate_id}; preserve the existing run and use "
                "a new candidate ID for changed code, data, weights, environment, or arguments."
            )
        if complete_path.is_file():
            return {**_read_json(complete_path), "status": "skipped_complete"}
        if not resume:
            raise FileExistsError(
                f"Incomplete run already exists: {run_root}. Rerun with --resume."
            )
        last_checkpoint = run_root / "weights" / "last.pt"
        if not last_checkpoint.is_file():
            raise FileNotFoundError(
                f"Cannot resume {candidate_id}: missing epoch checkpoint {last_checkpoint}"
            )
        print(f"[resume] {candidate_id} from {last_checkpoint}", flush=True)
        model = YOLO(str(last_checkpoint), task="detect")
        result = model.train(resume=True)
    else:
        if run_root.exists():
            raise FileExistsError(
                f"Unrecognized pre-existing run directory has no manifest: {run_root}"
            )
        run_root.mkdir(parents=True)
        _write_json(manifest_path, proposed)
        print(
            f"[start] {candidate_id}: {spec.model}, imgsz={spec.image_size}, "
            f"batch={spec.batch_size}, epochs={PHASE_D_EPOCHS}",
            flush=True,
        )
        model = YOLO(str(weight), task="detect")
        result = model.train(**_training_arguments(spec, device))

    best_checkpoint = run_root / "weights" / "best.pt"
    last_checkpoint = run_root / "weights" / "last.pt"
    if not best_checkpoint.is_file() or not last_checkpoint.is_file():
        raise FileNotFoundError(f"Training returned without complete checkpoints: {run_root}")
    payload = {
        "schema_version": "1.0",
        "status": "completed",
        "candidate_id": candidate_id,
        "completed_at": _utc_now(),
        "best_checkpoint": _relative(best_checkpoint),
        "best_checkpoint_sha256": _sha256(best_checkpoint),
        "last_checkpoint": _relative(last_checkpoint),
        "last_checkpoint_sha256": _sha256(last_checkpoint),
        "result_type": type(result).__name__,
    }
    _write_json(complete_path, payload)
    print(f"[complete] {candidate_id}: {best_checkpoint}", flush=True)
    return payload


def _last_training_row(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def phase_d_status() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in PHASE_D_RUN_SPECS:
        root = _run_root(spec)
        complete_path = root / "training_complete.json"
        validation_path = root / "validation_metrics.json"
        row: dict[str, Any] = {**asdict(spec), "status": "pending", "epoch": None}
        if spec.candidate_id in PHASE_D_EXCLUSIONS:
            row["status"] = "excluded"
            row["exclusion_reason"] = PHASE_D_EXCLUSIONS[spec.candidate_id]
            rows.append(row)
            continue
        last_row = _last_training_row(root / "results.csv")
        if last_row:
            raw_epoch = last_row.get("epoch")
            row["epoch"] = int(float(raw_epoch)) + 1 if raw_epoch is not None else None
            row["map50_95"] = last_row.get("metrics/mAP50-95(B)")
            row["map50"] = last_row.get("metrics/mAP50(B)")
        if complete_path.is_file():
            row["status"] = "completed"
            row.update({"training": _read_json(complete_path)})
        elif root.exists():
            row["status"] = "interrupted_or_running" if last_row else "incomplete_without_epoch"
        if validation_path.is_file():
            validation = _read_json(validation_path)
            row["validation_map50_95"] = validation.get("metrics", {}).get("map50_95")
            row["validation_small_recall"] = (
                validation.get("operational", {}).get("small", {}).get("recall")
            )
            row["selected_confidence"] = validation.get("selected_confidence")
        rows.append(row)
    return {
        "schema_version": "1.0",
        "release_id": PHASE_D_RELEASE_ID,
        "completed_runs": sum(row["status"] == "completed" for row in rows),
        "total_runs": len(phase_d_active_run_specs()),
        "runs": rows,
    }


def _dataset_images(split: Literal["validation", "test"]) -> list[Path]:
    payload: dict[str, Any] = yaml.safe_load(
        project_path(PHASE_D_DATA_YAML).read_text(encoding="utf-8")
    )
    key = "val" if split == "validation" else "test"
    root = Path(str(payload["path"])) / str(payload[key])
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.casefold() in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    )


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    index = parts.index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _read_ground_truth(
    image_path: Path,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    with Image.open(image_path) as image:
        width, height = image.size
    boxes: list[list[float]] = []
    areas: list[float] = []
    for line_number, line in enumerate(
        _label_path(image_path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        values = [float(value) for value in line.split()]
        coordinates = values[1:]
        if len(coordinates) == 4:
            center_x, center_y, box_width, box_height = coordinates
            boxes.append(
                [
                    (center_x - box_width / 2) * width,
                    (center_y - box_height / 2) * height,
                    (center_x + box_width / 2) * width,
                    (center_y + box_height / 2) * height,
                ]
            )
            areas.append(box_width * box_height)
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            x_values = np.asarray(coordinates[0::2], dtype=np.float64)
            y_values = np.asarray(coordinates[1::2], dtype=np.float64)
            boxes.append(
                [
                    float(x_values.min() * width),
                    float(y_values.min() * height),
                    float(x_values.max() * width),
                    float(y_values.max() * height),
                ]
            )
            areas.append(
                float((x_values.max() - x_values.min()) * (y_values.max() - y_values.min()))
            )
        else:
            raise ValueError(f"Malformed YOLO label {_label_path(image_path)}:{line_number}")
    box_array = np.asarray(boxes, dtype=np.float64).reshape((-1, 4))
    area_array = np.asarray(areas, dtype=np.float64)
    return box_array, area_array


def _manifest_by_image(split: Literal["validation", "test"]) -> dict[Path, dict[str, str]]:
    records: dict[Path, dict[str, str]] = {}
    with project_path(PHASE_D_MANIFEST).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            records[project_path(row["dataset_image_path"]).resolve()] = row
    return records


def _numpy(value: Any, *, columns: int | None = None) -> np.ndarray[Any, np.dtype[np.float64]]:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    return array.reshape((-1, columns)) if columns is not None else array.reshape((-1,))


def _collect_predictions(
    model: Any,
    *,
    split: Literal["validation", "test"],
    image_size: int,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    images = _dataset_images(split)
    manifest = _manifest_by_image(split)
    results = model.predict(
        source=[str(path) for path in images],
        stream=True,
        batch=max(1, batch_size),
        imgsz=image_size,
        conf=0.001,
        device=device,
        verbose=False,
        max_det=300,
    )
    records: list[dict[str, Any]] = []
    for index, (image_path, result) in enumerate(zip(images, results, strict=True), start=1):
        ground_truth, ground_truth_areas = _read_ground_truth(image_path)
        boxes = result.boxes
        predicted = (
            np.empty((0, 4), dtype=np.float64) if boxes is None else _numpy(boxes.xyxy, columns=4)
        )
        scores = np.empty((0,), dtype=np.float64) if boxes is None else _numpy(boxes.conf)
        row = manifest.get(image_path.resolve())
        if row is None:
            raise ValueError(f"Image absent from frozen Phase-D manifest: {image_path}")
        records.append(
            {
                "image": _relative(image_path),
                "ground_truth": ground_truth,
                "ground_truth_areas": ground_truth_areas,
                "predicted": predicted,
                "scores": scores,
                "layout_root_id": row["layout_root_id"],
                "source_kind": row["source_kind"],
                "source_id": row["source_id"],
                "is_negative": row["is_negative"].casefold() == "true",
            }
        )
        if index % 100 == 0 or index == len(images):
            print(f"[{split}] predictions {index}/{len(images)}", flush=True)
    return records


def _counts_to_metrics(counts: dict[str, int]) -> dict[str, Any]:
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def _empty_counts() -> dict[str, int]:
    return {"images": 0, "true_positive": 0, "false_positive": 0, "false_negative": 0}


def _add_record_counts(
    counts: dict[str, int],
    *,
    ground_truth_count: int,
    prediction_count: int,
    matched_count: int,
) -> None:
    counts["images"] += 1
    counts["true_positive"] += matched_count
    counts["false_positive"] += prediction_count - matched_count
    counts["false_negative"] += ground_truth_count - matched_count


def operational_metrics(
    records: list[dict[str, Any]],
    *,
    confidence: float,
    match_iou: float = PHASE_D_MATCH_IOU,
) -> dict[str, Any]:
    overall = _empty_counts()
    dense = _empty_counts()
    layouts: dict[str, dict[str, int]] = {}
    source_kinds: dict[str, dict[str, int]] = {}
    source_ids: dict[str, dict[str, int]] = {}
    small_ground_truth = small_matched = 0
    very_small_ground_truth = very_small_matched = 0
    negative_images = negative_images_with_predictions = negative_boxes = 0

    for record in records:
        ground_truth = cast(np.ndarray[Any, np.dtype[np.float64]], record["ground_truth"])
        areas = cast(np.ndarray[Any, np.dtype[np.float64]], record["ground_truth_areas"])
        predicted_all = cast(np.ndarray[Any, np.dtype[np.float64]], record["predicted"])
        scores = cast(np.ndarray[Any, np.dtype[np.float64]], record["scores"])
        predicted = predicted_all[scores >= confidence]
        pairs = greedy_box_match_pairs(ground_truth, predicted, match_iou)
        matched_ground_truth = {pair[0] for pair in pairs}
        matched_count = len(pairs)
        _add_record_counts(
            overall,
            ground_truth_count=len(ground_truth),
            prediction_count=len(predicted),
            matched_count=matched_count,
        )
        layout = str(record["layout_root_id"] or "unspecified")
        layout_counts = layouts.setdefault(layout, _empty_counts())
        _add_record_counts(
            layout_counts,
            ground_truth_count=len(ground_truth),
            prediction_count=len(predicted),
            matched_count=matched_count,
        )
        for source_groups, value in (
            (source_kinds, str(record.get("source_kind") or "unspecified")),
            (source_ids, str(record.get("source_id") or "unspecified")),
        ):
            source_counts = source_groups.setdefault(value, _empty_counts())
            _add_record_counts(
                source_counts,
                ground_truth_count=len(ground_truth),
                prediction_count=len(predicted),
                matched_count=matched_count,
            )
        if len(ground_truth) >= 3:
            _add_record_counts(
                dense,
                ground_truth_count=len(ground_truth),
                prediction_count=len(predicted),
                matched_count=matched_count,
            )
        for ground_truth_index, area in enumerate(areas):
            if float(area) <= PHASE_D_SMALL_AREA:
                small_ground_truth += 1
                small_matched += ground_truth_index in matched_ground_truth
            if float(area) <= PHASE_D_VERY_SMALL_AREA:
                very_small_ground_truth += 1
                very_small_matched += ground_truth_index in matched_ground_truth
        if bool(record["is_negative"]):
            negative_images += 1
            negative_boxes += len(predicted)
            negative_images_with_predictions += len(predicted) > 0

    return {
        "confidence": confidence,
        "match_iou": match_iou,
        "overall": _counts_to_metrics(overall),
        "small": {
            "definition": f"normalized ground-truth box area <= {PHASE_D_SMALL_AREA}",
            "ground_truth": small_ground_truth,
            "matched": small_matched,
            "recall": small_matched / small_ground_truth if small_ground_truth else None,
        },
        "very_small_proxy": {
            "definition": f"normalized ground-truth box area <= {PHASE_D_VERY_SMALL_AREA}",
            "ground_truth": very_small_ground_truth,
            "matched": very_small_matched,
            "recall": (
                very_small_matched / very_small_ground_truth if very_small_ground_truth else None
            ),
            "caveat": "Box area is a size proxy; the release has no reviewed distance field.",
        },
        "dense": {
            "definition": "image contains at least 3 ground-truth boxes",
            **_counts_to_metrics(dense),
        },
        "layout_root_id": {
            name: _counts_to_metrics(counts) for name, counts in sorted(layouts.items())
        },
        "source_kind": {
            name: _counts_to_metrics(counts) for name, counts in sorted(source_kinds.items())
        },
        "source_id": {
            name: _counts_to_metrics(counts) for name, counts in sorted(source_ids.items())
        },
        "no_sign": {
            "images": negative_images,
            "images_with_false_positive": negative_images_with_predictions,
            "image_false_positive_rate": (
                negative_images_with_predictions / negative_images if negative_images else None
            ),
            "false_boxes": negative_boxes,
            "false_boxes_per_100_images": (
                negative_boxes * 100 / negative_images if negative_images else None
            ),
        },
        "environment_slices": {
            "night": {"status": "unavailable", "reason": "no reviewed night label"},
            "rain": {"status": "unavailable", "reason": "no reviewed rain label"},
        },
    }


def _select_confidence(records: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for threshold in np.arange(0.01, 1.0, 0.01):
        confidence = round(float(threshold), 2)
        metrics = operational_metrics(records, confidence=confidence)
        overall = metrics["overall"]
        rows.append(
            {
                "confidence": confidence,
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "small_recall": metrics["small"]["recall"],
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            float(row["f1"] or 0.0),
            float(row["recall"] or 0.0),
            -float(row["confidence"]),
        ),
    )
    return float(selected["confidence"]), rows


def _ultralytics_metric_summary(metrics: Any) -> dict[str, Any]:
    values = {str(key): float(value) for key, value in dict(metrics.results_dict).items()}
    precision = values.get("metrics/precision(B)")
    recall = values.get("metrics/recall(B)")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    speed = {str(key): float(value) for key, value in dict(metrics.speed).items()}
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": values.get("metrics/mAP50(B)"),
        "map50_95": values.get("metrics/mAP50-95(B)"),
        "speed_ms": speed,
        "raw": values,
    }


def _evaluate_validation_candidate(spec: PhaseDRunSpec, device: str) -> dict[str, Any]:
    from ultralytics import YOLO

    _disable_ultralytics_image_repair()
    root = _run_root(spec)
    complete_path = root / "training_complete.json"
    output_path = root / "validation_metrics.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"Phase-D training is incomplete: {spec.candidate_id}")
    complete = _read_json(complete_path)
    checkpoint = project_path(complete["best_checkpoint"])
    if _sha256(checkpoint) != complete["best_checkpoint_sha256"]:
        raise ValueError(f"Best checkpoint changed after training: {checkpoint}")
    if output_path.is_file():
        report = _read_json(output_path)
        if report.get("checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError(f"Stale validation report for {spec.candidate_id}")
        if report.get("evaluation_split") != "validation" or report.get("test_evaluated"):
            raise ValueError(f"Unsafe validation report for {spec.candidate_id}")
        return report

    print(f"[validation] evaluating {spec.candidate_id}", flush=True)
    model = YOLO(str(checkpoint), task="detect")
    validation = model.val(
        data=str(project_path(PHASE_D_DATA_YAML)),
        split="val",
        imgsz=spec.image_size,
        conf=0.001,
        iou=0.7,
        device=device,
        project=str(project_path(PHASE_D_EVALUATION_ROOT) / "validation"),
        name=spec.candidate_id,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    predictions = _collect_predictions(
        model,
        split="validation",
        image_size=spec.image_size,
        device=device,
        batch_size=spec.batch_size,
    )
    selected_confidence, threshold_rows = _select_confidence(predictions)
    report = {
        "schema_version": "1.0",
        "candidate_id": spec.candidate_id,
        "dataset_id": PHASE_D_RELEASE_ID,
        "evaluation_split": "validation",
        "test_evaluated": False,
        "task": "detect",
        "image_size": spec.image_size,
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics": _ultralytics_metric_summary(validation),
        "selected_confidence": selected_confidence,
        "threshold_selection_rule": "maximum validation box F1, then recall, then lower confidence",
        "threshold_search": threshold_rows,
        "threshold_caveat": "The frozen validation split contains no no-sign images.",
        "nms_policy": "not_applicable_yolo26_end_to_end",
        "operational": operational_metrics(predictions, confidence=selected_confidence),
        "evaluated_at": _utc_now(),
    }
    _write_json(output_path, report)
    return report


def _rank_phase_d_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        raise ValueError("Phase-D selection has no validation candidates")
    best_map = max(float(row["validation_map50_95"]) for row in candidates)
    map_eligible = [
        row for row in candidates if float(row["validation_map50_95"]) >= best_map - 0.005
    ]
    best_small = max(float(row["validation_small_recall"]) for row in map_eligible)
    finalists = [
        row for row in map_eligible if float(row["validation_small_recall"]) >= best_small - 0.01
    ]
    ranked_finalists = sorted(
        finalists,
        key=lambda row: (
            -float(row["validation_map50_95"]),
            -float(row["validation_recall"] or 0.0),
            float(row["gpu_inference_ms"] or float("inf")),
            float(row["gflops"]),
            str(row["candidate_id"]),
        ),
    )
    return map_eligible, ranked_finalists, ranked_finalists[0]


def select_phase_d_detector(
    output_path: str | Path = PHASE_D_SELECTION,
    *,
    device: str = "0",
) -> dict[str, Any]:
    """Evaluate and select candidates using validation only; never opens test images."""
    validate_phase_d_release()
    candidates: list[dict[str, Any]] = []
    for spec in phase_d_active_run_specs():
        report = _evaluate_validation_candidate(spec, device)
        metrics = report["metrics"]
        operational = report["operational"]
        map50_95 = metrics.get("map50_95")
        small_recall = operational.get("small", {}).get("recall")
        if map50_95 is None or small_recall is None:
            raise ValueError(f"Incomplete validation metrics for {spec.candidate_id}")
        validation_path = _run_root(spec) / "validation_metrics.json"
        if not validation_path.is_file():
            raise FileNotFoundError(validation_path)
        candidates.append(
            {
                **asdict(spec),
                "validation_report": _relative(validation_path),
                "validation_report_sha256": _sha256(validation_path),
                "checkpoint": report["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "selected_confidence": report["selected_confidence"],
                "validation_map50_95": map50_95,
                "validation_map50": metrics.get("map50"),
                "validation_precision": metrics.get("precision"),
                "validation_recall": metrics.get("recall"),
                "validation_small_recall": small_recall,
                "validation_very_small_recall": operational.get("very_small_proxy", {}).get(
                    "recall"
                ),
                "gpu_inference_ms": metrics.get("speed_ms", {}).get("inference"),
            }
        )

    map_eligible, ranked_finalists, selected = _rank_phase_d_candidates(candidates)
    report = {
        "schema_version": "1.0",
        "phase": "D",
        "dataset_id": PHASE_D_RELEASE_ID,
        "selection_split": "validation",
        "test_evaluated": False,
        "selection_rule": {
            "map50_95_window": 0.005,
            "small_recall_window": 0.01,
            "final_order": [
                "higher_map50_95",
                "higher_overall_recall",
                "lower_gpu_latency",
                "lower_gflops",
                "candidate_id",
            ],
        },
        "candidates": candidates,
        "map_eligible_candidates": [row["candidate_id"] for row in map_eligible],
        "finalists": [row["candidate_id"] for row in ranked_finalists],
        "selected": selected,
        "locked_test_allowed": True,
        "locked_configuration": {
            "candidate_id": selected["candidate_id"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "image_size": selected["image_size"],
            "confidence": selected["selected_confidence"],
            "match_iou": PHASE_D_MATCH_IOU,
            "nms_policy": "not_applicable_yolo26_end_to_end",
        },
        "created_at": _utc_now(),
    }
    output = project_path(output_path)
    if output.is_file():
        recorded = _read_json(output)
        comparable_recorded = {key: value for key, value in recorded.items() if key != "created_at"}
        comparable_proposed = {key: value for key, value in report.items() if key != "created_at"}
        if comparable_recorded != comparable_proposed:
            raise ValueError(
                "An immutable Phase-D selection already exists with different validation evidence"
            )
        return recorded
    _write_json(output, report)
    return report


def _verify_phase_d_selection(selection: dict[str, Any]) -> PhaseDRunSpec:
    if (
        selection.get("dataset_id") != PHASE_D_RELEASE_ID
        or selection.get("selection_split") != "validation"
        or selection.get("test_evaluated") is not False
        or selection.get("locked_test_allowed") is not True
    ):
        raise ValueError("Phase-D selection report does not authorize the locked test")
    raw_candidates = selection.get("candidates")
    expected_specs = {spec.candidate_id: spec for spec in phase_d_active_run_specs()}
    if not isinstance(raw_candidates, list):
        raise ValueError("Phase-D selection does not contain the complete active matrix")
    candidates = cast(list[dict[str, Any]], raw_candidates)
    if {str(row.get("candidate_id")) for row in candidates} != set(expected_specs):
        raise ValueError("Phase-D selection does not contain the complete active matrix")

    for row in candidates:
        candidate_id = str(row["candidate_id"])
        spec = expected_specs[candidate_id]
        for key, expected in asdict(spec).items():
            if row.get(key) != expected:
                raise ValueError(f"Selection candidate specification changed: {candidate_id}/{key}")
        validation_path = project_path(row["validation_report"])
        if not validation_path.is_file() or _sha256(validation_path) != row.get(
            "validation_report_sha256"
        ):
            raise ValueError(f"Validation evidence changed for {candidate_id}")
        validation = _read_json(validation_path)
        if (
            validation.get("candidate_id") != candidate_id
            or validation.get("dataset_id") != PHASE_D_RELEASE_ID
            or validation.get("evaluation_split") != "validation"
            or validation.get("test_evaluated") is not False
            or validation.get("checkpoint") != row.get("checkpoint")
            or validation.get("checkpoint_sha256") != row.get("checkpoint_sha256")
            or validation.get("selected_confidence") != row.get("selected_confidence")
        ):
            raise ValueError(f"Invalid validation evidence for {candidate_id}")

    map_eligible, ranked_finalists, selected = _rank_phase_d_candidates(candidates)
    if selection.get("selected") != selected:
        raise ValueError("Selected candidate does not follow the locked validation ranking")
    if selection.get("map_eligible_candidates") != [row["candidate_id"] for row in map_eligible]:
        raise ValueError("Phase-D mAP eligibility record changed")
    if selection.get("finalists") != [row["candidate_id"] for row in ranked_finalists]:
        raise ValueError("Phase-D finalist record changed")
    locked = selection.get("locked_configuration", {})
    expected_locked = {
        "candidate_id": selected["candidate_id"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "image_size": selected["image_size"],
        "confidence": selected["selected_confidence"],
        "match_iou": PHASE_D_MATCH_IOU,
        "nms_policy": "not_applicable_yolo26_end_to_end",
    }
    if locked != expected_locked:
        raise ValueError("Phase-D locked configuration is inconsistent with validation selection")
    return expected_specs[str(selected["candidate_id"])]


def _parity_report(pytorch: dict[str, Any], onnx: dict[str, Any]) -> dict[str, Any]:
    tolerances = {"map50": 0.005, "map50_95": 0.005, "precision": 0.01, "recall": 0.01}
    differences: dict[str, float] = {}
    failures: list[str] = []
    for key, tolerance in tolerances.items():
        first = pytorch.get(key)
        second = onnx.get(key)
        if first is None or second is None:
            failures.append(f"missing_{key}")
            continue
        difference = abs(float(first) - float(second))
        differences[key] = difference
        if difference > tolerance:
            failures.append(key)
    return {
        "passed": not failures,
        "tolerances": tolerances,
        "absolute_differences": differences,
        "failures": failures,
    }


def _benchmark_model(
    model: Any,
    *,
    images: list[Path],
    image_size: int,
    confidence: float,
    device: str,
    profile_name: str,
) -> dict[str, Any]:
    import torch

    if not images:
        raise ValueError("Benchmark image list is empty")
    is_cuda = str(device).casefold() not in {"cpu", "-1"} and torch.cuda.is_available()
    for index in range(10):
        model.predict(
            source=str(images[index % len(images)]),
            imgsz=image_size,
            conf=confidence,
            device=device,
            verbose=False,
        )
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    wall_times: list[float] = []
    inference_times: list[float] = []
    started_all = time.perf_counter()
    for index, image_path in enumerate(images, start=1):
        if is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            imgsz=image_size,
            conf=confidence,
            device=device,
            verbose=False,
        )[0]
        if is_cuda:
            torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - started) * 1000)
        inference_times.append(float(result.speed.get("inference", 0.0)))
        if index % 25 == 0 or index == len(images):
            print(f"[benchmark:{profile_name}] {index}/{len(images)}", flush=True)
    elapsed = time.perf_counter() - started_all
    return {
        "profile": profile_name,
        "device": device,
        "images": len(images),
        "warmup_images": 10,
        "wall_latency_ms": {
            "mean": statistics.fmean(wall_times),
            "median": statistics.median(wall_times),
            "p95": percentile(wall_times, 95),
            "maximum": max(wall_times),
        },
        "model_inference_ms": {
            "mean": statistics.fmean(inference_times),
            "p95": percentile(inference_times, 95),
        },
        "throughput_images_per_second": len(images) / elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if is_cuda else None,
    }


def _benchmark_images() -> list[Path]:
    manifest = _manifest_by_image("test")
    negatives = sorted(
        path for path, row in manifest.items() if row["is_negative"].casefold() == "true"
    )
    positives = sorted(
        path for path, row in manifest.items() if row["is_negative"].casefold() != "true"
    )
    if len(negatives) != 120 or len(positives) < 80:
        raise ValueError("Frozen deterministic benchmark sample cannot be assembled")
    return [*negatives, *positives[:80]]


def evaluate_phase_d_selected_detector(*, device: str = "0") -> dict[str, Any]:
    """Open the locked test only for the validation-selected immutable checkpoint."""
    from ultralytics import YOLO

    validate_phase_d_release()
    _disable_ultralytics_image_repair()
    selection_path = project_path(PHASE_D_SELECTION)
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Phase-D validation selection is required before the locked test: {selection_path}"
        )
    selection = _read_json(selection_path)
    spec = _verify_phase_d_selection(selection)
    locked = selection.get("locked_configuration", {})
    checkpoint = project_path(selection["selected"]["checkpoint"])
    if _sha256(checkpoint) != locked.get("checkpoint_sha256"):
        raise ValueError("Selected checkpoint hash changed after validation selection")
    if int(locked.get("image_size")) != spec.image_size:
        raise ValueError("Selected image size does not match the predeclared candidate")
    confidence = float(locked["confidence"])
    selection_sha256 = _sha256(selection_path)
    evaluation_root = project_path(PHASE_D_EVALUATION_ROOT) / spec.candidate_id / "locked_test"
    protocol_path = evaluation_root / "locked_test_protocol.json"
    report_path = evaluation_root / "locked_test_report.json"
    if report_path.is_file():
        report = _read_json(report_path)
        if report.get("selection_sha256") != selection_sha256:
            raise ValueError("A locked-test report already exists for a different selection")
        if report.get("gate_d_passed") is not True:
            raise RuntimeError(f"Phase-D locked test did not pass; see {report_path}")
        return report
    protocol = {
        "schema_version": "1.0",
        "started_at": _utc_now(),
        "selection": _relative(selection_path),
        "selection_sha256": selection_sha256,
        "candidate_id": spec.candidate_id,
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "image_size": spec.image_size,
        "confidence": confidence,
        "match_iou": PHASE_D_MATCH_IOU,
        "test_retuning_allowed": False,
    }
    if protocol_path.is_file() and _read_json(protocol_path) != protocol:
        recorded = _read_json(protocol_path)
        comparable = {key: value for key, value in recorded.items() if key != "started_at"}
        expected = {key: value for key, value in protocol.items() if key != "started_at"}
        if comparable != expected:
            raise ValueError("Locked-test protocol mismatch; refusing to reopen test")
    else:
        _write_json(protocol_path, protocol)

    print(f"[locked test] PyTorch evaluation: {spec.candidate_id}", flush=True)
    pytorch_model = YOLO(str(checkpoint), task="detect")
    pytorch_result = pytorch_model.val(
        data=str(project_path(PHASE_D_DATA_YAML)),
        split="test",
        imgsz=spec.image_size,
        conf=0.001,
        iou=0.7,
        device=device,
        project=str(evaluation_root),
        name="pytorch",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    predictions = _collect_predictions(
        pytorch_model,
        split="test",
        image_size=spec.image_size,
        device=device,
        batch_size=spec.batch_size,
    )
    operational = operational_metrics(predictions, confidence=confidence)
    exported = Path(
        pytorch_model.export(
            format="onnx",
            imgsz=spec.image_size,
            batch=1,
            dynamic=True,
            opset=17,
            simplify=False,
            half=False,
            device=device,
        )
    )
    if not exported.is_file():
        raise FileNotFoundError(f"Ultralytics ONNX export is missing: {exported}")
    print("[locked test] ONNX parity evaluation", flush=True)
    onnx_model = YOLO(str(exported), task="detect")
    onnx_result = onnx_model.val(
        data=str(project_path(PHASE_D_DATA_YAML)),
        split="test",
        imgsz=spec.image_size,
        conf=0.001,
        iou=0.7,
        device=device,
        project=str(evaluation_root),
        name="onnx",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    pytorch_metrics = _ultralytics_metric_summary(pytorch_result)
    onnx_metrics = _ultralytics_metric_summary(onnx_result)
    parity = _parity_report(pytorch_metrics, onnx_metrics)
    benchmark_images = _benchmark_images()
    # Ultralytics caches its predictor (including the ONNX Runtime execution
    # provider) after the first inference.  Use a fresh model for the CPU
    # profile so the earlier CUDA validation cannot leak into that benchmark.
    onnx_cpu_model = YOLO(str(exported), task="detect")
    benchmarks = {
        "pytorch_gpu": _benchmark_model(
            pytorch_model,
            images=benchmark_images,
            image_size=spec.image_size,
            confidence=confidence,
            device=device,
            profile_name="pytorch_gpu",
        ),
        "onnx_gpu": _benchmark_model(
            onnx_model,
            images=benchmark_images,
            image_size=spec.image_size,
            confidence=confidence,
            device=device,
            profile_name="onnx_gpu",
        ),
        "onnx_cpu": _benchmark_model(
            onnx_cpu_model,
            images=benchmark_images,
            image_size=spec.image_size,
            confidence=confidence,
            device="cpu",
            profile_name="onnx_cpu",
        ),
    }
    report = {
        "schema_version": "1.0",
        "phase": "D",
        "gate_d_passed": parity["passed"],
        "dataset_id": PHASE_D_RELEASE_ID,
        "evaluation_split": "test",
        "test_evaluated": True,
        "test_retuning_allowed": False,
        "selection_sha256": selection_sha256,
        "candidate_id": spec.candidate_id,
        "checkpoint_sha256": _sha256(checkpoint),
        "image_size": spec.image_size,
        "confidence": confidence,
        "nms_policy": "not_applicable_yolo26_end_to_end",
        "pytorch": pytorch_metrics,
        "onnx": onnx_metrics,
        "parity": parity,
        "operational": operational,
        "benchmarks": benchmarks,
        "assignment_only": True,
        "redistribution_allowed": False,
        "runtime_promoted": False,
        "completed_at": _utc_now(),
    }
    _write_json(report_path, report)
    if not parity["passed"]:
        raise RuntimeError(f"Phase-D ONNX parity failed; see {report_path}")

    bundle = project_path(PHASE_D_CANDIDATE_ROOT) / spec.candidate_id
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, bundle / "model.pt")
    shutil.copy2(exported, bundle / "model.onnx")
    shutil.copy2(selection_path, bundle / "selection.json")
    shutil.copy2(report_path, bundle / "locked_test_report.json")
    _write_json(
        bundle / "candidate_manifest.json",
        {
            "schema_version": "1.0",
            "candidate_id": spec.candidate_id,
            "dataset_id": PHASE_D_RELEASE_ID,
            "assignment_only": True,
            "redistribution_allowed": False,
            "runtime_promoted": False,
            "gate_d_passed": True,
            "artifacts": {
                "pytorch": {"path": "model.pt", "sha256": _sha256(bundle / "model.pt")},
                "onnx": {"path": "model.onnx", "sha256": _sha256(bundle / "model.onnx")},
            },
        },
    )
    return report
