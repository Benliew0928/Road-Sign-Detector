import json
from pathlib import Path

import numpy as np
import pytest

from roadsign_assist.classification.folder_training import (
    FolderClassifierRunPaths,
    _build_training_datasets,
    compare_classifier_outputs,
    prepare_classifier_run_paths,
    select_confidence_threshold,
    should_export_classifier_experimentally,
    validate_classifier_dataset,
)


def test_validation_only_training_does_not_enumerate_test_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_roots: list[Path] = []

    class RecordingDataset:
        def __init__(self, root: Path, labels: list[str], transform: object) -> None:
            del labels, transform
            observed_roots.append(root)

    monkeypatch.setattr(
        "roadsign_assist.classification.folder_training.CropFolderDataset",
        RecordingDataset,
    )

    _, _, test_dataset = _build_training_datasets(
        tmp_path,
        ["stop", "yield"],
        object(),
        object(),
        include_test=False,
    )

    assert test_dataset is None
    assert observed_roots == [tmp_path / "train", tmp_path / "validation"]


def test_classifier_training_requires_review_or_experimental_override(
    tmp_path: Path,
) -> None:
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "annotation_status": "source_boxes_unreviewed",
                "coursework_images_included": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="experimental override"):
        validate_classifier_dataset(tmp_path, allow_unreviewed_experiment=False)
    metadata = validate_classifier_dataset(
        tmp_path,
        allow_unreviewed_experiment=True,
    )
    assert metadata["annotation_status"] == "source_boxes_unreviewed"


def test_experimental_override_protects_approved_classifier_export() -> None:
    approved_metadata = {"annotation_status": "approved"}
    unreviewed_metadata = {"annotation_status": "source_boxes_unreviewed"}

    assert (
        should_export_classifier_experimentally(
            approved_metadata,
            allow_unreviewed_experiment=False,
        )
        is False
    )
    assert (
        should_export_classifier_experimentally(
            approved_metadata,
            allow_unreviewed_experiment=True,
        )
        is True
    )
    assert (
        should_export_classifier_experimentally(
            unreviewed_metadata,
            allow_unreviewed_experiment=False,
        )
        is True
    )


def test_classifier_output_parity() -> None:
    pytorch_logits = np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    onnx_logits = pytorch_logits + np.float32(1e-5)
    parity = compare_classifier_outputs(pytorch_logits, onnx_logits)
    assert parity["passed"] is True
    assert parity["top1_agreement"] == 1.0
    assert parity["acceptance_agreement"] == 1.0


def test_classifier_output_parity_allows_one_borderline_disagreement() -> None:
    pytorch_logits = np.tile(np.asarray([[4.0, 0.1]], dtype=np.float32), (1000, 1))
    onnx_logits = pytorch_logits.copy()
    pytorch_logits[0] = np.asarray([0.01, 0.011], dtype=np.float32)
    onnx_logits[0] = np.asarray([0.011, 0.01], dtype=np.float32)

    parity = compare_classifier_outputs(pytorch_logits, onnx_logits)

    assert parity["passed"] is True
    assert parity["top1_agreement"] == 0.999


def test_classifier_output_parity_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_classifier_outputs(
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )


def test_classifier_run_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "run"
    paths = FolderClassifierRunPaths(
        run_root=run_root,
        checkpoint_root=tmp_path / "checkpoints" / "run",
        checkpoint_path=tmp_path / "checkpoints" / "run" / "best.pt",
        latest_checkpoint_path=tmp_path / "checkpoints" / "run" / "latest.pt",
        progress_path=run_root / "progress.json",
        artifact_root=tmp_path / "artifacts" / "run",
        onnx_path=tmp_path / "artifacts" / "run" / "sign_classifier.onnx",
        labels_path=tmp_path / "artifacts" / "run" / "sign_classifier.labels.json",
        calibration_path=tmp_path
        / "artifacts"
        / "run"
        / "sign_classifier.calibration.json",
    )
    run_root.mkdir(parents=True)
    (run_root / "existing-output.txt").write_text("fixture", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Choose a unique --name"):
        prepare_classifier_run_paths(paths, overwrite=False)

    prepare_classifier_run_paths(paths, overwrite=True)
    assert paths.checkpoint_root.is_dir()
    assert paths.artifact_root.is_dir()


def test_classifier_run_accepts_an_empty_run_directory_after_a_failed_launch(
    tmp_path: Path,
) -> None:
    paths = FolderClassifierRunPaths(
        run_root=tmp_path / "outputs" / "run",
        checkpoint_root=tmp_path / "checkpoints" / "run",
        checkpoint_path=tmp_path / "checkpoints" / "run" / "best.pt",
        latest_checkpoint_path=tmp_path / "checkpoints" / "run" / "latest.pt",
        progress_path=tmp_path / "outputs" / "run" / "progress.json",
        artifact_root=tmp_path / "artifacts" / "run",
        onnx_path=tmp_path / "artifacts" / "run" / "sign_classifier.onnx",
        labels_path=tmp_path / "artifacts" / "run" / "sign_classifier.labels.json",
        calibration_path=tmp_path / "artifacts" / "run" / "sign_classifier.calibration.json",
    )
    # Model initialisation can fail before the first epoch (for example, a
    # transient pretrained-weight download failure), after all three output
    # directories have been created but before any result exists.
    paths.run_root.mkdir(parents=True)
    paths.checkpoint_root.mkdir(parents=True)
    paths.artifact_root.mkdir(parents=True)

    prepare_classifier_run_paths(paths, overwrite=False)

    assert paths.checkpoint_root.is_dir()
    assert paths.artifact_root.is_dir()


def test_authorized_internal_release_is_not_experimental(tmp_path: Path) -> None:
    metadata = {
        "annotation_status": "approved_with_internal_academic_exception",
        "coursework_images_included": 0,
        "phase_b_authorized": True,
        "internal_academic_only": True,
        "internal_runtime_promotion_eligible": True,
        "external_data_or_model_release_allowed": False,
        "dvc_remote_push_allowed": False,
        "publication_prohibited": True,
    }
    (tmp_path / "dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    assert validate_classifier_dataset(tmp_path, allow_unreviewed_experiment=False) == metadata
    assert should_export_classifier_experimentally(metadata, allow_unreviewed_experiment=False) is False

    metadata["publication_prohibited"] = False
    (tmp_path / "dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Phase-B safeguards"):
        validate_classifier_dataset(tmp_path, allow_unreviewed_experiment=False)


def test_validation_threshold_selection_maximizes_qualifying_coverage() -> None:
    probabilities = np.asarray(
        [[0.90, 0.10], [0.80, 0.20], [0.55, 0.45], [0.40, 0.60]],
        dtype=np.float32,
    )
    targets = np.asarray([0, 0, 1, 0], dtype=np.int64)

    selected = select_confidence_threshold(
        probabilities,
        targets,
        fallback_threshold=0.72,
        target_selective_accuracy=0.80,
    )

    assert selected["accepted_samples"] == 2
    assert selected["selective_accuracy"] == 1.0
    assert selected["coverage"] == 0.5
    assert selected["threshold"] == pytest.approx(0.70)
    assert selected["threshold_stability_margin"] == pytest.approx(0.10)


def test_stable_validation_threshold_avoids_backend_rounding_flip() -> None:
    probabilities = np.asarray(
        [[0.9783694, 0.0216306], [0.9671449, 0.0328551]],
        dtype=np.float32,
    )
    targets = np.asarray([0, 1], dtype=np.int64)

    selected = select_confidence_threshold(
        probabilities,
        targets,
        fallback_threshold=0.9783693,
        target_selective_accuracy=1.0,
    )

    assert 0.9671449 < selected["threshold"] < 0.9783694
    assert probabilities[0, 0] - selected["threshold"] > 0.005


def test_classifier_run_resume_requires_latest_checkpoint(tmp_path: Path) -> None:
    paths = FolderClassifierRunPaths(
        run_root=tmp_path / "outputs" / "run",
        checkpoint_root=tmp_path / "checkpoints" / "run",
        checkpoint_path=tmp_path / "checkpoints" / "run" / "best.pt",
        latest_checkpoint_path=tmp_path / "checkpoints" / "run" / "latest.pt",
        progress_path=tmp_path / "outputs" / "run" / "progress.json",
        artifact_root=tmp_path / "artifacts" / "run",
        onnx_path=tmp_path / "artifacts" / "run" / "sign_classifier.onnx",
        labels_path=tmp_path / "artifacts" / "run" / "sign_classifier.labels.json",
        calibration_path=tmp_path / "artifacts" / "run" / "sign_classifier.calibration.json",
    )
    paths.run_root.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="latest checkpoint"):
        prepare_classifier_run_paths(paths, overwrite=False, resume=True)
    paths.latest_checkpoint_path.parent.mkdir(parents=True)
    paths.latest_checkpoint_path.write_bytes(b"fixture")
    prepare_classifier_run_paths(paths, overwrite=False, resume=True)
