from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from roadsign_assist.inference import freezing


def test_freeze_rejects_dirty_worktree_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dirty_git(_root: Path) -> dict[str, object]:
        return {"clean_worktree": False, "status_porcelain": [" M file.py"]}

    monkeypatch.setattr(
        freezing,
        "_git_metadata",
        dirty_git,
    )
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="clean Git worktree"):
        freezing.freeze_runtime_bundle(
            config_path="candidate.yaml",
            output_path=output,
            dataset_releases={"data_v1": tmp_path / "data.json"},
            evaluation_commands=["evaluate --locked"],
            project_root=tmp_path,
        )

    assert not output.exists()


def test_freeze_materializes_hashed_runtime_and_dataset_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "candidate.yaml"
    lock = tmp_path / "uv.lock"
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    labels = tmp_path / "labels.json"
    calibration = tmp_path / "calibration.json"
    catalogue = tmp_path / "catalogue.json"
    tracker = tmp_path / "tracker.yaml"
    dataset = tmp_path / "dataset.json"
    for path in (config, lock, detector, classifier, labels, calibration, catalogue, tracker, dataset):
        path.write_text(path.name, encoding="utf-8")

    identity = {
        name: {"path": str(path), "available": True}
        for name, path in {
            "detector": detector,
            "classifier": classifier,
            "classifier_labels": labels,
            "classifier_calibration": calibration,
            "catalogue": catalogue,
            "tracker_config": tracker,
        }.items()
    }
    fake_engine = SimpleNamespace(
        runtime_badge="CANDIDATE",
        config_name="candidate_v1",
        preprocessing_version="roadsign_raw_bgr_v1",
        bundle_identity=identity,
    )

    def clean_git(_root: Path) -> dict[str, object]:
        return {
            "commit": "a" * 40,
            "branch": "codex/test",
            "clean_worktree": True,
            "status_porcelain": [],
        }

    def build_fake_engine(_config: Path) -> SimpleNamespace:
        return fake_engine

    def create_fake_git_bundle(_root: Path, destination: Path) -> dict[str, object]:
        destination.write_bytes(b"test git bundle")
        return {
            "source_path": str(tmp_path),
            "bundle_path": destination.name,
            "sha256": freezing._sha256(destination),
            "size_bytes": destination.stat().st_size,
            "materialization": "git_bundle",
            "verified": True,
        }

    monkeypatch.setattr(
        freezing,
        "_git_metadata",
        clean_git,
    )
    monkeypatch.setattr(freezing, "InferenceEngine", build_fake_engine)
    monkeypatch.setattr(freezing, "_create_git_bundle", create_fake_git_bundle)

    result = freezing.freeze_runtime_bundle(
        config_path=config,
        output_path="bundles/candidate_v1",
        dataset_releases={"data_v1": dataset},
        evaluation_commands=["roadsign-assist evaluate --locked"],
        project_root=tmp_path,
    )

    output = tmp_path / "bundles/candidate_v1"
    manifest = json.loads((output / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert result["runtime_badge"] == "CANDIDATE"
    assert manifest["git"]["clean_worktree"] is True
    assert manifest["dataset_release_ids"] == ["data_v1"]
    assert set(manifest["artifacts"]) >= {
        "config",
        "dependency_lock",
        "detector",
        "classifier",
        "classifier_labels",
        "classifier_calibration",
        "catalogue",
        "tracker_config",
        "dataset__data_v1",
        "source_repository",
    }
    assert manifest["artifacts"]["source_repository"]["verified"] is True
    assert len(str(result["manifest_sha256"])) == 64
