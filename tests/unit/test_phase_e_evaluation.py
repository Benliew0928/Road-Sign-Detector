from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
from pathlib import Path
from typing import Any, cast

import pytest

from roadsign_assist.catalogue.models import ActionCode
from roadsign_assist.evaluation import phase_e
from roadsign_assist.inference.models import BoundingBoxModel


def _metrics(profile: str) -> dict[str, Any]:
    return {
        "detector": {
            "overall": {"recall": 0.91},
            "full_road": {"recall": 0.91},
            "small": {"recall": 0.81},
        },
        "conditional_classifier": {"macro_f1": 0.86},
        "end_to_end": {"macro_f1": 0.86, "safety_critical_recall": 0.91},
        "unknown": {"auroc": 0.86},
        "no_sign": {"image_false_positive_rate": 0.02, "false_boxes_per_100_images": 2.0},
        "safety": {
            "unsafe_strong_actions": 0,
            "error_signatures": [],
            "ocr_advisory_contract": {"passed": True},
        },
        "runtime": {
            "coursework": {
                "images": 84,
                "completed": 84,
                "primary_correct": 80,
                "semantic_accuracy": 80 / 84,
                "numeric_accuracy": 0.9,
                "all_under_two_seconds": True,
                "unsafe_strong_actions": 0,
                "runtime_ms": {"maximum": 1000.0},
            }
            if profile == "cpu"
            else None,
            "streaming": {
                "frames_per_second": 16.0,
                "first_stable_warning_seconds": 0.5,
            }
            if profile == "gpu"
            else None,
        },
    }


def _write_profile_metrics(root: Path, profile: str, payload: dict[str, Any]) -> None:
    path = root / phase_e.PHASE_E_OUTPUT / profile / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_box_matching_is_one_to_one_at_gate_iou() -> None:
    truth = [
        BoundingBoxModel(x1=0, y1=0, x2=10, y2=10),
        BoundingBoxModel(x1=20, y1=20, x2=30, y2=30),
    ]
    predicted = [
        BoundingBoxModel(x1=0, y1=0, x2=10, y2=10),
        BoundingBoxModel(x1=1, y1=1, x2=9, y2=9),
    ]

    matches = phase_e.match_boxes(truth, predicted, minimum_iou=0.50)

    assert matches == [(0, 0, 1.0)]


def test_wrong_numeric_parameter_is_an_unsafe_strong_action() -> None:
    assert phase_e._unsafe_strong_action(
        "maximum_speed",
        "maximum_speed",
        ActionCode.SET_TARGET_SPEED,
        True,
        parameter_correct=False,
    )


def test_gate_emits_do_not_promote_and_cannot_mutate_runtime(tmp_path: Path) -> None:
    gpu = _metrics("gpu")
    cpu = _metrics("cpu")
    _write_profile_metrics(tmp_path, "gpu", gpu)
    _write_profile_metrics(tmp_path, "cpu", cpu)
    passed = phase_e.finalize_phase_e_gate(project_root=tmp_path)
    assert passed["decision"] == "promote"
    assert passed["passed_gate_count"] == passed["total_gate_count"]

    gpu["detector"]["full_road"]["recall"] = 0.89
    _write_profile_metrics(tmp_path, "gpu", gpu)
    failed = phase_e.finalize_phase_e_gate(project_root=tmp_path)
    assert failed["decision"] == "do_not_promote"
    assert "full_road_detector_recall" in cast(list[str], failed["failures"])

    runtime = tmp_path / "models/exported/runtime"
    runtime.mkdir(parents=True)
    marker = runtime / "unchanged.txt"
    marker.write_text("legacy", encoding="utf-8")
    with pytest.raises(ValueError, match="Gate E did not pass"):
        phase_e.promote_phase_e_runtime(project_root=tmp_path, internal_only=True)
    assert marker.read_text(encoding="utf-8") == "legacy"


class _FakeEngine:
    def __init__(self, _config: object) -> None:
        pass

    def warmup(self) -> dict[str, bool]:
        return {"detector": True, "classifier": True}

    def process_frame(self, _image: object, *, assume_stable: bool) -> object:
        assert assume_stable is True
        return object()


def test_successful_promotion_has_exact_recoverable_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "models/exported/runtime"
    runtime.mkdir(parents=True)
    classifier_files = {
        "sign_classifier.onnx": b"classifier",
        "sign_classifier.labels.json": b"labels",
        "sign_classifier.calibration.json": b"calibration",
        "sign_classifier.runtime.json": b"runtime",
        "old_note.txt": b"legacy bundle",
    }
    for name, value in classifier_files.items():
        (runtime / name).write_bytes(value)
    original_hashes = phase_e._tree_hashes(runtime)
    candidate = tmp_path / "models/candidates/phase_d" / phase_e.DETECTOR_ID
    candidate.mkdir(parents=True)
    (candidate / "model.pt").write_bytes(b"pt-detector")
    (candidate / "model.onnx").write_bytes(b"onnx-detector")
    gate = tmp_path / phase_e.GATE_PATH
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps({"gate_e_passed": True, "decision": "promote"}), encoding="utf-8")
    dvc = tmp_path / "models/exported/runtime.dvc"
    dvc.write_bytes(b"unchanged dvc pointer")

    monkeypatch.setattr(phase_e, "DETECTOR_PT_SHA256", phase_e._sha256(candidate / "model.pt"))
    monkeypatch.setattr(phase_e, "DETECTOR_ONNX_SHA256", phase_e._sha256(candidate / "model.onnx"))
    monkeypatch.setattr(
        phase_e, "CLASSIFIER_ONNX_SHA256", phase_e._sha256(runtime / "sign_classifier.onnx")
    )
    monkeypatch.setattr(
        phase_e,
        "CLASSIFIER_LABELS_SHA256",
        phase_e._sha256(runtime / "sign_classifier.labels.json"),
    )
    monkeypatch.setattr(
        phase_e,
        "CLASSIFIER_CALIBRATION_SHA256",
        phase_e._sha256(runtime / "sign_classifier.calibration.json"),
    )

    def fake_preflight(**_kwargs: object) -> dict[str, object]:
        return {"benchmark": {"sha256": "benchmark"}}

    monkeypatch.setattr(phase_e, "validate_phase_e_artifacts", fake_preflight)
    monkeypatch.setattr(phase_e, "InferenceEngine", _FakeEngine)

    promoted = phase_e.promote_phase_e_runtime(project_root=tmp_path, internal_only=True)

    backup_id = str(promoted["backup_id"])
    backup = tmp_path / "models/exported/runtime_backups" / backup_id
    assert phase_e._tree_hashes(backup) == original_hashes
    assert (runtime / "sign_detector.pt").read_bytes() == b"pt-detector"
    assert not (runtime / "old_note.txt").exists()
    assert dvc.read_bytes() == b"unchanged dvc pointer"
    assert len(str(promoted["pipeline_manifest_sha256"])) == 64

    restored = phase_e.rollback_phase_e_runtime(backup_id, project_root=tmp_path)

    assert restored["restored_backup_id"] == backup_id
    assert phase_e._tree_hashes(runtime) == original_hashes
    assert (runtime / "old_note.txt").read_bytes() == b"legacy bundle"
