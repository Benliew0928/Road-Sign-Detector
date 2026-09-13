from __future__ import annotations

# pyright: reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownMemberType=false
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from roadsign_assist.detection import phase_d


def _prediction_record(
    *,
    ground_truth: list[list[float]],
    areas: list[float],
    predicted: list[list[float]],
    scores: list[float],
    layout: str = "source/EMTD_YOLO_33",
    negative: bool = False,
) -> dict[str, Any]:
    return {
        "image": "unused.jpg",
        "ground_truth": np.asarray(ground_truth, dtype=np.float64).reshape((-1, 4)),
        "ground_truth_areas": np.asarray(areas, dtype=np.float64),
        "predicted": np.asarray(predicted, dtype=np.float64).reshape((-1, 4)),
        "scores": np.asarray(scores, dtype=np.float64),
        "layout_root_id": layout,
        "is_negative": negative,
    }


def test_phase_d_matrix_is_fixed_and_balanced() -> None:
    specs = phase_d.phase_d_run_specs()
    assert [spec.candidate_id for spec in specs] == [
        "pd_v1_yolo26s_640_s2513",
        "pd_v1_yolo26s_960_s2513",
        "pd_v1_yolo26m_960_s2513",
        "pd_v1_yolo26s_p2_960_s2513",
    ]
    assert {spec.image_size for spec in specs} == {640, 960}
    assert sum(spec.p2 for spec in specs) == 1
    assert [spec.candidate_id for spec in phase_d.phase_d_active_run_specs()] == [
        "pd_v1_yolo26s_640_s2513",
        "pd_v1_yolo26s_960_s2513",
        "pd_v1_yolo26m_960_s2513",
    ]


def test_detection_and_empty_labels_are_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "images" / "validation" / "sample.jpg"
    label_path = tmp_path / "labels" / "validation" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(image_path)
    label_path.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    monkeypatch.setattr(phase_d, "_label_path", lambda _path: label_path)

    boxes, areas = phase_d._read_ground_truth(image_path)

    assert boxes == pytest.approx(np.asarray([[80.0, 30.0, 120.0, 70.0]]))
    assert areas.tolist() == pytest.approx([0.08])

    label_path.write_text("", encoding="utf-8")
    boxes, areas = phase_d._read_ground_truth(image_path)
    assert boxes.shape == (0, 4)
    assert areas.shape == (0,)


def test_operational_metrics_cover_small_dense_and_no_sign() -> None:
    records = [
        _prediction_record(
            ground_truth=[[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
            areas=[0.0005, 0.005, 0.2],
            predicted=[[0, 0, 10, 10], [20, 20, 30, 30], [60, 60, 70, 70]],
            scores=[0.9, 0.8, 0.7],
        ),
        _prediction_record(
            ground_truth=[],
            areas=[],
            predicted=[[1, 1, 2, 2]],
            scores=[0.8],
            layout="negative",
            negative=True,
        ),
    ]

    report = phase_d.operational_metrics(records, confidence=0.5)

    assert report["overall"]["true_positive"] == 2
    assert report["overall"]["false_positive"] == 2
    assert report["overall"]["false_negative"] == 1
    assert report["small"]["recall"] == 1.0
    assert report["very_small_proxy"]["recall"] == 1.0
    assert report["dense"]["images"] == 1
    assert report["source_kind"]["unspecified"]["images"] == 2
    assert report["source_id"]["unspecified"]["images"] == 2
    assert report["no_sign"]["image_false_positive_rate"] == 1.0
    assert report["no_sign"]["false_boxes_per_100_images"] == 100.0


def test_selection_uses_validation_windows_and_stable_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(phase_d, "validate_phase_d_release", lambda: {})
    monkeypatch.setattr(phase_d, "project_path", lambda path: tmp_path / Path(path))
    metrics_by_candidate = {
        "pd_v1_yolo26s_640_s2513": (0.700, 0.80, 0.70, 8.0),
        "pd_v1_yolo26s_960_s2513": (0.704, 0.89, 0.75, 12.0),
        "pd_v1_yolo26m_960_s2513": (0.705, 0.90, 0.76, 20.0),
        "pd_v1_yolo26s_p2_960_s2513": (0.699, 0.95, 0.77, 15.0),
    }

    def fake_evaluate(spec: phase_d.PhaseDRunSpec, _device: str) -> dict[str, Any]:
        map_value, small_recall, recall, latency = metrics_by_candidate[spec.candidate_id]
        report = {
            "candidate_id": spec.candidate_id,
            "dataset_id": phase_d.PHASE_D_RELEASE_ID,
            "evaluation_split": "validation",
            "test_evaluated": False,
            "checkpoint": f"outputs/training/{spec.candidate_id}/weights/best.pt",
            "checkpoint_sha256": spec.candidate_id,
            "selected_confidence": 0.25,
            "metrics": {
                "map50_95": map_value,
                "map50": map_value + 0.1,
                "precision": 0.8,
                "recall": recall,
                "speed_ms": {"inference": latency},
            },
            "operational": {
                "small": {"recall": small_recall},
                "very_small_proxy": {"recall": small_recall - 0.1},
            },
        }
        report_path = (
            tmp_path / "outputs" / "training" / spec.candidate_id / "validation_metrics.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(phase_d, "_evaluate_validation_candidate", fake_evaluate)
    output = tmp_path / "selection.json"

    report = phase_d.select_phase_d_detector(output, device="0")

    assert report["selection_split"] == "validation"
    assert report["test_evaluated"] is False
    assert report["selected"]["candidate_id"] == "pd_v1_yolo26m_960_s2513"
    assert "pd_v1_yolo26s_p2_960_s2513" not in report["map_eligible_candidates"]
    assert (
        phase_d._verify_phase_d_selection(report).candidate_id == report["selected"]["candidate_id"]
    )

    tampered = json.loads(json.dumps(report))
    tampered["selected"] = tampered["candidates"][0]
    with pytest.raises(ValueError, match="locked validation ranking"):
        phase_d._verify_phase_d_selection(tampered)


def test_locked_test_requires_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase_d, "validate_phase_d_release", lambda: {})
    monkeypatch.setattr(phase_d, "project_path", lambda path: tmp_path / Path(path))

    with pytest.raises(FileNotFoundError, match="validation selection"):
        phase_d.evaluate_phase_d_selected_detector()


def test_run_signature_mismatch_refuses_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = phase_d.phase_d_run_specs()[0]
    run_root = tmp_path / spec.candidate_id
    run_root.mkdir()
    (run_root / "run_manifest.json").write_text(
        json.dumps({"run_signature": "recorded"}), encoding="utf-8"
    )
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"weight")
    monkeypatch.setattr(phase_d, "validate_phase_d_release", lambda: {})
    monkeypatch.setattr(phase_d, "_resolve_pretrained_weight", lambda _spec: weight)
    monkeypatch.setattr(phase_d, "_run_root", lambda _spec: run_root)
    monkeypatch.setattr(
        phase_d,
        "_build_run_manifest",
        lambda _spec, _weight, _device: {"run_signature": "changed"},
    )

    with pytest.raises(ValueError, match="signature mismatch"):
        phase_d.train_phase_d_detector(spec.candidate_id, resume=True)


def test_matching_complete_run_is_skipped_and_incomplete_run_requires_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = phase_d.phase_d_run_specs()[0]
    run_root = tmp_path / spec.candidate_id
    run_root.mkdir()
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"weight")
    (run_root / "run_manifest.json").write_text(
        json.dumps({"run_signature": "same"}), encoding="utf-8"
    )
    monkeypatch.setattr(phase_d, "validate_phase_d_release", lambda: {})
    monkeypatch.setattr(phase_d, "_disable_ultralytics_image_repair", lambda: None)
    monkeypatch.setattr(phase_d, "_resolve_pretrained_weight", lambda _spec: weight)
    monkeypatch.setattr(phase_d, "_run_root", lambda _spec: run_root)
    monkeypatch.setattr(
        phase_d,
        "_build_run_manifest",
        lambda _spec, _weight, _device: {"run_signature": "same"},
    )

    with pytest.raises(FileExistsError, match="--resume"):
        phase_d.train_phase_d_detector(spec.candidate_id)
    with pytest.raises(FileNotFoundError, match="missing epoch checkpoint"):
        phase_d.train_phase_d_detector(spec.candidate_id, resume=True)

    (run_root / "training_complete.json").write_text(
        json.dumps({"status": "completed", "candidate_id": spec.candidate_id}),
        encoding="utf-8",
    )
    result = phase_d.train_phase_d_detector(spec.candidate_id, resume=True)
    assert result["status"] == "skipped_complete"


def test_status_reports_completed_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase_d, "_run_root", lambda spec: tmp_path / spec.candidate_id)
    spec = phase_d.phase_d_active_run_specs()[0]
    run_root = tmp_path / spec.candidate_id
    run_root.mkdir()
    (run_root / "results.csv").write_text(
        "epoch,metrics/mAP50-95(B),metrics/mAP50(B)\n1,0.55,0.75\n",
        encoding="utf-8",
    )
    (run_root / "training_complete.json").write_text(
        json.dumps({"status": "completed", "candidate_id": spec.candidate_id}),
        encoding="utf-8",
    )

    status = phase_d.phase_d_status()

    assert status["completed_runs"] == 1
    assert status["total_runs"] == 3
    assert status["runs"][0]["status"] == "completed"
    assert status["runs"][0]["epoch"] == 2


def test_parity_uses_metric_specific_tolerances() -> None:
    passing = phase_d._parity_report(
        {"map50": 0.8, "map50_95": 0.7, "precision": 0.9, "recall": 0.8},
        {"map50": 0.796, "map50_95": 0.696, "precision": 0.891, "recall": 0.791},
    )
    failing = phase_d._parity_report(
        {"map50": 0.8, "map50_95": 0.7, "precision": 0.9, "recall": 0.8},
        {"map50": 0.79, "map50_95": 0.7, "precision": 0.9, "recall": 0.8},
    )
    assert passing["passed"] is True
    assert failing["passed"] is False
    assert failing["failures"] == ["map50"]
