from __future__ import annotations

import json
from pathlib import Path

import pytest

from roadsign_assist.classification import phase_b


def test_phase_b_matrix_is_the_predeclared_15_run_sweep() -> None:
    specs = phase_b.phase_b_run_specs()
    assert len(specs) == 15
    assert len(phase_b.phase_b_run_specs("baseline")) == 9
    assert len(phase_b.phase_b_run_specs("capacity")) == 6
    assert {spec.seed for spec in specs} == {2513, 1337, 2026}
    assert {spec.image_size for spec in specs} == {224, 256, 320}


def test_phase_b_selection_reads_validation_only_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(phase_b, "project_path", lambda path: tmp_path / path)
    for spec in phase_b.phase_b_run_specs():
        run_root = tmp_path / "outputs/training" / spec.run_name
        run_root.mkdir(parents=True)
        f1 = 0.90
        if spec.architecture == "convnext_tiny":
            f1 = 0.94
        elif spec.architecture == "efficientnet_v2_m":
            f1 = 0.93
        payload = {
            "dataset_id": phase_b.PHASE_B_RELEASE_ID,
            "experimental": False,
            "evaluation_split": "validation",
            "configuration": {
                "architecture": spec.architecture,
                "image_size": spec.image_size,
                "batch_size": spec.batch_size,
                "seed": spec.seed,
                "epochs": 40,
            },
            "validation": {"macro_f1_all_labels": f1, "accuracy": f1, "ece": 0.02},
        }
        (run_root / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    report = phase_b.select_phase_b_candidate("outputs/training/selection.json")

    assert report["selected_single"]["architecture"] == "convnext_tiny"
    assert report["selected_single"]["primary_run"] == "pb_v3_convnexttiny_320_s2513"
    assert report["ensemble"] is not None
    assert (tmp_path / "outputs/training/selection.json").is_file()
