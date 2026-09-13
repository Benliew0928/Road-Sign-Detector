from __future__ import annotations

# pyright: reportPrivateUsage=false
from collections import Counter
from pathlib import Path

import pytest

from roadsign_assist.datasets.recovery_v12 import (
    _assign_components,
    _components,
    _is_placeholder,
    _known_group_leakage,
)
from roadsign_assist.recovery_training import _run_stage


def _row(index: int, *, kind: str = "sign", source: str = "kartaview_malaysia") -> dict[str, str]:
    sample = f"sample-{index:03d}"
    group = f"group-{index:03d}"
    return {
        "sample_id": sample,
        "instance_id": f"instance-{index:03d}",
        "split": "train",
        "expected_kind": kind,
        "semantic_sign_id": "stop" if kind == "sign" else "unknown_sign",
        "source_id": source,
        "capture_evidence": "public_road_image",
        "image_sha256": f"{index:064x}",
        "source_artwork_id": "",
        "base_rendering_id": "",
        "session_id": group,
        "route_id": group,
        "video_id": "",
        "physical_sign_id": "unknown_not_recorded",
        "related_capture_group_id": group,
        "review_notes": "Project owner accepted corrected full-context sheet fixture.",
    }


def test_placeholders_do_not_fabricate_identity_links() -> None:
    assert _is_placeholder("unknown_not_recorded")
    assert _is_placeholder("")
    assert not _is_placeholder("kartaview-sequence:123")
    rows = [_row(index) for index in range(30)]
    components = _components(rows)
    assert len(components) == 30
    assert all(component.eligible for component in components)


def test_known_related_groups_are_kept_together_and_coarse_sources_stay_train() -> None:
    first = _row(1)
    second = _row(2)
    second["related_capture_group_id"] = first["related_capture_group_id"]
    second["session_id"] = first["session_id"]
    second["route_id"] = first["route_id"]
    coarse = [_row(100 + index, source="tt100k_official") for index in range(3)]
    for row in coarse:
        row["related_capture_group_id"] = "tt100k-official:unknown-capture-group"
        row["session_id"] = "unknown_not_recorded"
        row["route_id"] = "unknown_not_recorded"
        row["capture_evidence"] = "unverified_context"
    components = _components([first, second, *coarse])
    assert sorted(len(component.samples) for component in components) == [2, 3]
    assignments = _assign_components(components)
    coarse_component = next(component for component in components if not component.eligible)
    assert assignments[coarse_component.component_id] == "train"


def test_eligible_components_receive_deterministic_balanced_split() -> None:
    rows: list[dict[str, str]] = []
    for index in range(60):
        kind = "sign" if index < 20 else "unknown_sign" if index < 40 else "no_sign"
        rows.append(_row(index, kind=kind))
    components = _components(rows)
    first = _assign_components(components)
    second = _assign_components(list(reversed(components)))
    assert first == second
    counts = Counter(first.values())
    assert counts == {"train": 42, "validation": 9, "internal_test": 9}


def test_known_leakage_is_distinct_from_unknown_identity_limitations() -> None:
    first = _row(1)
    second = _row(2)
    second["split"] = "validation"
    second["session_id"] = first["session_id"]
    leakage, unknowns = _known_group_leakage([first, second])
    assert leakage["session_id"][0]["splits"] == ["train", "validation"]
    assert "physical_sign_id" in unknowns
    assert "physical_sign_id" not in leakage


def test_stage_receipts_verify_outputs_and_preflight_can_be_forced(tmp_path: Path) -> None:
    calls = 0
    output = tmp_path / "artifact.txt"

    def stage() -> list[Path]:
        nonlocal calls
        calls += 1
        output.write_text(str(calls), encoding="utf-8")
        return [output]

    _run_stage(tmp_path, "fixture", stage)
    _run_stage(tmp_path, "fixture", stage)
    assert calls == 1

    _run_stage(tmp_path, "fixture", stage, always_run=True)
    assert calls == 2

    output.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid outputs"):
        _run_stage(tmp_path, "fixture", stage)
