import csv
import json
from pathlib import Path

from roadsign_assist.datasets.split import build_grouped_splits


def test_grouped_split_keeps_physical_sign_together(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.csv"
    rows = [
        {
            "sample_id": f"sample-{index}",
            "source_id": "test",
            "path": f"{index}.jpg",
            "semantic_sign_id": "stop",
            "group_id": f"group-{index // 2}",
            "route_id": "",
            "session_id": "",
            "physical_sign_id": f"sign-{index // 2}",
            "has_sign": "true",
            "is_synthetic": "false",
            "annotation_status": "approved",
        }
        for index in range(30)
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    splits = build_grouped_splits(
        manifest,
        tmp_path / "splits",
        seed=2513,
        train_fraction=0.7,
        validation_fraction=0.15,
    )
    group_splits: dict[str, set[str]] = {}
    for split_rows in splits.values():
        for row in split_rows:
            group_splits.setdefault(row["effective_group"], set()).add(row["split"])
    assert all(len(values) == 1 for values in group_splits.values())
    assert all(splits[split] for split in ("train", "validation", "test"))


def test_grouped_split_stratifies_repeated_labels(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.csv"
    rows = [
        {
            "sample_id": f"{label}-{index}",
            "source_id": "test",
            "path": f"{label}-{index}.jpg",
            "semantic_sign_id": label,
            "group_id": f"{label}-group-{index}",
            "route_id": "",
            "session_id": "",
            "physical_sign_id": "",
            "has_sign": "true",
            "is_synthetic": "false",
            "annotation_status": "approved",
        }
        for label in ("stop", "maximum_speed", "school_zone")
        for index in range(10)
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "splits"
    first = build_grouped_splits(
        manifest,
        output,
        seed=2513,
        train_fraction=0.7,
        validation_fraction=0.15,
    )
    second = build_grouped_splits(
        manifest,
        output,
        seed=2513,
        train_fraction=0.7,
        validation_fraction=0.15,
    )
    assert {
        split: [row["sample_id"] for row in split_rows] for split, split_rows in first.items()
    } == {split: [row["sample_id"] for row in split_rows] for split, split_rows in second.items()}
    for split_rows in first.values():
        assert {row["semantic_sign_id"] for row in split_rows} == {
            "stop",
            "maximum_speed",
            "school_zone",
        }
    diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["splits"]["train"]["samples"] > 0


def test_adjacent_video_frames_with_same_session_never_cross_splits(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, str]] = []
    for session_index in range(12):
        for frame_index in range(5):
            rows.append(
                {
                    "sample_id": f"session-{session_index}-frame-{frame_index:03d}",
                    "source_id": "local_video",
                    "path": f"session-{session_index}/frame-{frame_index:03d}.jpg",
                    "semantic_sign_id": "stop" if session_index % 2 else "give_way",
                    "group_id": "",
                    "physical_sign_id": "",
                    "route_id": f"route-{session_index // 3}",
                    "session_id": f"session-{session_index}",
                    "has_sign": "true",
                    "is_synthetic": "false",
                    "annotation_status": "approved",
                }
            )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    splits = build_grouped_splits(
        manifest,
        tmp_path / "splits",
        seed=2513,
        train_fraction=0.7,
        validation_fraction=0.15,
    )
    session_splits: dict[str, set[str]] = {}
    for split_rows in splits.values():
        for row in split_rows:
            session_splits.setdefault(row["session_id"], set()).add(row["split"])
    assert all(len(values) == 1 for values in session_splits.values())
