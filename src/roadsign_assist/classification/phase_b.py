"""Stable Phase-B experiment matrix, progress summaries, and validation selection."""

from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from roadsign_assist.paths import project_path

PHASE_B_RELEASE_ID = "classifier_production_78_v3_20260829"
PHASE_B_DATA_ROOT = "data/processed/classifier_production_78_v3_20260829"
PHASE_B_SEEDS = (2513, 1337, 2026)
PHASE_B_OUTPUT = "outputs/training/phase_b_v3_selection.json"


@dataclass(frozen=True)
class PhaseBRunSpec:
    stage: Literal["baseline", "capacity"]
    architecture: str
    image_size: int
    batch_size: int
    seed: int

    @property
    def run_name(self) -> str:
        architecture = self.architecture.replace("efficientnet_v2_", "effnetv2").replace(
            "convnext_", "convnext"
        )
        return f"pb_v3_{architecture}_{self.image_size}_s{self.seed}"

    @property
    def family(self) -> str:
        return "convnext" if self.architecture.startswith("convnext") else "efficientnet"


def phase_b_run_specs(stage: Literal["baseline", "capacity", "all"] = "all") -> list[PhaseBRunSpec]:
    definitions = (
        ("baseline", "efficientnet_v2_s", 224, 32),
        ("baseline", "efficientnet_v2_s", 256, 32),
        ("baseline", "efficientnet_v2_s", 320, 32),
        ("capacity", "convnext_tiny", 320, 24),
        ("capacity", "efficientnet_v2_m", 320, 12),
    )
    return [
        PhaseBRunSpec(cast(Literal["baseline", "capacity"], cast_stage), architecture, image_size, batch_size, seed)
        for cast_stage, architecture, image_size, batch_size in definitions
        if stage == "all" or stage == cast_stage
        for seed in PHASE_B_SEEDS
    ]


def phase_b_status() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in phase_b_run_specs():
        run_root = project_path("outputs/training") / spec.run_name
        metrics_path = run_root / "metrics.json"
        progress_path = run_root / "progress.json"
        row: dict[str, Any] = {**asdict(spec), "run_name": spec.run_name, "status": "pending"}
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            validation = metrics.get("validation", {})
            row.update(
                {
                    "status": "completed",
                    "best_epoch": metrics.get("best_epoch"),
                    "validation_macro_f1": validation.get("macro_f1_all_labels"),
                    "validation_accuracy": validation.get("accuracy"),
                    "validation_ece": validation.get("ece"),
                }
            )
        elif progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            row.update({"status": progress.get("status", "running"), **progress})
        elif run_root.exists():
            row["status"] = "incomplete_without_progress"
        rows.append(row)
    completed = sum(row["status"] == "completed" for row in rows)
    return {
        "schema_version": "1.0",
        "release_id": PHASE_B_RELEASE_ID,
        "completed_runs": completed,
        "total_runs": len(rows),
        "runs": rows,
    }


def _completed_metrics(spec: PhaseBRunSpec) -> dict[str, Any]:
    path = project_path("outputs/training") / spec.run_name / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Phase-B run is incomplete: {spec.run_name}")
    metrics: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    configuration = metrics.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"Phase-B run lacks configuration: {spec.run_name}")
    expected = {
        "architecture": spec.architecture,
        "image_size": spec.image_size,
        "batch_size": spec.batch_size,
        "seed": spec.seed,
        "epochs": 40,
    }
    actual: dict[str, Any] = {key: configuration.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Phase-B run configuration mismatch for {spec.run_name}: {actual}")
    if metrics.get("dataset_id") != PHASE_B_RELEASE_ID:
        raise ValueError(f"Phase-B run used the wrong release: {spec.run_name}")
    if metrics.get("experimental") is True:
        raise ValueError(f"Phase-B run must not be experimental: {spec.run_name}")
    if metrics.get("evaluation_split") != "validation":
        raise ValueError(
            f"Phase-B selection refuses test-evaluated run {spec.run_name}; use a fresh run."
        )
    return metrics


def select_phase_b_candidate(
    output_path: str | Path = PHASE_B_OUTPUT,
) -> dict[str, Any]:
    """Select only from validation reports; this function never reads model/test data."""
    groups: dict[tuple[str, int], list[tuple[PhaseBRunSpec, dict[str, Any]]]] = {}
    for spec in phase_b_run_specs():
        groups.setdefault((spec.architecture, spec.image_size), []).append(
            (spec, _completed_metrics(spec))
        )

    candidates: list[dict[str, Any]] = []
    for (architecture, image_size), members in groups.items():
        ordered = sorted(members, key=lambda value: value[0].seed)
        f1s = [float(metrics["validation"]["macro_f1_all_labels"]) for _, metrics in ordered]
        accuracies = [float(metrics["validation"]["accuracy"]) for _, metrics in ordered]
        eces = [float(metrics["validation"]["ece"]) for _, metrics in ordered]
        primary = next(spec for spec, _ in ordered if spec.seed == 2513)
        candidates.append(
            {
                "architecture": architecture,
                "family": primary.family,
                "image_size": image_size,
                "batch_size": primary.batch_size,
                "run_names": [spec.run_name for spec, _ in ordered],
                "primary_run": primary.run_name,
                "mean_validation_macro_f1": float(np.mean(f1s)),
                "mean_validation_accuracy": float(np.mean(accuracies)),
                "mean_validation_ece": float(np.mean(eces)),
                "validation_macro_f1_standard_deviation": float(np.std(f1s, ddof=1)),
            }
        )
    architecture_cost = {"efficientnet_v2_s": 0, "convnext_tiny": 1, "efficientnet_v2_m": 2}
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["mean_validation_macro_f1"]),
            -float(item["mean_validation_accuracy"]),
            float(item["mean_validation_ece"]),
            float(item["validation_macro_f1_standard_deviation"]),
            int(item["image_size"]),
            architecture_cost[str(item["architecture"])],
        ),
    )
    selected_single = ranked[0]
    primary_320 = [item for item in ranked if int(item["image_size"]) == 320]
    strongest_by_family: dict[str, dict[str, Any]] = {}
    for item in primary_320:
        strongest_by_family.setdefault(str(item["family"]), item)
    ensemble_inputs = list(strongest_by_family.values())
    ensemble: dict[str, Any] | None = None
    if len(ensemble_inputs) == 2:
        ensemble = {
            "status": "validation_pending",
            "method": "equal_unscaled_logit_mean_then_temperature_calibration",
            "runs": sorted(str(item["primary_run"]) for item in ensemble_inputs),
            "minimum_macro_f1_gain": 0.0025,
            "maximum_selective_coverage_loss": 0.01,
            "reference_single_run": selected_single["primary_run"],
        }
    report = {
        "schema_version": "1.0",
        "selection_split": "validation",
        "dataset_id": PHASE_B_RELEASE_ID,
        "selection_rule": (
            "highest three-seed mean macro-F1; exact ties: accuracy, lower ECE, "
            "lower seed variance, lower compute cost"
        ),
        "candidates_ranked": ranked,
        "selected_single": selected_single,
        "embedding_gate": {
            "status": "validation_pending",
            "minimum_selective_accuracy_gain": 0.0025,
            "maximum_coverage_loss": 0.01,
        },
        "ensemble": ensemble,
        "locked_test_allowed": False,
    }
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def locked_test_slice_report(
    *,
    paths: list[Path],
    targets: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
    labels: list[str],
    manifest_path: str | Path = "data/manifests/classifier_production_78_v3_20260829.csv",
) -> dict[str, Any]:
    """Join locked predictions to v3 provenance without inventing a source-quality field."""
    manifest = project_path(manifest_path)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    by_path = {row["dataset_image_path"].replace("\\", "/"): row for row in rows}
    relative_paths = [str(path.relative_to(project_path("."))).replace("\\", "/") for path in paths]
    matched = [by_path.get(path) for path in relative_paths]
    if any(row is None for row in matched):
        missing = [path for path, row in zip(relative_paths, matched, strict=True) if row is None]
        raise ValueError(f"Locked-test paths are absent from the v3 manifest: {missing[:3]}")

    def metrics(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"samples": 0, "accuracy": None, "macro_f1": None}
        selected_targets = targets[indices]
        selected_predictions = predictions[indices]
        accuracy = float((selected_targets == selected_predictions).mean())
        observed = sorted(set(selected_targets.tolist()) | set(selected_predictions.tolist()))
        per_label_f1: list[float] = []
        for label in observed:
            target_match = selected_targets == label
            prediction_match = selected_predictions == label
            denominator = int(target_match.sum()) + int(prediction_match.sum())
            per_label_f1.append(
                0.0
                if denominator == 0
                else 2.0 * int((target_match & prediction_match).sum()) / denominator
            )
        return {
            "samples": len(indices),
            "accuracy": accuracy,
            "macro_f1": float(np.mean(per_label_f1)),
        }

    records = [row for row in matched if row is not None]
    source_groups: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        source_groups.setdefault(row["source_dataset"] or "unspecified", []).append(index)
    provenance = {
        source: metrics(indices)
        for source, indices in sorted(source_groups.items())
        if len(indices) >= 10
    }
    tail = [index for indices in source_groups.values() if len(indices) < 10 for index in indices]
    if tail:
        provenance["other_source_dataset_under_10"] = metrics(tail)
    small_crop = [
        index
        for index, row in enumerate(records)
        if min(int(row["crop_width"]), int(row["crop_height"])) <= 96
    ]
    explicit_small_object = [
        index for index, row in enumerate(records) if row.get("small_object", "").casefold() == "true"
    ]
    directional = {
        "no_left_or_right_turn",
        "no_straight_or_left",
        "turn_left_or_right",
        "no_left_turn",
        "no_right_turn",
        "no_straight_ahead",
        "turn_left",
        "turn_right",
    }
    confusions: list[dict[str, Any]] = []
    for expected, predicted in zip(targets, predictions, strict=True):
        expected_name, predicted_name = labels[int(expected)], labels[int(predicted)]
        if (expected_name in directional or predicted_name in directional) and expected_name != predicted_name:
            confusions.append({"expected": expected_name, "predicted": predicted_name})
    return {
        "provenance_source_dataset": provenance,
        "small_crop_min_side_lte_96": metrics(small_crop),
        "explicit_small_object": {
            **metrics(explicit_small_object),
            "note": "No explicit small_object=true records exist in the frozen evaluation split."
            if not explicit_small_object
            else None,
        },
        "directional_or_compound_confusions": confusions,
    }
