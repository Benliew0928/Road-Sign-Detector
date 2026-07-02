from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from roadsign_assist.classification.embedding import EmbeddingGate  # noqa: E402

IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def project_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def softmax(logits: np.ndarray[Any, Any], temperature: float) -> np.ndarray[Any, Any]:
    calibrated = np.asarray(logits, dtype=np.float32) / max(temperature, 0.05)
    shifted = calibrated - calibrated.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def preprocess_image(path: Path, image_size: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    with Image.open(path) as source:
        image = source.convert("RGB")
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    normalized = (array - IMAGE_MEAN) / IMAGE_STD
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)


def load_rows(manifest_path: Path, labels: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    label_set = set(labels)
    rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            expected = row.get("semantic_sign_id", "").strip()
            image_value = row.get("path", "").strip()
            image_path = project_path(image_value)
            reason = ""
            if not expected:
                reason = "missing semantic_sign_id"
            elif expected not in label_set:
                reason = f"label {expected!r} is not in classifier vocabulary"
            elif not image_value:
                reason = "missing path"
            elif not image_path.is_file():
                reason = f"missing image: {image_value}"
            if reason:
                skipped.append({**row, "skip_reason": reason})
                continue
            rows.append(row)
    return rows, skipped


def macro_f1(targets: list[str], predictions: list[str]) -> float:
    observed = sorted(set(targets))
    if not observed:
        return 0.0
    scores: list[float] = []
    for label in observed:
        true_positive = sum(
            1 for target, prediction in zip(targets, predictions, strict=True)
            if target == label and prediction == label
        )
        false_positive = sum(
            1 for target, prediction in zip(targets, predictions, strict=True)
            if target != label and prediction == label
        )
        false_negative = sum(
            1 for target, prediction in zip(targets, predictions, strict=True)
            if target == label and prediction != label
        )
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def confusion_matrix(labels: list[str], targets: list[str], predictions: list[str]) -> list[list[int]]:
    index = {label: offset for offset, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for target, prediction in zip(targets, predictions, strict=True):
        if target in index and prediction in index:
            matrix[index[target]][index[prediction]] += 1
    return matrix


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch

        _ = torch.__version__
    except ImportError:
        pass
    import onnxruntime as ort

    model_path = project_path(args.model)
    labels_path = project_path(args.labels)
    calibration_path = project_path(args.calibration) if args.calibration else None
    manifest_path = project_path(args.manifest)
    labels = [str(value) for value in json.loads(labels_path.read_text(encoding="utf-8"))]
    calibration: dict[str, Any] = {}
    if calibration_path is not None and calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    temperature = float(calibration.get("temperature", 1.0))
    confidence_threshold = (
        float(args.confidence_threshold)
        if args.confidence_threshold is not None
        else float(calibration.get("confidence_threshold", 0.72))
    )
    embedding_gate = None
    embedding_payload = calibration.get("embedding_gate")
    if isinstance(embedding_payload, dict):
        embedding_gate = EmbeddingGate.from_payload(embedding_payload)

    rows, skipped = load_rows(manifest_path, labels)
    if not rows:
        raise ValueError(f"No evaluable rows found in {manifest_path}")

    installed = set(ort.get_available_providers())
    requested = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in installed
    ]
    session = ort.InferenceSession(str(model_path), providers=requested or ["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset in range(0, len(rows), args.batch):
        batch_rows = rows[offset : offset + args.batch]
        batch = np.stack(
            [
                preprocess_image(project_path(row["path"]), args.image_size)
                for row in batch_rows
            ]
        )
        outputs = session.run(None, {input_name: batch})
        probabilities = softmax(np.asarray(outputs[0], dtype=np.float32), temperature)
        embeddings = np.asarray(outputs[1], dtype=np.float32) if len(outputs) > 1 else None
        for row_index, row in enumerate(batch_rows):
            order = np.argsort(probabilities[row_index])[::-1][:5]
            top_k = [
                {
                    "label": labels[int(index)],
                    "confidence": round(float(probabilities[row_index, index]), 6),
                }
                for index in order
            ]
            predicted = top_k[0]["label"]
            confidence = float(top_k[0]["confidence"])
            accepted = confidence >= confidence_threshold
            nearest_prototype: str | None = None
            embedding_distance: float | None = None
            rejection_reasons: list[str] = []
            if not accepted:
                rejection_reasons.append("confidence")
            if embedding_gate is not None:
                if embeddings is None:
                    raise RuntimeError(
                        "Calibration has an embedding gate but model has no embedding output"
                    )
                decision = embedding_gate.decide(embeddings[row_index], str(predicted))
                nearest_prototype = decision.nearest_label
                embedding_distance = decision.distance
                rejection_reasons.extend(decision.reasons)
                accepted = accepted and decision.accepted
            expected = row["semantic_sign_id"].strip()
            output_label = str(predicted) if accepted else "unknown_sign"
            predictions.append(
                {
                    "external_test_id": row.get("external_test_id", ""),
                    "path": row["path"],
                    "expected": expected,
                    "predicted_raw": predicted,
                    "confidence": round(confidence, 6),
                    "accepted": accepted,
                    "output_label": output_label,
                    "correct_raw": str(predicted) == expected,
                    "correct_accepted": accepted and str(predicted) == expected,
                    "nearest_prototype": nearest_prototype or "",
                    "embedding_distance": (
                        round(embedding_distance, 6) if embedding_distance is not None else ""
                    ),
                    "rejection_reasons": "|".join(rejection_reasons),
                    "top5": json.dumps(top_k, separators=(",", ":")),
                }
            )
    elapsed = time.perf_counter() - started

    targets = [str(row["expected"]) for row in predictions]
    raw_predictions = [str(row["predicted_raw"]) for row in predictions]
    raw_correct = [bool(row["correct_raw"]) for row in predictions]
    accepted_correct = [bool(row["correct_accepted"]) for row in predictions]
    accepted_mask = [bool(row["accepted"]) for row in predictions]
    class_totals = Counter(targets)
    class_correct = Counter(
        target for target, correct in zip(targets, raw_correct, strict=True) if correct
    )
    class_accepted_correct = Counter(
        target
        for target, correct in zip(targets, accepted_correct, strict=True)
        if correct
    )
    class_coverage: dict[str, int] = defaultdict(int)
    for target, accepted in zip(targets, accepted_mask, strict=True):
        if accepted:
            class_coverage[target] += 1

    accepted_count = sum(accepted_mask)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "model": project_rel(model_path),
        "labels": project_rel(labels_path),
        "calibration": project_rel(calibration_path) if calibration_path else None,
        "manifest": project_rel(manifest_path),
        "providers": list(session.get_providers()),
        "image_size": args.image_size,
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "embedding_gate_enabled": embedding_gate is not None,
        "samples": len(predictions),
        "skipped_rows": skipped,
        "observed_labels": len(class_totals),
        "raw_accuracy": float(np.mean(raw_correct)),
        "raw_macro_f1_observed": macro_f1(targets, raw_predictions),
        "accepted_samples": accepted_count,
        "selective_coverage": accepted_count / len(predictions),
        "selective_accuracy": (
            sum(accepted_correct) / accepted_count if accepted_count else None
        ),
        "accepted_correct_rate": sum(accepted_correct) / len(predictions),
        "rejected_correct_samples": sum(
            correct and not accepted
            for correct, accepted in zip(raw_correct, accepted_mask, strict=True)
        ),
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(labels, targets, raw_predictions),
        "per_class": {
            label: {
                "samples": class_totals[label],
                "raw_correct": class_correct[label],
                "raw_accuracy": class_correct[label] / class_totals[label],
                "accepted_samples": class_coverage[label],
                "accepted_correct": class_accepted_correct[label],
                "accepted_correct_rate": class_accepted_correct[label] / class_totals[label],
            }
            for label in sorted(class_totals)
        },
        "latency": {
            "total_seconds": elapsed,
            "mean_ms_per_image": elapsed * 1000.0 / len(predictions),
            "batch_size": args.batch,
        },
    }

    predictions_output = project_path(args.predictions_output)
    report_output = project_path(args.report_output)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "external_test_id",
                "path",
                "expected",
                "predicted_raw",
                "confidence",
                "accepted",
                "output_label",
                "correct_raw",
                "correct_accepted",
                "nearest_prototype",
                "embedding_distance",
                "rejection_reasons",
                "top5",
            ],
        )
        writer.writeheader()
        writer.writerows(predictions)
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Stage G classifier on held-out crops.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--calibration")
    parser.add_argument(
        "--manifest",
        default="data/manifests/assignment_external_test.csv",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument(
        "--predictions-output",
        default="outputs/evaluation/stage_g_classifier/assignment_predictions.csv",
    )
    parser.add_argument(
        "--report-output",
        default="outputs/evaluation/stage_g_classifier/assignment_report.json",
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    report = evaluate(build_parser().parse_args())
    print(
        "Stage G classifier evaluation complete: "
        f"samples={report['samples']}, "
        f"raw_accuracy={report['raw_accuracy']:.3f}, "
        f"accepted_correct_rate={report['accepted_correct_rate']:.3f}, "
        f"coverage={report['selective_coverage']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
