from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, cast

from roadsign_assist.diagnostics import diagnostics_json
from roadsign_assist.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roadsign-assist",
        description="Offline Malaysian road-sign intelligence tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check environment, hardware, and official inputs.")
    subparsers.add_parser("inventory-official", help="Build official input manifests.")
    subparsers.add_parser("validate-catalogue", help="Validate the Malaysian sign catalogue.")
    subparsers.add_parser("build-splits", help="Create grouped leakage-safe dataset splits.")
    final_classifier_release = subparsers.add_parser(
        "freeze-final-classifier-release",
        help="Audit contributor intake and freeze the train-only classifier v3 release.",
    )
    final_classifier_release.add_argument(
        "--release-id",
        default="classifier_production_78_v3_20260829",
    )
    final_classifier_release.add_argument("--overwrite", action="store_true")
    final_classifier_release.add_argument(
        "--disallow-internal-academic-exceptions",
        action="store_true",
        help="Exclude rows that need the explicitly approved internal-use exception.",
    )
    final_classifier_release.add_argument(
        "--allow-must-have-coverage-exception",
        action="store_true",
        help="Record an explicit owner waiver for any remaining must-have coverage gap.",
    )
    final_classifier_release.add_argument(
        "--coverage-exception-note",
        default="",
        help="Owner-approved rationale retained in release metadata when coverage is waived.",
    )
    final_detector_release = subparsers.add_parser(
        "prepare-final-detector-release",
        help="Quarantine, audit, review, and freeze the Phase C assignment-only detector release.",
    )
    final_detector_release.add_argument("--overwrite", action="store_true")
    final_detector_release.add_argument(
        "--bundle-root",
        default=r"C:\MiniProject-data-audit\Ultimate_Datasets_v1_DVC_Transfer_96MiB",
    )
    final_detector_release.add_argument(
        "--quarantine-root",
        default=r"C:\MiniProject-data-audit\_validation\phase_c_detector_assignment_v1",
    )
    final_detector_release.add_argument(
        "--negative-root",
        default="data/raw/phase_c_no_sign_frames",
        help="Directory of full-frame no-sign images accompanied by review decisions.",
    )
    final_detector_release.add_argument(
        "--reuse-verified-quarantine",
        action="store_true",
        help="Revalidate and reuse an existing verified quarantine without deleting or re-extracting it.",
    )
    negative_candidates = subparsers.add_parser(
        "collect-phase-c-negative-candidates",
        help="Collect the 120 annotation-screened, owner-review no-sign frames for Phase C.",
    )
    negative_candidates.add_argument("--overwrite", action="store_true")
    subparsers.add_parser(
        "approve-phase-c-negative-candidates",
        help="Record the owner's explicit acceptance of all 120 Phase C no-sign candidates.",
    )
    negative_audit = subparsers.add_parser(
        "audit-phase-c-approved-negatives",
        help="Check approved no-sign candidates against quarantined teammate positives.",
    )
    negative_audit.add_argument(
        "--bundle-root",
        default=r"C:\MiniProject-data-audit\Ultimate_Datasets_v1_DVC_Transfer_96MiB",
    )
    negative_audit.add_argument(
        "--quarantine-root",
        default=r"C:\MiniProject-data-audit\_validation\phase_c_detector_assignment_v1",
    )
    negative_audit.add_argument("--negative-root", default="data/raw/phase_c_no_sign_frames")
    teammate_review = subparsers.add_parser(
        "render-phase-c-teammate-review",
        help="Render labelled overlay sheets for every Phase C teammate review row.",
    )
    teammate_review.add_argument("--overwrite", action="store_true")
    emtd_review = subparsers.add_parser(
        "render-phase-c-emtd-review",
        help="Render labelled overlay sheets for every Phase C EMTD merge-eligibility row.",
    )
    emtd_review.add_argument("--overwrite", action="store_true")
    emtd_decision = subparsers.add_parser(
        "record-phase-c-emtd-review",
        help="Record an owner decision for every reviewed Phase C EMTD merge-eligibility row.",
    )
    emtd_decision.add_argument("--decision", choices=("accept", "reject"), required=True)
    emtd_decision.add_argument("--note", required=True)
    teammate_layout_review = subparsers.add_parser(
        "record-phase-c-teammate-layout-review",
        help="Record an owner decision for explicitly named, fully reviewed Phase C teammate layouts.",
    )
    teammate_layout_review.add_argument(
        "--layout-root-id",
        action="append",
        required=True,
        help="Canonical layout root ID to update; repeat for each reviewed layout.",
    )
    teammate_layout_review.add_argument("--decision", choices=("accept", "reject"), required=True)
    teammate_layout_review.add_argument("--note", required=True)
    subparsers.add_parser(
        "coursework-contact-sheets",
        help="Create full and representative coursework review sheets.",
    )
    subparsers.add_parser(
        "coursework-review-manifest",
        help="Expand the draft coursework ID mapping to every official image.",
    )
    subparsers.add_parser(
        "emtd-contact-sheet",
        help="Create an EMTD class crop review sheet from downloaded images.",
    )
    subparsers.add_parser(
        "import-emtd",
        help="Validate EMTD boxes and build leakage-safe model datasets.",
    )
    subparsers.add_parser(
        "inventory-ocr-assets",
        help="Hash and record the local offline PaddleOCR model assets.",
    )
    subparsers.add_parser(
        "verify-ocr-assets",
        help="Verify local PaddleOCR models against their frozen manifest.",
    )
    masks = subparsers.add_parser(
        "generate-emtd-masks",
        help="Generate SAM 2.1 box-prompted draft masks for manual review.",
    )
    masks.add_argument("--model", default="models/pretrained/sam2.1_s.pt")
    masks.add_argument("--device", default="0")
    masks.add_argument("--limit", type=int)
    subparsers.add_parser(
        "create-ocr-smoke-set",
        help="Create a labelled synthetic multilingual OCR pipeline smoke set.",
    )
    ocr_evaluate = subparsers.add_parser(
        "evaluate-ocr-smoke",
        help="Run frozen offline OCR models on the synthetic smoke set.",
    )
    ocr_evaluate.add_argument(
        "--manifest",
        default="data/processed/ocr_smoke/manifest.json",
    )
    subparsers.add_parser(
        "emtd-mask-review-sheets",
        help="Render contact sheets for accepted and failed SAM mask drafts.",
    )
    subparsers.add_parser(
        "compare-classifiers",
        help="Compare completed EMTD classifier runs on held-out test metrics.",
    )
    coursework_evaluation = subparsers.add_parser(
        "evaluate-coursework",
        help="Run all 84 official images through a selected inference profile.",
    )
    coursework_evaluation.add_argument(
        "--config",
        default="configs/inference/default.yaml",
    )
    coursework_evaluation.add_argument(
        "--output",
        default="outputs/evaluation/coursework",
    )

    baseline = subparsers.add_parser("baseline-batch", help="Run the classical baseline.")
    baseline.add_argument("--input", required=True)
    baseline.add_argument("--output", default="outputs/baseline")

    baseline_benchmark = subparsers.add_parser(
        "baseline-benchmark",
        help="Compare SVM and Random Forest classifiers on frozen crop splits.",
    )
    baseline_benchmark.add_argument(
        "--data",
        default="data/processed/emtd_classification",
    )
    baseline_benchmark.add_argument(
        "--output",
        default="outputs/evaluation/baseline_classifiers",
    )
    baseline_benchmark.add_argument("--experimental", action="store_true")

    reset_audit = subparsers.add_parser(
        "verify-reset",
        help="Verify the external official backup against restored inputs.",
    )
    reset_audit.add_argument(
        "--backup",
        default=r"C:\MiniProject_OfficialBackup",
    )

    detector_benchmark = subparsers.add_parser(
        "benchmark-detector",
        help="Benchmark an exported segmentation model on a frozen dataset split.",
    )
    detector_benchmark.add_argument("--model", required=True)
    detector_benchmark.add_argument(
        "--data",
        default="data/processed/emtd_segmentation/data.yaml",
    )
    detector_benchmark.add_argument("--output", required=True)
    detector_benchmark.add_argument("--split", default="test")
    detector_benchmark.add_argument("--imgsz", type=int, default=512)
    detector_benchmark.add_argument("--confidence", type=float, default=0.25)
    detector_benchmark.add_argument("--device", default="cpu")
    detector_benchmark.add_argument("--limit", type=int)
    detector_benchmark.add_argument("--task", choices=("detect", "segment"))

    detector_tuning = subparsers.add_parser(
        "tune-detector-thresholds",
        help="Evaluate segmentation confidence thresholds on validation data.",
    )
    detector_tuning.add_argument("--model", required=True)
    detector_tuning.add_argument(
        "--data",
        default="data/processed/emtd_segmentation/data.yaml",
    )
    detector_tuning.add_argument("--output", required=True)
    detector_tuning.add_argument("--imgsz", type=int, default=512)
    detector_tuning.add_argument("--device", default="0")
    detector_tuning.add_argument("--task", choices=("detect", "segment"))
    detector_tuning.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.25, 0.35, 0.50],
    )

    classifier_safety = subparsers.add_parser(
        "evaluate-classifier-safety",
        help="Report safety-critical class recall from a frozen confusion matrix.",
    )
    classifier_safety.add_argument("--metrics", required=True)
    classifier_safety.add_argument("--labels", required=True)
    classifier_safety.add_argument("--output", required=True)

    tracking_motion = subparsers.add_parser(
        "evaluate-tracking-motion",
        help="Run synthetic P11 tracking, camera-motion, blur, and cooldown checks.",
    )
    tracking_motion.add_argument(
        "--output",
        default="outputs/evaluation/tracking_motion",
    )
    tracking_motion.add_argument(
        "--report",
        default="docs/P11_TRACKING_MOTION_REPORT.md",
    )

    detector_slices = subparsers.add_parser(
        "evaluate-detector-slices",
        help="Measure overall and normalized-area small-sign box recall.",
    )
    detector_slices.add_argument("--model", required=True)
    detector_slices.add_argument(
        "--data",
        default="data/processed/emtd_segmentation/data.yaml",
    )
    detector_slices.add_argument("--output", required=True)
    detector_slices.add_argument("--split", default="test")
    detector_slices.add_argument("--imgsz", type=int, default=640)
    detector_slices.add_argument("--confidence", type=float, default=0.25)
    detector_slices.add_argument("--device", default="0")
    detector_slices.add_argument("--match-iou", type=float, default=0.50)
    detector_slices.add_argument("--small-area-ratio", type=float, default=0.01)
    detector_slices.add_argument("--task", choices=("detect", "segment"))

    serve = subparsers.add_parser("serve", help="Run the local FastAPI application.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--config",
        default="configs/inference/default.yaml",
    )
    serve.add_argument("--ssl-certfile", default=None)
    serve.add_argument("--ssl-keyfile", default=None)
    serve.add_argument("--public-host", default=None)

    detector = subparsers.add_parser(
        "train-detector",
        help="Train an experimental or approved Ultralytics road-sign detector.",
    )
    detector.add_argument(
        "--data",
        default="data/processed/emtd_detection/data.yaml",
    )
    detector.add_argument("--model", default="yolo26n.pt")
    detector.add_argument("--task", choices=("detect", "segment"), default="detect")
    detector.add_argument("--epochs", type=int, default=100)
    detector.add_argument("--batch", type=int, default=16)
    detector.add_argument("--imgsz", type=int, default=640)
    detector.add_argument("--device", default="0")
    detector.add_argument("--name", default="malaysia_sign_detector")
    detector.add_argument("--experimental", action="store_true")

    phase_d_train = subparsers.add_parser(
        "train-phase-d-detector",
        help="Train or resume one immutable candidate in the Phase-D detector matrix.",
    )
    phase_d_train.add_argument("--candidate", required=True)
    phase_d_train.add_argument("--resume", action="store_true")
    phase_d_train.add_argument("--device", default="0")

    phase_d_status = subparsers.add_parser(
        "phase-d-detector-status",
        help="Show durable progress for the fixed Phase-D detector matrix.",
    )
    phase_d_status.add_argument("--json", action="store_true")

    phase_d_select = subparsers.add_parser(
        "select-phase-d-detector",
        help="Evaluate and select the Phase-D candidate using validation data only.",
    )
    phase_d_select.add_argument("--device", default="0")
    phase_d_select.add_argument("--output", default="outputs/training/phase_d_selection.json")

    phase_d_evaluate = subparsers.add_parser(
        "evaluate-phase-d-selected-detector",
        help="Run the guarded locked test, ONNX parity, and benchmarks for the selection.",
    )
    phase_d_evaluate.add_argument("--device", default="0")

    phase_e_prepare = subparsers.add_parser(
        "prepare-phase-e-benchmark",
        help="Prepare the prediction-free Phase-E owner-review benchmark.",
    )
    phase_e_prepare.add_argument("--overwrite", action="store_true")

    subparsers.add_parser(
        "validate-phase-e-review",
        help="Validate and freeze a completed owner review for Phase E.",
    )

    phase_e_evaluate = subparsers.add_parser(
        "evaluate-phase-e-pipeline",
        help="Evaluate one locked Phase-E runtime profile.",
    )
    phase_e_evaluate.add_argument("--profile", choices=("gpu", "cpu"), required=True)

    subparsers.add_parser(
        "finalize-phase-e-gate",
        help="Compare both profiles and emit the immutable Gate-E decision.",
    )

    phase_e_status = subparsers.add_parser(
        "phase-e-status",
        help="Show Phase-E benchmark, evaluation, and promotion status.",
    )
    phase_e_status.add_argument("--json", action="store_true")

    phase_e_promote = subparsers.add_parser(
        "promote-phase-e-runtime",
        help="Promote the Phase-E bundle only after every Gate-E check passes.",
    )
    phase_e_promote.add_argument("--internal-only", action="store_true")

    phase_e_rollback = subparsers.add_parser(
        "rollback-phase-e-runtime",
        help="Restore a complete runtime backup created by Phase-E promotion.",
    )
    phase_e_rollback.add_argument("--backup-id", required=True)

    finalize_detector = subparsers.add_parser(
        "finalize-detector",
        help="Evaluate a detector checkpoint, export ONNX, and verify runtime parity.",
    )
    finalize_detector.add_argument("--checkpoint", required=True)
    finalize_detector.add_argument(
        "--data",
        default="data/processed/emtd_detection/data.yaml",
    )
    finalize_detector.add_argument(
        "--task",
        choices=("detect", "segment"),
        default="detect",
    )
    finalize_detector.add_argument("--imgsz", type=int, default=640)
    finalize_detector.add_argument("--device", default="0")
    finalize_detector.add_argument("--name", default="sign_detector")
    finalize_detector.add_argument("--experimental", action="store_true")

    classifier = subparsers.add_parser(
        "train-classifier",
        help="Train and export a crop classifier from a prepared folder dataset.",
    )
    classifier.add_argument(
        "--data",
        default="data/processed/classifier_no_controlled_variants_20260812",
    )
    classifier.add_argument(
        "--architecture",
        choices=(
            "mobilenet_v3_large",
            "efficientnet_v2_s",
            "efficientnet_v2_m",
            "convnext_tiny",
        ),
        default="mobilenet_v3_large",
    )
    classifier.add_argument("--epochs", type=int, default=40)
    classifier.add_argument("--batch", type=int, default=32)
    classifier.add_argument("--imgsz", type=int, default=224)
    classifier.add_argument("--learning-rate", type=float, default=3e-4)
    classifier.add_argument("--weight-decay", type=float, default=1e-4)
    classifier.add_argument("--label-smoothing", type=float, default=0.05)
    classifier.add_argument("--seed", type=int, default=2513)
    classifier.add_argument("--workers", type=int, default=4)
    classifier.add_argument("--confidence-threshold", type=float, default=0.72)
    classifier.add_argument("--tune-confidence-threshold", action="store_true")
    classifier.add_argument("--target-selective-accuracy", type=float, default=0.80)
    classifier.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the locked test split after configuration selection.",
    )
    classifier.add_argument("--overwrite", action="store_true")
    classifier.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted matching run from its latest epoch checkpoint.",
    )
    classifier.add_argument("--device", default="auto")
    classifier.add_argument("--name", default="malaysia_sign_classifier")
    classifier.add_argument("--experimental", action="store_true")

    promote_classifier = subparsers.add_parser(
        "promote-classifier",
        help="Promote a reviewed, test-evaluated classifier candidate into the runtime bundle.",
    )
    promote_classifier.add_argument("--run", required=True)
    promote_classifier.add_argument("--overwrite", action="store_true")
    promote_classifier.add_argument(
        "--internal-only",
        action="store_true",
        help="Required to locally promote the internal-academic v3 release; never publishes DVC.",
    )

    rollback_classifier = subparsers.add_parser(
        "rollback-classifier-runtime",
        help="Restore the timestamped local runtime backup created by internal promotion.",
    )
    rollback_classifier.add_argument("--backup-id", required=True)

    phase_b_status = subparsers.add_parser(
        "phase-b-classifier-status",
        help="Show the durable status of the predeclared Phase-B classifier matrix.",
    )
    phase_b_status.add_argument("--json", action="store_true")
    phase_b_select = subparsers.add_parser(
        "select-phase-b-classifier",
        help="Rank completed Phase-B runs using validation data only.",
    )
    phase_b_select.add_argument("--output", default="outputs/training/phase_b_v3_selection.json")
    phase_b_ensemble = subparsers.add_parser(
        "prepare-phase-b-ensemble",
        help="Build a validation-only equal-logit ensemble from the selected 320 px runs.",
    )
    phase_b_ensemble.add_argument("--runs", nargs=2, required=True)
    phase_b_ensemble.add_argument("--name", required=True)
    phase_b_ensemble.add_argument("--target-selective-accuracy", type=float, default=0.98)
    phase_b_ensemble_test = subparsers.add_parser(
        "evaluate-phase-b-ensemble",
        help="Run the locked test once for a validation-qualified Phase-B ensemble.",
    )
    phase_b_ensemble_test.add_argument("--name", required=True)
    phase_b_ensemble_test.add_argument("--overwrite", action="store_true")
    phase_b_embedding = subparsers.add_parser(
        "prepare-phase-b-embedding-gate",
        help="Export and qualify a prototype gate on validation only for one selected Phase-B run.",
    )
    phase_b_embedding.add_argument("--run", required=True)
    phase_b_embedding.add_argument("--device", default="auto")
    phase_b_embedding.add_argument("--batch", type=int, default=32)
    phase_b_embedding.add_argument("--workers", type=int, default=0)

    evaluate_classifier = subparsers.add_parser(
        "evaluate-classifier-candidate",
        help="Tune the threshold on validation and run the selected candidate on locked test.",
    )
    evaluate_classifier.add_argument("--run", required=True)
    evaluate_classifier.add_argument("--target-selective-accuracy", type=float, default=0.98)
    evaluate_classifier.add_argument("--fixed-threshold", action="store_true")
    evaluate_classifier.add_argument("--overwrite", action="store_true")

    embedding_classifier = subparsers.add_parser(
        "finalize-classifier-embeddings",
        help="Export classifier logits plus embeddings and calibrate prototype rejection.",
    )
    embedding_classifier.add_argument("--checkpoint", required=True)
    embedding_classifier.add_argument(
        "--data",
        default="data/processed/classifier_no_controlled_variants_20260812",
    )
    embedding_classifier.add_argument("--model-output", required=True)
    embedding_classifier.add_argument("--calibration-output", required=True)
    embedding_classifier.add_argument("--report-output", required=True)
    embedding_classifier.add_argument("--device", default="auto")
    embedding_classifier.add_argument("--batch", type=int, default=64)
    embedding_classifier.add_argument("--workers", type=int, default=0)
    embedding_classifier.add_argument(
        "--retention-quantile",
        type=float,
        default=0.95,
    )
    embedding_classifier.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly evaluate the locked test split after a validation-only gate decision.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    configure_logging()
    args = build_parser().parse_args(argv)

    if args.command == "doctor":
        print(diagnostics_json())
        return 0

    if args.command == "inventory-official":
        from roadsign_assist.datasets.official import write_official_manifests

        write_official_manifests()
        return 0

    if args.command == "validate-catalogue":
        from roadsign_assist.catalogue.repository import load_default_catalogue

        catalogue = load_default_catalogue()
        print(f"Validated {len(catalogue)} catalogue entries.")
        return 0

    if args.command == "build-splits":
        from roadsign_assist.datasets.split import build_default_splits

        build_default_splits()
        return 0

    if args.command == "freeze-final-classifier-release":
        from roadsign_assist.datasets.final_classifier_release import (
            FinalClassifierReleaseConfig,
            build_final_classifier_release,
        )
        from roadsign_assist.paths import PROJECT_ROOT

        audit = build_final_classifier_release(
            FinalClassifierReleaseConfig(
                PROJECT_ROOT,
                release_id=args.release_id,
                allow_internal_academic_exceptions=not args.disallow_internal_academic_exceptions,
                allow_must_have_coverage_exception=args.allow_must_have_coverage_exception,
                coverage_exception_note=args.coverage_exception_note,
            ),
            overwrite=args.overwrite,
        )
        print(
            "Final classifier release complete: "
            f"{audit['dataset_id']} ({audit['counts']['retained_contributor_rows']} "
            "retained contributor rows)"
        )
        return 0

    if args.command == "prepare-final-detector-release":
        from pathlib import Path

        from roadsign_assist.datasets.final_detector_release import (
            DetectorReleaseConfig,
            ReviewRequiredError,
            build_final_detector_release,
        )
        from roadsign_assist.paths import PROJECT_ROOT

        try:
            audit = build_final_detector_release(
                DetectorReleaseConfig(
                    project_root=PROJECT_ROOT,
                    bundle_root=Path(args.bundle_root),
                    quarantine_root=Path(args.quarantine_root),
                    negative_root=Path(args.negative_root),
                    reuse_verified_quarantine=args.reuse_verified_quarantine,
                ),
                overwrite=args.overwrite,
            )
        except ReviewRequiredError as exc:
            print(str(exc))
            return 2
        print(
            "Final detector release complete: "
            f"{audit['dataset_id']} ({audit['counts']['positive_retained']} positive, "
            f"{audit['counts']['negative_retained']} negative test images)"
        )
        return 0

    if args.command == "collect-phase-c-negative-candidates":
        from roadsign_assist.datasets.phase_c_negative_candidates import (
            collect_phase_c_negative_candidates,
        )

        result = collect_phase_c_negative_candidates(overwrite=args.overwrite)
        print(
            "Phase C no-sign candidates ready for owner review: "
            f"{result['candidate_count']} images at {result['review_queue']}"
        )
        return 0

    if args.command == "approve-phase-c-negative-candidates":
        from roadsign_assist.datasets.phase_c_negative_candidates import (
            approve_phase_c_negative_candidates,
        )

        result = approve_phase_c_negative_candidates()
        print(f"Recorded {result['accepted_count']} approved Phase C no-sign candidates")
        return 0

    if args.command == "audit-phase-c-approved-negatives":
        from pathlib import Path

        from roadsign_assist.datasets.final_detector_release import (
            DetectorReleaseConfig,
            audit_approved_negatives_against_teammate,
        )
        from roadsign_assist.paths import PROJECT_ROOT

        report = audit_approved_negatives_against_teammate(
            DetectorReleaseConfig(
                project_root=PROJECT_ROOT,
                bundle_root=Path(args.bundle_root),
                quarantine_root=Path(args.quarantine_root),
                negative_root=Path(args.negative_root),
            )
        )
        print(
            "Approved no-sign audit passed: "
            f"{report['negative_candidate_count']} negatives versus "
            f"{report['teammate_positive_candidate_count']} teammate positives"
        )
        return 0

    if args.command == "record-phase-c-teammate-layout-review":
        from roadsign_assist.datasets.final_detector_release import record_teammate_layout_review
        from roadsign_assist.paths import PROJECT_ROOT

        review_queue = (
            PROJECT_ROOT
            / "data/manifests/detector_production_assignment_v1_20260829_reviews"
            / "teammate_visual_review.csv"
        )
        result = record_teammate_layout_review(
            review_queue,
            layout_root_ids=args.layout_root_id,
            decision=args.decision,
            reviewer_notes=args.note,
        )
        print(
            "Recorded teammate layout review: "
            f"{result['updated_count']} {result['decision']} rows; "
            f"{result['remaining_pending_count']} still pending"
        )
        return 0

    if args.command == "record-phase-c-emtd-review":
        from roadsign_assist.datasets.final_detector_release import record_emtd_review
        from roadsign_assist.paths import PROJECT_ROOT

        review_queue = (
            PROJECT_ROOT
            / "data/manifests/detector_production_assignment_v1_20260829_reviews"
            / "emtd_box_review.csv"
        )
        result = record_emtd_review(
            review_queue,
            decision=args.decision,
            reviewer_notes=args.note,
        )
        print(
            "Recorded EMTD review: "
            f"{result['updated_count']} {result['decision']} rows; "
            f"{result['remaining_pending_count']} still pending"
        )
        return 0

    if args.command == "render-phase-c-teammate-review":
        from roadsign_assist.datasets.detector_review_package import (
            build_teammate_detector_review_package,
        )

        summary = build_teammate_detector_review_package(overwrite=args.overwrite)
        print(
            "Phase C teammate overlay review package ready: "
            f"{summary['indexed_rows']} rows in {summary['sheet_count']} sheets"
        )
        return 0

    if args.command == "render-phase-c-emtd-review":
        from roadsign_assist.datasets.detector_review_package import (
            build_emtd_detector_review_package,
        )

        summary = build_emtd_detector_review_package(overwrite=args.overwrite)
        print(
            "Phase C EMTD overlay review package ready: "
            f"{summary['indexed_rows']} rows in {summary['sheet_count']} sheets"
        )
        return 0

    if args.command == "coursework-contact-sheets":
        from roadsign_assist.datasets.contact_sheet import (
            coursework_tiles,
            render_contact_sheet,
        )

        full = render_contact_sheet(
            coursework_tiles(),
            "outputs/review/coursework_all_images.jpg",
            columns=7,
        )
        representatives = render_contact_sheet(
            coursework_tiles(representatives_only=True),
            "outputs/review/coursework_class_representatives.jpg",
            columns=8,
        )
        print(f"Wrote {full}")
        print(f"Wrote {representatives}")
        return 0

    if args.command == "coursework-review-manifest":
        from roadsign_assist.datasets.official import write_coursework_review_manifest

        output = write_coursework_review_manifest()
        print(f"Wrote {output}")
        return 0

    if args.command == "emtd-contact-sheet":
        from roadsign_assist.datasets.contact_sheet import (
            emtd_class_tiles,
            render_contact_sheet,
        )

        tiles = emtd_class_tiles()
        output = render_contact_sheet(
            tiles,
            "outputs/review/emtd_class_representatives.jpg",
            columns=8,
        )
        print(f"Wrote {len(tiles)} classes to {output}")
        return 0

    if args.command == "import-emtd":
        from roadsign_assist.datasets.emtd_import import import_emtd_subset

        stats = import_emtd_subset()
        print(f"Imported EMTD subset: {stats}")
        return 0

    if args.command == "inventory-ocr-assets":
        from roadsign_assist.ocr.assets import write_ocr_asset_manifest

        output = write_ocr_asset_manifest()
        print(f"Wrote {output}")
        return 0

    if args.command == "verify-ocr-assets":
        from roadsign_assist.ocr.assets import verify_ocr_assets

        manifest = verify_ocr_assets()
        print(
            "Verified offline OCR assets: "
            f"{', '.join(manifest['models'])} "
            f"(PaddleOCR {manifest['paddleocr_version']})"
        )
        return 0

    if args.command == "generate-emtd-masks":
        from roadsign_assist.datasets.emtd_masks import (
            MaskGenerationConfig,
            generate_emtd_masks,
        )

        report = generate_emtd_masks(
            MaskGenerationConfig(
                model=args.model,
                device=args.device,
                limit=args.limit,
            )
        )
        print(f"Generated EMTD draft masks: {report}")
        return 1 if report["failed_images"] else 0

    if args.command == "create-ocr-smoke-set":
        from roadsign_assist.datasets.ocr_smoke import (
            create_synthetic_ocr_smoke_set,
        )

        output = create_synthetic_ocr_smoke_set()
        print(f"Wrote {output}")
        return 0

    if args.command == "evaluate-ocr-smoke":
        from roadsign_assist.evaluation.ocr import evaluate_ocr_manifest

        report = evaluate_ocr_manifest(args.manifest)
        print(
            "OCR synthetic smoke complete: "
            f"exact={report['exact_match_rate']:.3f}, "
            f"CER={report['mean_cer']:.3f}, "
            f"warm_mean={report['warm_mean_latency_ms']:.1f} ms"
        )
        return 0

    if args.command == "emtd-mask-review-sheets":
        from roadsign_assist.datasets.contact_sheet import (
            directory_tiles,
            render_contact_sheet,
        )

        accepted = directory_tiles("data/processed/emtd_segmentation/review/accepted")
        failed = directory_tiles("data/processed/emtd_segmentation/review/failed")
        accepted_output = render_contact_sheet(
            accepted,
            "outputs/review/emtd_masks_accepted.jpg",
            columns=5,
            tile_width=260,
            tile_height=220,
        )
        print(f"Wrote {accepted_output}")
        if failed:
            failed_output = render_contact_sheet(
                failed,
                "outputs/review/emtd_masks_failed.jpg",
                columns=min(3, len(failed)),
                tile_width=360,
                tile_height=300,
            )
            print(f"Wrote {failed_output}")
        return 0

    if args.command == "compare-classifiers":
        from roadsign_assist.evaluation.classifier_comparison import (
            compare_classifier_runs,
        )

        report = compare_classifier_runs()
        print(f"Compared {len(report['runs'])} classifier runs; best={report['best_run']}")
        return 0

    if args.command == "evaluate-coursework":
        from roadsign_assist.evaluation.coursework import (
            evaluate_coursework_images,
        )

        report = evaluate_coursework_images(args.config, args.output)
        print(
            "Coursework evaluation complete: "
            f"{report['completed']}/{report['images']}, "
            f"mean={report['mean_runtime_ms']:.1f} ms, "
            f"max={report['maximum_runtime_ms']:.1f} ms"
        )
        return 0

    if args.command == "baseline-batch":
        from roadsign_assist.baseline.batch import run_baseline_batch

        run_baseline_batch(args.input, args.output)
        return 0

    if args.command == "baseline-benchmark":
        from roadsign_assist.baseline.benchmark import (
            run_baseline_classifier_benchmark,
        )

        report = run_baseline_classifier_benchmark(
            data_root=args.data,
            output_root=args.output,
            allow_unreviewed_experiment=args.experimental,
        )
        best = report["best_run"]
        print(
            "Baseline benchmark complete: "
            f"best={best['model']}+{best['feature_set']}, "
            f"macro-F1={best['macro_f1']:.3f}"
        )
        return 0

    if args.command == "verify-reset":
        from roadsign_assist.datasets.reset_audit import verify_official_backup

        report = verify_official_backup(args.backup)
        print(
            "Reset audit complete: "
            f"images={report['restored_image_count']}, "
            f"documents={report['restored_document_count']}, "
            f"passed={report['passed']}"
        )
        return 0 if report["passed"] else 1

    if args.command == "benchmark-detector":
        from roadsign_assist.evaluation.detector import benchmark_detector_runtime

        report = benchmark_detector_runtime(
            model_path=args.model,
            data_yaml=args.data,
            output_path=args.output,
            split=args.split,
            image_size=args.imgsz,
            confidence=args.confidence,
            device=args.device,
            limit=args.limit,
            task=args.task,
        )
        print(
            "Detector benchmark complete: "
            f"images={report['images']}, "
            f"mean={report['wall_latency_ms']['mean']:.1f} ms, "
            f"p95={report['wall_latency_ms']['p95']:.1f} ms"
        )
        return 0

    if args.command == "tune-detector-thresholds":
        from roadsign_assist.evaluation.detector import tune_detector_thresholds

        report = tune_detector_thresholds(
            model_path=args.model,
            data_yaml=args.data,
            output_path=args.output,
            thresholds=tuple(args.thresholds),
            image_size=args.imgsz,
            device=args.device,
            task=args.task,
        )
        print(
            "Detector threshold tuning complete: "
            f"selected={report['selected_confidence']:.2f}, "
            f"F1={report['selected_f1']:.3f}"
        )
        return 0

    if args.command == "evaluate-classifier-safety":
        from roadsign_assist.evaluation.classifier_safety import (
            evaluate_critical_class_recall,
        )

        report = evaluate_critical_class_recall(
            args.metrics,
            args.labels,
            args.output,
        )
        print(
            "Classifier safety evaluation complete: "
            f"macro_recall={report['macro_recall_observed']:.3f}, "
            f"micro_recall={report['micro_recall_observed']:.3f}"
        )
        return 0

    if args.command == "evaluate-tracking-motion":
        from roadsign_assist.evaluation.tracking import evaluate_tracking_motion

        report = evaluate_tracking_motion(args.output, args.report)
        print(
            "P11 tracking motion evaluation complete: "
            f"{report['summary']['passed_count']}/{report['summary']['scenario_count']} "
            f"passed, id_switches={report['summary']['total_id_switches']}"
        )
        return 0 if report["passed"] else 1

    if args.command == "evaluate-detector-slices":
        from roadsign_assist.evaluation.detector import (
            evaluate_detector_recall_slices,
        )

        report = evaluate_detector_recall_slices(
            model_path=args.model,
            data_yaml=args.data,
            output_path=args.output,
            split=args.split,
            image_size=args.imgsz,
            confidence=args.confidence,
            device=args.device,
            match_iou=args.match_iou,
            small_area_ratio=args.small_area_ratio,
            task=args.task,
        )
        all_recall = report["slices"]["all"]["recall_at_iou"]
        small_recall = report["slices"]["small"]["recall_at_iou"]
        print(
            f"Detector slice evaluation complete: all={all_recall:.3f}, small={small_recall:.3f}"
            if all_recall is not None and small_recall is not None
            else f"all={all_recall}, small={small_recall}"
        )
        return 0

    if args.command == "finalize-classifier-embeddings":
        from roadsign_assist.classification.embedding_export import (
            export_classifier_with_embeddings,
        )

        report = export_classifier_with_embeddings(
            checkpoint_path=args.checkpoint,
            data_root=args.data,
            model_output=args.model_output,
            calibration_output=args.calibration_output,
            report_output=args.report_output,
            device=args.device,
            batch_size=args.batch,
            workers=args.workers,
            retention_quantile=args.retention_quantile,
            include_test=args.include_test,
        )
        print(
            "Embedding classifier export complete: "
            f"distance={report['distance_threshold']:.4f}, "
            f"validation_coverage={report['validation']['coverage']:.3f}, "
            f"test_evaluated={report['test'] is not None}, "
            f"parity={report['onnx_parity']['passed']}"
        )
        return 0

    if args.command == "serve":
        import uvicorn

        os.environ["ROADSIGN_CONFIG"] = args.config
        if args.public_host:
            os.environ["ROADSIGN_PUBLIC_HOST"] = args.public_host
        uvicorn.run(
            "roadsign_api.main:app",
            host=args.host,
            port=args.port,
            reload=False,
            ssl_certfile=args.ssl_certfile,
            ssl_keyfile=args.ssl_keyfile,
        )
        return 0

    if args.command == "train-detector":
        from pathlib import Path

        from roadsign_assist.detection.training import (
            DetectorTrainingConfig,
            train_detector,
        )

        train_detector(
            DetectorTrainingConfig(
                data_yaml=Path(args.data),
                task=args.task,
                base_model=args.model,
                image_size=args.imgsz,
                epochs=args.epochs,
                batch_size=args.batch,
                device=args.device,
                run_name=args.name,
                allow_unreviewed_experiment=args.experimental,
            )
        )
        return 0

    if args.command == "train-phase-d-detector":
        from roadsign_assist.detection.phase_d import train_phase_d_detector

        report = train_phase_d_detector(
            args.candidate,
            resume=args.resume,
            device=args.device,
        )
        print(f"Phase-D training {report['status']}: {report.get('candidate_id', args.candidate)}")
        return 0

    if args.command == "phase-d-detector-status":
        from roadsign_assist.detection.phase_d import phase_d_status

        report = phase_d_status()
        if args.json:
            print(json.dumps(report))
        else:
            print(
                f"Phase D: {report['completed_runs']}/{report['total_runs']} training runs complete"
            )
            for row in report["runs"]:
                detail = f" epoch={row['epoch']}/{100}" if row.get("epoch") else ""
                if row.get("validation_map50_95") is not None:
                    detail += f" val-mAP50-95={float(row['validation_map50_95']):.4f}"
                print(f"{row['status']:>22} {row['candidate_id']}{detail}")
        return 0

    if args.command == "select-phase-d-detector":
        from roadsign_assist.detection.phase_d import select_phase_d_detector

        report = select_phase_d_detector(args.output, device=args.device)
        selected = report["selected"]
        print(
            "Phase-D validation selection complete: "
            f"{selected['candidate_id']} "
            f"mAP50-95={selected['validation_map50_95']:.4f}, "
            f"small-recall={selected['validation_small_recall']:.4f}, "
            f"confidence={selected['selected_confidence']:.2f}"
        )
        return 0

    if args.command == "evaluate-phase-d-selected-detector":
        from roadsign_assist.detection.phase_d import evaluate_phase_d_selected_detector

        report = evaluate_phase_d_selected_detector(device=args.device)
        print(
            "Phase-D locked test complete: "
            f"candidate={report['candidate_id']}, "
            f"mAP50-95={report['pytorch']['map50_95']:.4f}, "
            f"parity={report['parity']['passed']}, "
            f"gate_d={report['gate_d_passed']}"
        )
        return 0

    if args.command == "prepare-phase-e-benchmark":
        from roadsign_assist.datasets.phase_e_benchmark import prepare_phase_e_benchmark

        report = prepare_phase_e_benchmark(overwrite=args.overwrite)
        typed_report = cast(dict[str, Any], report)
        external = cast(dict[str, Any], typed_report["external"])
        image_count = int(typed_report.get("image_count", 317 + int(external["selected_images"])))
        sign_count = int(typed_report.get("sign_count", 351 + int(external["selected_boxes"])))
        unresolved = int(typed_report.get("unresolved_review_rows", sign_count))
        print(
            "Phase-E review benchmark prepared: "
            f"images={image_count}, signs={sign_count}, unresolved={unresolved}"
        )
        return 0

    if args.command == "validate-phase-e-review":
        from roadsign_assist.datasets.phase_e_benchmark import validate_phase_e_review

        report = validate_phase_e_review()
        print(
            "Phase-E owner review frozen: "
            f"images={report['images']}, signs={report['sign_boxes']}, "
            f"sha256={report['canonical_sha256']}"
        )
        return 0

    if args.command == "evaluate-phase-e-pipeline":
        from roadsign_assist.evaluation.phase_e import evaluate_phase_e_profile

        report = evaluate_phase_e_profile(args.profile)
        typed_report = cast(dict[str, Any], report)
        print(
            "Phase-E profile evaluation complete: "
            f"profile={args.profile}, "
            f"detector-recall={float(typed_report['detector']['overall']['recall']):.4f}, "
            f"end-to-end-macro-f1={float(typed_report['end_to_end']['macro_f1']):.4f}"
        )
        return 0

    if args.command == "finalize-phase-e-gate":
        from roadsign_assist.evaluation.phase_e import finalize_phase_e_gate

        report = finalize_phase_e_gate()
        print(
            "Gate E finalized: "
            f"decision={report['decision']}, "
            f"passed={report['passed_gate_count']}/{report['total_gate_count']}"
        )
        return 0

    if args.command == "phase-e-status":
        from roadsign_assist.evaluation.phase_e import phase_e_status

        report = phase_e_status()
        typed_report = cast(dict[str, Any], report)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Phase E: {typed_report['state']}")
            print(f"Benchmark: {typed_report['benchmark']['state']}")
            for profile, detail in typed_report["profiles"].items():
                print(f"{profile.upper()}: {detail['state']}")
            print(f"Gate: {typed_report['gate']['state']}")
            print(f"Runtime promoted: {typed_report['runtime_promoted']}")
        return 0

    if args.command == "promote-phase-e-runtime":
        from roadsign_assist.evaluation.phase_e import promote_phase_e_runtime

        report = promote_phase_e_runtime(internal_only=args.internal_only)
        print(
            "Phase-E runtime promoted: "
            f"backup={report['backup_id']}, "
            f"pipeline={report['pipeline_manifest_sha256']}"
        )
        return 0

    if args.command == "rollback-phase-e-runtime":
        from roadsign_assist.evaluation.phase_e import rollback_phase_e_runtime

        report = rollback_phase_e_runtime(args.backup_id)
        print(f"Phase-E runtime restored from backup {report['restored_backup_id']}")
        return 0

    if args.command == "promote-classifier":
        from roadsign_assist.classification.folder_training import (
            promote_classifier_candidate,
        )

        manifest = promote_classifier_candidate(
            args.run,
            overwrite=args.overwrite,
            internal_only=args.internal_only,
        )
        print(
            "Classifier promotion complete: "
            f"run={manifest['source_run']}, "
            f"release={manifest['release_status']}, "
            f"clean_final={manifest['clean_final']}"
        )
        return 0

    if args.command == "rollback-classifier-runtime":
        from roadsign_assist.classification.folder_training import rollback_classifier_runtime

        result = rollback_classifier_runtime(args.backup_id)
        print(f"Classifier runtime restored from backup {result['restored_backup_id']}")
        return 0

    if args.command == "phase-b-classifier-status":
        from roadsign_assist.classification.phase_b import phase_b_status

        report = phase_b_status()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Phase B: {report['completed_runs']}/{report['total_runs']} runs complete")
            for row in report["runs"]:
                summary = ""
                if row["status"] == "completed":
                    summary = f" val_f1={row.get('validation_macro_f1', 0.0):.4f}"
                elif row["status"] == "running":
                    summary = f" epoch={row.get('epoch')}/{row.get('epochs')}"
                print(f"{row['status']:>10} {row['run_name']}{summary}")
        return 0

    if args.command == "select-phase-b-classifier":
        from roadsign_assist.classification.phase_b import select_phase_b_candidate

        report = select_phase_b_candidate(args.output)
        print(f"Phase-B validation selection complete: {report['selected_single']['primary_run']}")
        return 0

    if args.command == "prepare-phase-b-ensemble":
        from roadsign_assist.classification.ensemble import prepare_phase_b_ensemble

        report = prepare_phase_b_ensemble(
            args.runs,
            name=args.name,
            target_selective_accuracy=args.target_selective_accuracy,
        )
        print(
            "Phase-B ensemble validation complete: "
            f"qualified={report['ensemble_qualification']['qualified']}"
        )
        return 0

    if args.command == "evaluate-phase-b-ensemble":
        from roadsign_assist.classification.ensemble import evaluate_phase_b_ensemble

        report = evaluate_phase_b_ensemble(args.name, overwrite=args.overwrite)
        print(
            "Phase-B ensemble locked test complete: "
            f"accuracy={report['accuracy_percent']:.2f}%, "
            f"macro-F1={report['macro_f1_all_labels'] * 100:.2f}%"
        )
        return 0

    if args.command == "prepare-phase-b-embedding-gate":
        from roadsign_assist.classification.embedding_export import (
            export_classifier_with_embeddings,
        )
        from roadsign_assist.paths import project_path

        metrics_path = project_path("outputs/training") / args.run / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("dataset_id") != "classifier_production_78_v3_20260829":
            raise ValueError("Embedding gate is limited to the Phase-B v3 release")
        artifacts = metrics["artifacts"]
        candidate_root = project_path("models/candidates") / args.run
        report = export_classifier_with_embeddings(
            checkpoint_path=artifacts["checkpoint"],
            data_root=metrics["configuration"]["data_root"],
            model_output=candidate_root / "sign_classifier.embedding.onnx",
            calibration_output=candidate_root / "sign_classifier.embedding.calibration.json",
            report_output=project_path("outputs/training") / args.run / "embedding_validation.json",
            device=args.device,
            batch_size=args.batch,
            workers=args.workers,
        )
        print(
            "Phase-B embedding gate validation complete: "
            f"qualified={report.get('validation_qualification', {}).get('qualified', False)}"
        )
        return 0

    if args.command == "evaluate-classifier-candidate":
        from roadsign_assist.classification.folder_training import (
            evaluate_classifier_candidate,
        )

        report = evaluate_classifier_candidate(
            args.run,
            tune_confidence_threshold=not args.fixed_threshold,
            target_selective_accuracy=args.target_selective_accuracy,
            overwrite=args.overwrite,
        )
        print(
            "Locked classifier test complete: "
            f"accuracy={report['accuracy_percent']:.2f}%, "
            f"macro-F1={report['macro_f1_all_labels_percent']:.2f}%, "
            f"coverage={report['selective_coverage']:.3f}"
        )
        return 0

    if args.command == "train-classifier":
        from pathlib import Path

        from roadsign_assist.classification.folder_training import (
            FolderClassifierTrainingConfig,
            train_folder_classifier,
        )

        train_folder_classifier(
            FolderClassifierTrainingConfig(
                data_root=Path(args.data),
                architecture=args.architecture,
                image_size=args.imgsz,
                epochs=args.epochs,
                batch_size=args.batch,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                workers=args.workers,
                device=args.device,
                seed=args.seed,
                run_name=args.name,
                confidence_threshold=args.confidence_threshold,
                allow_unreviewed_experiment=args.experimental,
                tune_confidence_threshold=args.tune_confidence_threshold,
                target_selective_accuracy=args.target_selective_accuracy,
                evaluate_test=args.evaluate_test,
                overwrite=args.overwrite,
                resume=args.resume,
            )
        )
        return 0

    if args.command == "finalize-detector":
        from pathlib import Path

        from roadsign_assist.detection.training import (
            DetectorExportConfig,
            evaluate_and_export_detector,
        )

        report = evaluate_and_export_detector(
            DetectorExportConfig(
                checkpoint=Path(args.checkpoint),
                data_yaml=Path(args.data),
                task=args.task,
                image_size=args.imgsz,
                device=args.device,
                artifact_name=args.name,
                allow_unreviewed_experiment=args.experimental,
            )
        )
        print(
            "Detector export complete: "
            f"parity={report['parity']['passed']}, "
            f"experimental={report['experimental']}"
        )
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
