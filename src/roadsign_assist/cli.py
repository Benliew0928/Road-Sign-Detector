from __future__ import annotations

import argparse
import os
import sys

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
        default="data/processed/emtd_classification",
    )
    classifier.add_argument(
        "--architecture",
        choices=("mobilenet_v3_large", "efficientnet_v2_s"),
        default="mobilenet_v3_large",
    )
    classifier.add_argument("--epochs", type=int, default=40)
    classifier.add_argument("--batch", type=int, default=32)
    classifier.add_argument("--imgsz", type=int, default=224)
    classifier.add_argument("--device", default="auto")
    classifier.add_argument("--name", default="malaysia_sign_classifier")
    classifier.add_argument("--experimental", action="store_true")

    embedding_classifier = subparsers.add_parser(
        "finalize-classifier-embeddings",
        help="Export classifier logits plus embeddings and calibrate prototype rejection.",
    )
    embedding_classifier.add_argument("--checkpoint", required=True)
    embedding_classifier.add_argument(
        "--data",
        default="data/processed/emtd_classification",
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
        )
        print(
            "Detector threshold tuning complete: "
            f"selected={report['selected_confidence']:.2f}, "
            f"mask-F1={report['selected']['mask_f1']:.3f}"
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
        )
        print(
            "Embedding classifier export complete: "
            f"distance={report['distance_threshold']:.4f}, "
            f"test_coverage={report['test']['coverage']:.3f}, "
            f"test_selective_accuracy={report['test']['accepted_accuracy']}, "
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
                device=args.device,
                run_name=args.name,
                allow_unreviewed_experiment=args.experimental,
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
