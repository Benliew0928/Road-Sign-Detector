from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass

import cv2

from roadsign_assist.paths import OFFICIAL_ROOT, PROJECT_ROOT


@dataclass(frozen=True)
class DiagnosticReport:
    python: str
    platform: str
    project_root: str
    opencv: str
    cuda_available: bool
    onnxruntime_available: bool
    official_image_count: int
    writable: bool
    ffmpeg_available: bool
    offline_ocr_ready: bool
    production_segmenter_present: bool
    production_classifier_present: bool
    semantic_ai_ready: bool
    experimental_model_count: int

    @property
    def healthy(self) -> bool:
        return self.official_image_count == 84 and self.writable


def _cuda_available() -> bool:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False
    result = subprocess.run(
        [executable, "-L"],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    return result.returncode == 0 and "NVIDIA" in result.stdout


def _offline_ocr_ready() -> bool:
    try:
        from roadsign_assist.ocr.assets import verify_ocr_assets

        verify_ocr_assets()
    except (FileNotFoundError, ImportError, ValueError):
        return False
    return True


def collect_diagnostics() -> DiagnosticReport:
    image_root = OFFICIAL_ROOT / "assignment_images"
    image_count = sum(
        1
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
    )
    probe = PROJECT_ROOT / ".write-probe"
    writable = False
    try:
        probe.write_text("ok", encoding="ascii")
        writable = probe.read_text(encoding="ascii") == "ok"
    finally:
        probe.unlink(missing_ok=True)

    segmenter = PROJECT_ROOT / "models" / "exported" / "sign_segmenter.onnx"
    classifier = PROJECT_ROOT / "models" / "exported" / "sign_classifier.onnx"
    experimental_root = PROJECT_ROOT / "models" / "exported" / "experimental"
    return DiagnosticReport(
        python=sys.version.split()[0],
        platform=platform.platform(),
        project_root=str(PROJECT_ROOT),
        opencv=cv2.__version__,
        cuda_available=_cuda_available(),
        onnxruntime_available=importlib.util.find_spec("onnxruntime") is not None,
        official_image_count=image_count,
        writable=writable,
        ffmpeg_available=shutil.which("ffmpeg") is not None,
        offline_ocr_ready=_offline_ocr_ready(),
        production_segmenter_present=segmenter.is_file(),
        production_classifier_present=classifier.is_file(),
        semantic_ai_ready=segmenter.is_file() and classifier.is_file(),
        experimental_model_count=(
            sum(1 for path in experimental_root.glob("*") if path.is_file())
            if experimental_root.exists()
            else 0
        ),
    )


def diagnostics_json(indent: int = 2) -> str:
    report = collect_diagnostics()
    payload = {**asdict(report), "healthy": report.healthy}
    return json.dumps(payload, indent=indent)
