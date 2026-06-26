from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from roadsign_assist.paths import project_path

MODEL_NAMES = ("PP-OCRv6_small_det", "PP-OCRv6_small_rec")
REQUIRED_MODEL_FILES = ("inference.json", "inference.pdiparams", "inference.yml")
OFFICIAL_DOCUMENTATION = "https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_ocr_asset_manifest(
    model_root: str | Path = "models/ocr",
) -> dict[str, Any]:
    root = project_path(model_root)
    files: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        model_dir = root / model_name
        for filename in REQUIRED_MODEL_FILES:
            path = model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {
        "schema_version": "1.0",
        "provider": "PaddleOCR",
        "paddleocr_version": importlib.metadata.version("paddleocr"),
        "models": list(MODEL_NAMES),
        "documentation": OFFICIAL_DOCUMENTATION,
        "device": "cpu",
        "fully_local": True,
        "files": files,
    }


def write_ocr_asset_manifest(
    model_root: str | Path = "models/ocr",
) -> Path:
    root = project_path(model_root)
    manifest_path = root / "manifest.json"
    payload = build_ocr_asset_manifest(root)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def verify_ocr_assets(
    model_root: str | Path = "models/ocr",
) -> dict[str, Any]:
    root = project_path(model_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    expected: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = build_ocr_asset_manifest(root)
    expected_files = {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in expected.get("files", [])
    }
    current_files = {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"])) for item in current["files"]
    }
    if expected_files != current_files:
        raise ValueError("Local OCR assets do not match models/ocr/manifest.json")
    return expected
