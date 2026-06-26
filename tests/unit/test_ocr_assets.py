import json
from pathlib import Path

import pytest

from roadsign_assist.ocr.assets import (
    MODEL_NAMES,
    REQUIRED_MODEL_FILES,
    verify_ocr_assets,
    write_ocr_asset_manifest,
)


def test_ocr_asset_manifest_detects_changes(tmp_path: Path) -> None:
    for model_name in MODEL_NAMES:
        model_root = tmp_path / model_name
        model_root.mkdir()
        for filename in REQUIRED_MODEL_FILES:
            (model_root / filename).write_bytes(f"{model_name}:{filename}".encode())

    manifest_path = write_ocr_asset_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["fully_local"] is True
    assert len(manifest["files"]) == 6
    verify_ocr_assets(tmp_path)

    changed = tmp_path / MODEL_NAMES[0] / REQUIRED_MODEL_FILES[0]
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="do not match"):
        verify_ocr_assets(tmp_path)
