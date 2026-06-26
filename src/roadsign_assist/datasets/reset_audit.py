from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from roadsign_assist.paths import OFFICIAL_ROOT, project_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def verify_official_backup(
    backup_root: str | Path = r"C:\MiniProject_OfficialBackup",
    output_path: str | Path = "outputs/evaluation/reset_audit.json",
) -> dict[str, Any]:
    backup = Path(backup_root)
    backup_images = backup / "Color Inputs"
    backup_documents = backup / "Practical 1"
    restored_images = OFFICIAL_ROOT / "assignment_images"
    restored_documents = OFFICIAL_ROOT / "coursework_documents"
    required = (
        backup_images,
        backup_documents,
        restored_images,
        restored_documents,
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing reset audit directories: {missing}")

    image_hashes = _relative_hashes(backup_images)
    restored_image_hashes = _relative_hashes(restored_images)
    document_hashes = _relative_hashes(backup_documents)
    restored_document_hashes = _relative_hashes(restored_documents)

    unreadable_images: list[str] = []
    for path in sorted(restored_images.rglob("*")):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            unreadable_images.append(path.relative_to(restored_images).as_posix())

    unreadable_documents: list[str] = []
    for path in sorted(restored_documents.rglob("*.docx")):
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    unreadable_documents.append(path.relative_to(restored_documents).as_posix())
        except (OSError, zipfile.BadZipFile):
            unreadable_documents.append(path.relative_to(restored_documents).as_posix())

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "backup_root": str(backup.resolve()),
        "backup_outside_workspace": not backup.resolve().is_relative_to(
            project_path(".").resolve()
        ),
        "backup_image_count": len(image_hashes),
        "restored_image_count": len(restored_image_hashes),
        "backup_document_count": len(document_hashes),
        "restored_document_count": len(restored_document_hashes),
        "image_checksum_match": image_hashes == restored_image_hashes,
        "document_checksum_match": document_hashes == restored_document_hashes,
        "unreadable_images": unreadable_images,
        "unreadable_documents": unreadable_documents,
    }
    report["passed"] = bool(
        report["backup_outside_workspace"]
        and report["backup_image_count"] == 84
        and report["restored_image_count"] == 84
        and report["image_checksum_match"]
        and report["document_checksum_match"]
        and not unreadable_images
        and not unreadable_documents
    )
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
