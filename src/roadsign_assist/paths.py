from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "models"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
OFFICIAL_ROOT = DATA_ROOT / "official"
MANIFEST_ROOT = DATA_ROOT / "manifests"


def project_path(value: str | Path) -> Path:
    """Resolve a project-relative path without requiring the current directory."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_output_directories() -> None:
    for path in (MANIFEST_ROOT, MODEL_ROOT, OUTPUT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
