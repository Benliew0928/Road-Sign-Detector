from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from roadsign_assist.inference.engine import InferenceEngine
from roadsign_assist.paths import PROJECT_ROOT, project_path

BUNDLE_SCHEMA_VERSION = "1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("status", "--porcelain=v1")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "clean_worktree": not status,
        "status_porcelain": status.splitlines(),
    }


def _materialize(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _create_git_bundle(root: Path, destination: Path) -> dict[str, object]:
    """Embed the exact committed source history needed to recreate the checkout."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "bundle", "create", str(destination), "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "bundle", "verify", str(destination)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "source_path": str(root),
        "bundle_path": destination.name,
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
        "materialization": "git_bundle",
        "verified": True,
    }


def freeze_runtime_bundle(
    *,
    config_path: str | Path,
    output_path: str | Path,
    dataset_releases: Mapping[str, str | Path],
    evaluation_commands: Sequence[str],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Materialize an evaluated candidate only from a clean Git checkout."""
    if not dataset_releases:
        raise ValueError("At least one dataset release manifest is required")
    if not evaluation_commands or any(not command.strip() for command in evaluation_commands):
        raise ValueError("At least one non-empty reproducible evaluation command is required")
    root = project_root.resolve()
    git = _git_metadata(root)
    if git["clean_worktree"] is not True:
        raise ValueError("Runtime bundles can be frozen only from a clean Git worktree")

    config = project_path(config_path) if root == PROJECT_ROOT else (root / config_path).resolve()
    if Path(config_path).is_absolute():
        config = Path(config_path).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Immutable runtime bundle already exists: {output}")

    engine = InferenceEngine(config)
    if engine.runtime_badge not in {"CANDIDATE", "SHADOW"}:
        raise ValueError("Only a CANDIDATE or SHADOW config can be frozen for evaluation")

    release_paths = {name: Path(value).resolve() for name, value in dataset_releases.items()}
    missing_releases = [str(path) for path in release_paths.values() if not path.is_file()]
    if missing_releases:
        raise FileNotFoundError(f"Dataset release manifests are missing: {missing_releases}")

    materialized: dict[str, dict[str, object]] = {}
    sources: dict[str, Path] = {
        "config": config,
        "dependency_lock": root / "uv.lock",
    }
    identity = cast(dict[str, Any], engine.bundle_identity)
    for name in (
        "detector",
        "classifier",
        "classifier_labels",
        "classifier_calibration",
        "catalogue",
        "tracker_config",
    ):
        record = cast(dict[str, Any], identity.get(name, {}))
        source_value = record.get("path")
        if source_value and record.get("available") is True:
            sources[name] = Path(str(source_value)).resolve()
    detector_profiles = cast(dict[str, Any], identity.get("detector_profiles", {}))
    for profile_name, raw_record in detector_profiles.items():
        record = cast(dict[str, Any], raw_record)
        source_value = record.get("path")
        if source_value and record.get("available") is True:
            sources[f"detector_profile_{profile_name}"] = Path(str(source_value)).resolve()
    for release_id, path in release_paths.items():
        sources[f"dataset__{release_id}"] = path

    missing_sources = [str(source) for source in sources.values() if not source.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Runtime bundle artifacts are missing: {missing_sources}")

    output.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        directory = "datasets" if name.startswith("dataset__") else "artifacts"
        destination = output / directory / f"{name}__{source.name}"
        method = _materialize(source, destination)
        source_hash = _sha256(source)
        frozen_hash = _sha256(destination)
        if source_hash != frozen_hash:
            raise RuntimeError(f"Frozen artifact hash mismatch: {name}")
        materialized[name] = {
            "source_path": str(source),
            "bundle_path": destination.relative_to(output).as_posix(),
            "sha256": frozen_hash,
            "size_bytes": destination.stat().st_size,
            "materialization": method,
        }

    source_bundle_path = output / "source_repository.bundle"
    materialized["source_repository"] = _create_git_bundle(root, source_bundle_path)

    manifest: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_id": output.name,
        "frozen_at": datetime.now(UTC).isoformat(),
        "runtime_badge": engine.runtime_badge,
        "config_name": engine.config_name,
        "preprocessing_version": engine.preprocessing_version,
        "git": git,
        "runtime_identity": identity,
        "dataset_release_ids": sorted(release_paths),
        "evaluation_commands": list(evaluation_commands),
        "artifacts": materialized,
    }
    manifest_path = output / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_hash = _sha256(manifest_path)
    (output / "bundle_manifest.sha256").write_text(
        f"{manifest_hash}  bundle_manifest.json\n",
        encoding="ascii",
    )
    return {**manifest, "manifest_sha256": manifest_hash, "output_path": str(output)}
