"""Local inventory and ledger foundations for recovery acquisition Phase B0."""

from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from roadsign_assist.paths import PROJECT_ROOT, project_path

SCHEMA_VERSION = "1.0"
PLAN_ID = "legacy_replacement_autonomous_collection_v1_20260904"
DEFAULT_ACQUISITION_ROOT = Path("_archive/recovery_acquisition_v1")
DEFAULT_AUDIT_ROOT = Path("outputs/audit/recovery_acquisition_v1")
DEFAULT_SPENT_MANIFEST = Path("data/manifests/phase_e_benchmark_v1_20260901.csv")

IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff", ".webp"}
)
VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
ARCHIVE_EXTENSIONS = frozenset({".7z", ".bz2", ".gz", ".rar", ".tar", ".tgz", ".xz", ".zip"})
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | ARCHIVE_EXTENSIONS

HASH_FIELDS = (
    "sha256",
    "image_sha256",
    "original_sha256",
    "crop_sha256",
    "archive_sha256",
)
CLASS_FIELDS = ("semantic_sign_id", "class", "provisional_class", "source_semantic_hint")
SOURCE_FIELDS = ("source_id",)
DATASET_FIELDS = ("source_dataset", "source_kind", "website_dataset_name")
GROUP_FIELDS = (
    "group_id",
    "source_group",
    "source_video_group",
    "leakage_group",
    "layout_root_id",
    "related_capture_group_id",
    "session_id",
    "route_id",
    "video_id",
)
STATE_FIELDS = (
    "collection_status",
    "review_decision",
    "final_review_decision",
    "annotation_status",
    "review_status",
)
PATH_FIELDS = (
    "path",
    "image_path",
    "dataset_image_path",
    "source_path",
    "source_crop_path",
    "original_path",
    "crop_path",
    "intake_image_path",
    "candidate_release_image_path",
    "relative_path",
)
ID_FIELDS = ("sample_id", "image_id", "source_image_id", "candidate_id", "instance_id")


class SourceLedgerEntry(BaseModel):
    """Append-only source observation schema used by later acquisition phases."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    observed_at: datetime
    source_id: str
    source_name: str
    publisher: str
    owner: str
    source_url: str
    dataset_api_version: str = ""
    licence_name: str = ""
    licence_url: str = ""
    licence_status: str = "unreviewed"
    access_method: str
    adapter_version: str
    query: str = ""
    geographic_filter: str = ""
    page_cursor: str = ""
    batch_id: str
    attribution: str = ""
    restrictions: list[str] = Field(default_factory=list)
    redistribution_review_required: bool = True
    expected_archive_hash: str = ""
    actual_archive_hash: str = ""
    state: str
    state_reason: str = ""


class ObjectLedgerEntry(BaseModel):
    """Append-only acquired-object observation schema."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    observed_at: datetime
    object_id: str
    source_id: str
    batch_id: str
    group_id: str
    sequence_id: str = ""
    source_url: str = ""
    filename: str
    sha256: str
    byte_size: int = Field(ge=0)
    mime_type: str
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    source_timestamp: str = ""
    coarse_location: str = ""
    raw_path: str
    normalized_path: str = ""
    parent_object_id: str = ""
    privacy_flags: list[str] = Field(default_factory=list)
    quality_scores: dict[str, float] = Field(default_factory=dict)
    duplicate_cluster_id: str = ""
    rejection_reason: str = ""
    detector_summary: dict[str, Any] = Field(default_factory=dict)
    classifier_summary: dict[str, Any] = Field(default_factory=dict)
    capture_domain: str = "unknown"
    environment: str = "unknown"
    road_context: str = "unknown"
    apparent_size: str = "unknown"
    collection_state: str = "quarantined"
    licence_status: str = "unreviewed"
    permitted_use: str = "none_pending_review"
    review_status: str = "pending"
    promotion_status: str = "not_promoted"


class ArchiveLedgerEntry(BaseModel):
    """Append-only source archive observation schema."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    observed_at: datetime
    archive_id: str
    source_id: str
    batch_id: str
    path: str
    sha256: str
    byte_size: int = Field(ge=0)
    expected_sha256: str = ""
    member_count: int | None = Field(default=None, ge=0)
    extraction_state: str = "not_extracted"
    collection_state: str = "quarantined"


class GroupLedgerEntry(BaseModel):
    """Append-only related-capture group observation schema."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    observed_at: datetime
    group_id: str
    source_id: str
    group_kind: str
    parent_group_id: str = ""
    contributor_id: str = ""
    camera_id: str = ""
    route_id: str = ""
    sequence_id: str = ""
    session_id: str = ""
    object_count: int = Field(default=0, ge=0)
    split_lock: str = "unassigned"
    notes: str = ""


@dataclass(frozen=True)
class DiskBudget:
    path: str
    free_bytes: int
    free_gib: float
    minimum_free_bytes: int
    acquisition_bytes: int
    maximum_acquisition_bytes: int
    planned_additional_bytes: int
    allowed: bool
    reasons: tuple[str, ...]


@dataclass
class ManifestFacts:
    references: set[str] = field(default_factory=lambda: set[str]())
    classes: set[str] = field(default_factory=lambda: set[str]())
    sources: set[str] = field(default_factory=lambda: set[str]())
    datasets: set[str] = field(default_factory=lambda: set[str]())
    groups: set[str] = field(default_factory=lambda: set[str]())
    splits: set[str] = field(default_factory=lambda: set[str]())
    states: set[str] = field(default_factory=lambda: set[str]())
    licence_statuses: set[str] = field(default_factory=lambda: set[str]())
    use_policies: set[str] = field(default_factory=lambda: set[str]())
    expected_kinds: set[str] = field(default_factory=lambda: set[str]())
    domains: set[str] = field(default_factory=lambda: set[str]())
    label_paths: set[str] = field(default_factory=lambda: set[str]())
    negative: bool = False
    evaluation_only: bool = False

    def merge(self, other: ManifestFacts) -> None:
        self.references.update(other.references)
        self.classes.update(other.classes)
        self.sources.update(other.sources)
        self.datasets.update(other.datasets)
        self.groups.update(other.groups)
        self.splits.update(other.splits)
        self.states.update(other.states)
        self.licence_statuses.update(other.licence_statuses)
        self.use_policies.update(other.use_policies)
        self.expected_kinds.update(other.expected_kinds)
        self.domains.update(other.domains)
        self.label_paths.update(other.label_paths)
        self.negative = self.negative or other.negative
        self.evaluation_only = self.evaluation_only or other.evaluation_only


@dataclass
class ManifestIndex:
    by_path: dict[str, ManifestFacts] = field(default_factory=lambda: dict[str, ManifestFacts]())
    by_hash: dict[str, ManifestFacts] = field(default_factory=lambda: dict[str, ManifestFacts]())
    by_stem: dict[str, ManifestFacts] = field(default_factory=lambda: dict[str, ManifestFacts]())
    scanned: list[dict[str, object]] = field(default_factory=lambda: list[dict[str, object]]())
    failures: list[dict[str, str]] = field(default_factory=lambda: list[dict[str, str]]())


@dataclass(frozen=True)
class YoloSummary:
    bbox_count: int
    small_count: int
    very_small_count: int
    minimum_area_ratio: float | None
    class_ids: tuple[str, ...]
    error: str = ""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _json_values(values: Iterable[str]) -> str:
    return json.dumps(sorted({value for value in values if value}), separators=(",", ":"))


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _absolute_from_manifest(value: str, manifest: Path) -> Path:
    raw = Path(value.replace("/", os.sep))
    if raw.is_absolute():
        return raw
    project_candidate = PROJECT_ROOT / raw
    if project_candidate.exists() or value.startswith(("data/", "_archive/", "models/")):
        return project_candidate
    return manifest.parent / raw


def check_disk_budget(
    path: str | Path,
    *,
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    minimum_free_gib: float = 15.0,
    maximum_acquisition_gib: float = 10.0,
    planned_additional_bytes: int = 0,
) -> DiskBudget:
    """Check both the free-space floor and the acquisition allocation."""
    target = project_path(path)
    usage_probe = target
    while not usage_probe.exists() and usage_probe != usage_probe.parent:
        usage_probe = usage_probe.parent
    usage = shutil.disk_usage(usage_probe)
    acquisition = project_path(acquisition_root)
    acquisition_bytes = (
        sum(candidate.stat().st_size for candidate in acquisition.rglob("*") if candidate.is_file())
        if acquisition.exists()
        else 0
    )
    minimum = int(minimum_free_gib * 1024**3)
    maximum = int(maximum_acquisition_gib * 1024**3)
    reasons: list[str] = []
    if usage.free - planned_additional_bytes < minimum:
        reasons.append("free_space_floor")
    if acquisition_bytes + planned_additional_bytes > maximum:
        reasons.append("acquisition_budget")
    return DiskBudget(
        path=str(target),
        free_bytes=usage.free,
        free_gib=round(usage.free / 1024**3, 3),
        minimum_free_bytes=minimum,
        acquisition_bytes=acquisition_bytes,
        maximum_acquisition_bytes=maximum,
        planned_additional_bytes=planned_additional_bytes,
        allowed=not reasons,
        reasons=tuple(reasons),
    )


def _row_facts(row: Mapping[str, str], manifest: Path, row_number: int) -> ManifestFacts:
    fact = ManifestFacts(references={f"{_display_path(manifest)}:{row_number}"})
    for field_name in CLASS_FIELDS:
        if value := row.get(field_name, "").strip():
            fact.classes.add(value)
    for field_name in SOURCE_FIELDS:
        if value := row.get(field_name, "").strip():
            fact.sources.add(value)
    for field_name in DATASET_FIELDS:
        if value := row.get(field_name, "").strip():
            fact.datasets.add(value)
    for field_name in GROUP_FIELDS:
        if value := row.get(field_name, "").strip():
            fact.groups.add(value)
    if value := row.get("split", "").strip():
        fact.splits.add(value)
    for field_name in STATE_FIELDS:
        if value := row.get(field_name, "").strip():
            fact.states.add(f"{field_name}:{value}")
    if value := row.get("licence_status", row.get("rights_status", "")).strip():
        fact.licence_statuses.add(value)
    if value := row.get("use_policy", "").strip():
        fact.use_policies.add(value)
    if value := row.get("expected_kind", "").strip():
        fact.expected_kinds.add(value)
    for field_name in ("capture_domain", "domain", "scene"):
        if value := row.get(field_name, "").strip():
            fact.domains.add(value)
    if value := row.get("dataset_label_path", row.get("label_path", "")).strip():
        fact.label_paths.add(str(_absolute_from_manifest(value, manifest)))
    fact.negative = _truthy(row.get("is_negative", "")) or row.get("expected_kind", "") == "no_sign"
    fact.evaluation_only = _truthy(row.get("evaluation_only", ""))
    return fact


def _merge_index(target: dict[str, ManifestFacts], key: str, facts: ManifestFacts) -> None:
    if not key:
        return
    if key not in target:
        target[key] = ManifestFacts()
    target[key].merge(facts)


def build_manifest_index(manifest_roots: Sequence[str | Path]) -> ManifestIndex:
    """Index local CSV manifests/annotations by path, content hash, and stable IDs."""
    index = ManifestIndex()
    manifests = sorted(
        {
            path
            for raw_root in manifest_roots
            for path in (
                [project_path(raw_root)]
                if project_path(raw_root).is_file()
                else project_path(raw_root).rglob("*.csv")
            )
            if path.is_file()
        },
        key=lambda value: value.as_posix(),
    )
    for manifest in manifests:
        rows_read = 0
        try:
            with manifest.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                for row_number, raw_row in enumerate(reader, start=2):
                    row = {
                        str(key): str(value or "")
                        for key, value in raw_row.items()
                        if key is not None
                    }
                    rows_read += 1
                    facts = _row_facts(row, manifest, row_number)
                    for field_name in PATH_FIELDS:
                        value = row.get(field_name, "").strip()
                        if value:
                            _merge_index(
                                index.by_path, _key(_absolute_from_manifest(value, manifest)), facts
                            )
                    for field_name in HASH_FIELDS:
                        value = row.get(field_name, "").strip().lower()
                        if len(value) == 64:
                            _merge_index(index.by_hash, value, facts)
                    for field_name in ID_FIELDS:
                        value = row.get(field_name, "").strip()
                        if value:
                            _merge_index(index.by_stem, value, facts)
            index.scanned.append({"path": _display_path(manifest), "rows": rows_read})
        except (csv.Error, OSError, UnicodeError) as exc:
            index.failures.append({"path": str(manifest), "error": f"{type(exc).__name__}: {exc}"})
    return index


def _nearest_metadata(path: Path) -> dict[str, str]:
    current = path.parent
    while current != PROJECT_ROOT and PROJECT_ROOT in current.parents:
        metadata = current / "dataset_metadata.json"
        if metadata.is_file():
            try:
                raw = json.loads(metadata.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
            if isinstance(raw, Mapping):
                return {
                    str(key): str(value)
                    for key, value in cast(Mapping[object, object], raw).items()
                    if isinstance(value, (str, int, float, bool))
                }
            return {}
        current = current.parent
    return {}


def _path_facts(path: Path) -> ManifestFacts:
    facts = ManifestFacts()
    relative = _display_path(path)
    parts = [part.lower() for part in path.parts]
    metadata = _nearest_metadata(path)
    if source_id := metadata.get("source_id", ""):
        facts.sources.add(source_id)
    if "data" in parts:
        data_index = parts.index("data")
        if len(parts) > data_index + 2 and parts[data_index + 1] in {
            "processed",
            "raw",
            "official",
        }:
            facts.datasets.add(parts[data_index + 2])
    for split in ("train", "validation", "test", "internal_test"):
        if split in parts:
            facts.splits.add(split)
    if (
        ("classification" in relative or any("classifier" in part for part in parts))
        and len(parts) >= 2
        and path.parent.name not in {"train", "validation", "test"}
    ):
        facts.classes.add(path.parent.name)
    return facts


def _media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return "archive"


def _image_metadata(path: Path) -> tuple[int | None, int | None, str, str]:
    try:
        with Image.open(path) as raw:
            width, height = raw.size
            orientation = raw.getexif().get(274, 1)
            if orientation in {5, 6, 7, 8}:
                width, height = height, width
            return width, height, raw.format or "", ""
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return None, None, "", f"{type(exc).__name__}: {exc}"


def _corresponding_label(path: Path, facts: ManifestFacts) -> Path | None:
    candidates = [Path(value) for value in facts.label_paths]
    parts = list(path.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        candidates.append(Path(*parts).with_suffix(".txt"))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _yolo_summary(label_path: Path | None) -> YoloSummary:
    if label_path is None:
        return YoloSummary(0, 0, 0, None, ())
    areas: list[float] = []
    classes: set[str] = set()
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            classes.add(parts[0])
            width, height = float(parts[3]), float(parts[4])
            if width >= 0 and height >= 0:
                areas.append(width * height)
    except (OSError, UnicodeError, ValueError) as exc:
        return YoloSummary(0, 0, 0, None, (), f"{type(exc).__name__}: {exc}")
    return YoloSummary(
        bbox_count=len(areas),
        small_count=sum(area <= 0.01 for area in areas),
        very_small_count=sum(area <= 0.001 for area in areas),
        minimum_area_ratio=min(areas) if areas else None,
        class_ids=tuple(sorted(classes)),
    )


def _context_type(path: Path, facts: ManifestFacts) -> str:
    domains = " ".join(facts.domains).lower()
    normalized = path.as_posix().lower()
    if facts.negative:
        return "no_sign_or_negative"
    if "screen" in domains or "print" in domains or "presentation" in domains:
        return "presentation"
    if "full_road" in domains or (
        "/images/" in normalized
        and ("detector" in normalized or "detection" in normalized or "segmentation" in normalized)
    ):
        return "full_context"
    if "crop" in normalized or "classifier" in normalized or "classification" in normalized:
        return "crop_only"
    return "unknown"


def _collection_state(facts: ManifestFacts, *, spent: bool, decode_error: str) -> str:
    if decode_error:
        return "local_decode_failed"
    if spent or facts.evaluation_only:
        return "spent_evaluation_denylisted"
    joined = " ".join(facts.states).lower()
    if any(value in joined for value in ("accept", "approved")):
        return "existing_reviewed_release"
    if facts.references:
        return "existing_registered_unreviewed"
    return "local_unregistered"


def _recoverability_score(
    *, context_type: str, state: str, facts: ManifestFacts, duplicate_count: int, bbox_count: int
) -> int:
    if state in {"spent_evaluation_denylisted", "local_decode_failed"}:
        return 0
    score = {
        "full_context": 100,
        "no_sign_or_negative": 90,
        "presentation": 80,
        "unknown": 55,
        "crop_only": 25,
    }[context_type]
    if facts.sources or facts.datasets:
        score += 10
    if facts.groups:
        score += 8
    if facts.classes:
        score += 7
    if bbox_count:
        score += 10
    if state == "existing_registered_unreviewed":
        score += 5
    if duplicate_count > 1:
        score -= min(25, duplicate_count - 1)
    return max(0, score)


def _schema_files(schema_root: Path) -> dict[str, str]:
    models: dict[str, type[BaseModel]] = {
        "source_ledger": SourceLedgerEntry,
        "object_ledger": ObjectLedgerEntry,
        "archive_ledger": ArchiveLedgerEntry,
        "group_ledger": GroupLedgerEntry,
    }
    schema_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    for name, model in models.items():
        path = schema_root / f"{name}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result[name] = str(path)
    return result


def _scan_dvc_roots(scan_roots: Sequence[Path]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for descriptor in sorted(PROJECT_ROOT.rglob("*.dvc"), key=lambda value: value.as_posix()):
        if not descriptor.is_file() or any(
            part in {".git", ".venv", "_archive", "node_modules"} for part in descriptor.parts
        ):
            continue
        try:
            raw_object: object = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            results.append({"descriptor": str(descriptor), "state": "invalid", "error": str(exc)})
            continue
        if not isinstance(raw_object, Mapping):
            continue
        raw = cast(Mapping[object, object], raw_object)
        outs_object = raw.get("outs", [])
        if not isinstance(outs_object, list):
            continue
        for out_object in cast(list[object], outs_object):
            if not isinstance(out_object, Mapping):
                continue
            out = cast(Mapping[object, object], out_object)
            declared_path = out.get("path")
            if not isinstance(declared_path, str) or not declared_path:
                continue
            declared = (descriptor.parent / declared_path).resolve(strict=False)
            if not any(
                declared == root or root in declared.parents or declared in root.parents
                for root in scan_roots
            ):
                continue
            results.append(
                {
                    "descriptor": str(descriptor.relative_to(PROJECT_ROOT)),
                    "path": str(declared),
                    "state": "available" if declared.exists() else "not_pulled",
                    "declared_hash": str(out.get("md5", "")),
                    "declared_bytes": _as_int(out.get("size", 0) or 0),
                    "declared_files": _as_int(out.get("nfiles", 0) or 0),
                }
            )
    return results


def build_spent_denylist(
    manifest_path: str | Path = DEFAULT_SPENT_MANIFEST,
) -> tuple[list[dict[str, str]], set[str]]:
    """Build a unique denylist from every full-image and crop hash in the spent benchmark."""
    manifest = project_path(manifest_path)
    rows: dict[tuple[str, str], dict[str, str]] = {}
    if not manifest.is_file():
        return [], set()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            for hash_kind, field_name in (("image", "sha256"), ("crop", "crop_sha256")):
                digest = str(row.get(field_name, "")).strip().lower()
                if len(digest) != 64:
                    continue
                key = (hash_kind, digest)
                if key not in rows:
                    rows[key] = {
                        "benchmark_id": str(row.get("benchmark_id", "")),
                        "image_id": str(row.get("image_id", "")),
                        "instance_id": str(row.get("instance_id", "")),
                        "image_path": str(row.get("image_path", "")),
                        "hash_kind": hash_kind,
                        "sha256": digest,
                        "reason": "spent_phase_e_benchmark_exclude_from_training_and_new_evaluation",
                        "source_manifest": _display_path(manifest),
                    }
    ordered = [rows[key] for key in sorted(rows)]
    return ordered, {row["sha256"] for row in ordered}


def _write_csv_atomic(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _existing_inventory(path: Path, *, resume: bool) -> dict[str, dict[str, object]]:
    if not resume or not path.is_file():
        return {}
    parquet: Any = pq
    table: Any = parquet.read_table(path)
    raw_rows = cast(list[dict[str, object]], table.to_pylist())
    return {str(row["absolute_path"]): row for row in raw_rows}


def _distribution(rows: Sequence[Mapping[str, object]], field_name: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field_name, "")) for row in rows).items()))


def _multi_distribution(rows: Sequence[Mapping[str, object]], field_name: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        raw = row.get(field_name, "[]")
        if not isinstance(raw, str):
            continue
        try:
            values_object: object = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(values_object, list):
            counts.update(str(value) for value in cast(list[object], values_object))
    return dict(sorted(counts.items()))


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected integer-compatible value, got {type(value).__name__}")


def _resolution_bucket(width: object, height: object) -> str:
    if not isinstance(width, int) or not isinstance(height, int):
        return "unknown"
    pixels = width * height
    if pixels < 256 * 256:
        return "thumbnail_under_0_07mp"
    if pixels < 1_000_000:
        return "small_under_1mp"
    if pixels < 4_000_000:
        return "medium_1_to_4mp"
    return "large_4mp_or_more"


def inventory_local_candidates(
    *,
    scan_roots: Sequence[str | Path] = ("data",),
    manifest_roots: Sequence[str | Path] = ("data/manifests", "data/annotations"),
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    spent_manifest: str | Path = DEFAULT_SPENT_MANIFEST,
    minimum_free_gib: float = 15.0,
    maximum_acquisition_gib: float = 10.0,
    max_objects: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Inventory all local candidate media and write the required Phase B0 outputs."""
    roots = [project_path(root).resolve(strict=False) for root in scan_roots]
    missing_roots = [str(root) for root in roots if not root.exists()]
    roots = [root for root in roots if root.exists()]
    candidates = sorted(
        {
            path.resolve()
            for root in roots
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
        },
        key=lambda value: value.as_posix(),
    )
    if max_objects is not None:
        candidates = candidates[:max_objects]
    planned_output = sum(path.stat().st_size for path in candidates)
    budget = check_disk_budget(
        project_path(acquisition_root),
        acquisition_root=acquisition_root,
        minimum_free_gib=minimum_free_gib,
        maximum_acquisition_gib=maximum_acquisition_gib,
    )
    preflight: dict[str, Any] = {
        "plan_id": PLAN_ID,
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run" if dry_run else "inventory",
        "scan_roots": [str(root) for root in roots],
        "missing_scan_roots": missing_roots,
        "candidate_objects": len(candidates),
        "candidate_bytes": planned_output,
        "disk_budget": asdict(budget),
        "dvc_roots": _scan_dvc_roots(roots),
    }
    if dry_run:
        return preflight
    if not budget.allowed:
        raise RuntimeError(f"Disk budget guard blocked inventory: {', '.join(budget.reasons)}")

    acquisition = project_path(acquisition_root)
    manifests_dir = acquisition / "manifests"
    audit = project_path(audit_root)
    inventory_path = manifests_dir / "local_inventory.parquet"
    denylist_path = manifests_dir / "spent_data_denylist.csv"
    report_path = audit / "inventory.json"
    log_path = acquisition / "logs" / "inventory.jsonl"
    for directory in (manifests_dir, audit, log_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    existing_outputs = [
        path for path in (inventory_path, denylist_path, report_path) if path.exists()
    ]
    if existing_outputs and not resume:
        raise FileExistsError(
            f"Refusing to overwrite Phase B0 outputs without --resume: {existing_outputs}"
        )

    prior = _existing_inventory(inventory_path, resume=resume)
    manifest_index = build_manifest_index(manifest_roots)
    denylist_rows, spent_hashes = build_spent_denylist(spent_manifest)
    base_rows: list[dict[str, object]] = []
    integrity_failures: list[dict[str, str]] = []
    reused = 0
    for path in candidates:
        stat = path.stat()
        absolute = str(path)
        cached = prior.get(absolute)
        if (
            cached
            and cached.get("byte_size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
        ):
            digest = str(cached["sha256"])
            width = cast(int | None, cached.get("width"))
            height = cast(int | None, cached.get("height"))
            image_format = str(cached.get("image_format", ""))
            decode_error = str(cached.get("decode_error", ""))
            reused += 1
        else:
            digest = _sha256(path)
            width, height, image_format, decode_error = (
                _image_metadata(path) if _media_kind(path) == "image" else (None, None, "", "")
            )
        facts = ManifestFacts()
        facts.merge(_path_facts(path))
        for candidate_facts in (
            manifest_index.by_path.get(_key(path)),
            manifest_index.by_hash.get(digest),
            manifest_index.by_stem.get(path.stem),
        ):
            if candidate_facts is not None:
                facts.merge(candidate_facts)
        yolo = (
            _yolo_summary(_corresponding_label(path, facts))
            if _media_kind(path) == "image"
            else YoloSummary(0, 0, 0, None, ())
        )
        if decode_error or yolo.error:
            integrity_failures.append({"path": absolute, "error": decode_error or yolo.error})
        relative = (
            path.relative_to(PROJECT_ROOT).as_posix() if PROJECT_ROOT in path.parents else absolute
        )
        base_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "inventory_id": hashlib.sha256(f"local-path:{relative}".encode()).hexdigest()[:24],
                "object_id": f"sha256:{digest}",
                "relative_path": relative,
                "absolute_path": absolute,
                "media_kind": _media_kind(path),
                "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "extension": path.suffix.lower(),
                "byte_size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
                "width": width,
                "height": height,
                "image_format": image_format,
                "decode_error": decode_error or yolo.error,
                "context_type": _context_type(path, facts),
                "classes_json": _json_values(facts.classes),
                "sources_json": _json_values(facts.sources),
                "datasets_json": _json_values(facts.datasets),
                "groups_json": _json_values(facts.groups),
                "splits_json": _json_values(facts.splits),
                "manifest_states_json": _json_values(facts.states),
                "manifest_references_json": _json_values(facts.references),
                "licence_statuses_json": _json_values(facts.licence_statuses),
                "use_policies_json": _json_values(facts.use_policies),
                "bbox_count": yolo.bbox_count,
                "small_bbox_count": yolo.small_count,
                "very_small_bbox_count": yolo.very_small_count,
                "minimum_bbox_area_ratio": yolo.minimum_area_ratio,
                "bbox_class_ids_json": _json_values(yolo.class_ids),
                "spent_benchmark": digest in spent_hashes or facts.evaluation_only,
                "collection_state": _collection_state(
                    facts,
                    spent=digest in spent_hashes,
                    decode_error=decode_error or yolo.error,
                ),
                "origin_status": "known" if facts.sources else "unknown_recorded",
                "group_status": "known" if facts.groups else "unknown_recorded",
                "split_status": "known" if facts.splits else "unknown_recorded",
                "licence_status": next(iter(sorted(facts.licence_statuses)), "unreviewed"),
                "permitted_use": next(iter(sorted(facts.use_policies)), "none_pending_review"),
            }
        )

    sha_counts = Counter(str(row["sha256"]) for row in base_rows)
    rows: list[dict[str, object]] = []
    for row in base_rows:
        digest = str(row["sha256"])
        facts = ManifestFacts(
            sources=set(cast(list[str], json.loads(str(row["sources_json"])))),
            datasets=set(cast(list[str], json.loads(str(row["datasets_json"])))),
            groups=set(cast(list[str], json.loads(str(row["groups_json"])))),
            classes=set(cast(list[str], json.loads(str(row["classes_json"])))),
        )
        count = sha_counts[digest]
        row["duplicate_cluster_id"] = f"sha256:{digest}" if count > 1 else ""
        row["exact_duplicate_path_count"] = count
        row["recoverability_score"] = _recoverability_score(
            context_type=str(row["context_type"]),
            state=str(row["collection_state"]),
            facts=facts,
            duplicate_count=count,
            bbox_count=_as_int(row["bbox_count"]),
        )
        rows.append(row)

    arrow: Any = pa
    parquet: Any = pq
    table: Any = arrow.Table.from_pylist(rows)
    temp_parquet = inventory_path.with_suffix(".parquet.tmp")
    parquet.write_table(table, temp_parquet, compression="zstd")
    os.replace(temp_parquet, inventory_path)
    deny_fields = (
        "benchmark_id",
        "image_id",
        "instance_id",
        "image_path",
        "hash_kind",
        "sha256",
        "reason",
        "source_manifest",
    )
    _write_csv_atomic(denylist_path, deny_fields, denylist_rows)
    schema_paths = _schema_files(manifests_dir / "schemas")
    duplicate_clusters = sum(count > 1 for count in sha_counts.values())
    duplicate_paths = sum(count for count in sha_counts.values() if count > 1)
    resolution_counts = Counter(
        _resolution_bucket(row.get("width"), row.get("height")) for row in rows
    )
    top_recoverable = sorted(
        (
            {
                "relative_path": row["relative_path"],
                "score": row["recoverability_score"],
                "context_type": row["context_type"],
                "classes": json.loads(str(row["classes_json"])),
                "sources": json.loads(str(row["sources_json"])),
                "datasets": json.loads(str(row["datasets_json"])),
                "bbox_count": row["bbox_count"],
            }
            for row in rows
            if _as_int(row["recoverability_score"]) > 0
        ),
        key=lambda item: (-_as_int(item["score"]), str(item["relative_path"])),
    )[:100]
    report: dict[str, Any] = {
        **preflight,
        "mode": "inventory",
        "status": "phase_b0_complete" if len(rows) == len(candidates) else "phase_b0_incomplete",
        "generated_at": _now(),
        "outputs": {
            "inventory_parquet": str(inventory_path),
            "spent_data_denylist": str(denylist_path),
            "schemas": schema_paths,
        },
        "counts": {
            "objects": len(rows),
            "bytes": sum(_as_int(row["byte_size"]) for row in rows),
            "images": sum(row["media_kind"] == "image" for row in rows),
            "videos": sum(row["media_kind"] == "video" for row in rows),
            "archives": sum(row["media_kind"] == "archive" for row in rows),
            "unique_sha256": len(sha_counts),
            "exact_duplicate_clusters": duplicate_clusters,
            "paths_in_duplicate_clusters": duplicate_paths,
            "spent_denylist_hashes": len(spent_hashes),
            "spent_local_objects": sum(bool(row["spent_benchmark"]) for row in rows),
            "bbox_instances": sum(_as_int(row["bbox_count"]) for row in rows),
            "small_bbox_instances_at_most_1pct": sum(
                _as_int(row["small_bbox_count"]) for row in rows
            ),
            "very_small_bbox_instances_at_most_0_1pct": sum(
                _as_int(row["very_small_bbox_count"]) for row in rows
            ),
            "resume_rows_reused": reused,
        },
        "distributions": {
            "media_kind": _distribution(rows, "media_kind"),
            "extension": _distribution(rows, "extension"),
            "resolution": dict(sorted(resolution_counts.items())),
            "context_type": _distribution(rows, "context_type"),
            "collection_state": _distribution(rows, "collection_state"),
            "origin_status": _distribution(rows, "origin_status"),
            "group_status": _distribution(rows, "group_status"),
            "split_status": _distribution(rows, "split_status"),
            "classes": _multi_distribution(rows, "classes_json"),
            "sources": _multi_distribution(rows, "sources_json"),
            "datasets": _multi_distribution(rows, "datasets_json"),
            "groups": _multi_distribution(rows, "groups_json"),
            "splits": _multi_distribution(rows, "splits_json"),
        },
        "metadata_missing": {
            "origin": sum(row["origin_status"] != "known" for row in rows),
            "dataset": sum(row["datasets_json"] == "[]" for row in rows),
            "group": sum(row["group_status"] != "known" for row in rows),
            "split": sum(row["split_status"] != "known" for row in rows),
            "licence": sum(row["licence_status"] == "unreviewed" for row in rows),
            "manifest_join": sum(row["manifest_references_json"] == "[]" for row in rows),
        },
        "manifests": {
            "scanned": manifest_index.scanned,
            "failures": manifest_index.failures,
        },
        "integrity_failures": integrity_failures,
        "top_recoverable_existing_candidates": top_recoverable,
        "exit_condition": {
            "all_candidates_hashed": all(len(str(row["sha256"])) == 64 for row in rows),
            "all_candidates_have_explicit_state": all(
                bool(row["collection_state"]) for row in rows
            ),
            "unknown_origin_recorded_not_guessed": all(bool(row["origin_status"]) for row in rows),
            "passed": (
                len(rows) == len(candidates)
                and all(len(str(row["sha256"])) == 64 for row in rows)
                and all(bool(row["collection_state"]) for row in rows)
            ),
        },
    }
    _write_json_atomic(report_path, report)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "inventory_complete",
                    "at": _now(),
                    "report": str(report_path),
                    "counts": report["counts"],
                },
                sort_keys=True,
            )
            + "\n"
        )
    return report


__all__ = [
    "ArchiveLedgerEntry",
    "DiskBudget",
    "GroupLedgerEntry",
    "ObjectLedgerEntry",
    "SourceLedgerEntry",
    "build_manifest_index",
    "build_spent_denylist",
    "check_disk_budget",
    "inventory_local_candidates",
]
