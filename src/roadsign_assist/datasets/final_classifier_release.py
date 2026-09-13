"""Audit and freeze the final, train-only extension of the classifier release.

The frozen classifier release owns validation and test.  This module therefore
never re-splits data: it verifies those files and appends audited contributor
records to the training split only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

RELEASE_ID = "classifier_production_78_v3_20260829"
FROZEN_RELEASE_ID = "classifier_no_controlled_variants_20260812"
V2_RELEASE_ID = "classifier_production_78_v2_20260826"
PERCEPTUAL_HAMMING_THRESHOLD = 6
GENERIC_GROUP_MARKERS = ("not_", "none", "collection", "batch")


@dataclass(frozen=True)
class ContributorManifestSpec:
    """The reviewed semantic meaning of a source manifest.

    Source dataset class names are deliberately not used as semantic labels.
    """

    filename: str
    semantic_sign_id: str


CONTRIBUTOR_MANIFEST_SPECS = {
    spec.filename: spec
    for spec in (
        ContributorManifestSpec(
            "stage_c_no_left_or_right_turn_roboflow_class_137_all_20260827.csv",
            "no_left_or_right_turn",
        ),
        ContributorManifestSpec(
            "stage_c_no_left_or_right_turn_web_batch_01_20260826.csv",
            "no_left_or_right_turn",
        ),
        ContributorManifestSpec(
            "stage_c_no_left_or_right_turn_web_batch_02_20260827.csv",
            "no_left_or_right_turn",
        ),
        ContributorManifestSpec(
            "stage_c_no_left_or_right_turn_web_batch_03_20260827.csv",
            "no_left_or_right_turn",
        ),
        ContributorManifestSpec(
            "stage_c_no_left_or_right_turn_web_batch_04_20260827.csv",
            "no_left_or_right_turn",
        ),
        ContributorManifestSpec(
            "stage_c_no_left_or_right_turn_web_pilot_20260826.csv",
            "no_left_or_right_turn",
        ),
        ContributorManifestSpec(
            "stage_c_no_straight_or_left_catsad_batch_05_20260828.csv",
            "no_straight_or_left",
        ),
        ContributorManifestSpec(
            "stage_c_no_straight_or_left_mapillary_batch_02_20260828.csv",
            "no_straight_or_left",
        ),
        ContributorManifestSpec(
            "stage_c_no_straight_or_left_mapillary_batch_03_20260828.csv",
            "no_straight_or_left",
        ),
        ContributorManifestSpec(
            "stage_c_no_straight_or_left_web_batch_01_20260827.csv",
            "no_straight_or_left",
        ),
        ContributorManifestSpec(
            "stage_c_no_straight_or_left_web_batch_04_20260828.csv",
            "no_straight_or_left",
        ),
        ContributorManifestSpec(
            "stage_c_no_straight_or_left_web_batch_06_20260828.csv",
            "no_straight_or_left",
        ),
        ContributorManifestSpec(
            "stage_c_no_straight_or_left_web_pilot_20260827.csv",
            "no_straight_or_left",
        ),
        ContributorManifestSpec(
            "stage_c_residential_area_ahead_catsad_batch_01_20260828.csv",
            "residential_area_ahead",
        ),
        ContributorManifestSpec(
            "stage_c_side_road_right_catsad_batch_01_20260828.csv",
            "side_road_right",
        ),
        ContributorManifestSpec(
            "stage_c_sound_horn_roboflow_signboard_batch_02_20260827.csv",
            "sound_horn",
        ),
        ContributorManifestSpec(
            "stage_c_sound_horn_roboflow_vaudit_batch_01_20260827.csv",
            "sound_horn",
        ),
        ContributorManifestSpec(
            "stage_c_steep_descent_web_batch_01_20260827.csv",
            "steep_descent",
        ),
        ContributorManifestSpec(
            "stage_c_steep_descent_web_batch_02_20260827.csv",
            "steep_descent",
        ),
        ContributorManifestSpec(
            "stage_c_turn_left_or_right_catsad_batch_01_20260828.csv",
            "turn_left_or_right",
        ),
        ContributorManifestSpec(
            "stage_c_turn_left_or_right_gtsign220_batch_01_20260828.csv",
            "turn_left_or_right",
        ),
    )
}


@dataclass(frozen=True)
class FinalClassifierReleaseConfig:
    """Locations and policy choices for a deterministic final classifier release."""

    project_root: Path
    release_id: str = RELEASE_ID
    frozen_manifest: Path | None = None
    v2_manifest: Path | None = None
    v2_root: Path | None = None
    contributor_manifest_dir: Path | None = None
    progress_tracker: Path | None = None
    output_root: Path | None = None
    output_manifest: Path | None = None
    audit_root: Path | None = None
    contributor_manifest_specs: Mapping[str, ContributorManifestSpec] | None = None
    perceptual_hamming_threshold: int = PERCEPTUAL_HAMMING_THRESHOLD
    allow_internal_academic_exceptions: bool = True
    allow_must_have_coverage_exception: bool = False
    coverage_exception_note: str = ""

    def __post_init__(self) -> None:
        root = self.project_root.resolve()
        object.__setattr__(self, "project_root", root)
        object.__setattr__(
            self,
            "frozen_manifest",
            self.frozen_manifest or root / "data/manifests/classifier_release.csv",
        )
        object.__setattr__(
            self,
            "v2_manifest",
            self.v2_manifest
            or root / "data/manifests/classifier_production_78_v2_20260826.csv",
        )
        object.__setattr__(
            self,
            "v2_root",
            self.v2_root or root / "data/processed" / V2_RELEASE_ID,
        )
        object.__setattr__(
            self,
            "contributor_manifest_dir",
            self.contributor_manifest_dir or root / "data/manifests",
        )
        object.__setattr__(
            self,
            "progress_tracker",
            self.progress_tracker or root / "data/manifests/PRODUCTION_78_DATA_PROGRESS.csv",
        )
        object.__setattr__(
            self,
            "output_root",
            self.output_root or root / "data/processed" / self.release_id,
        )
        object.__setattr__(
            self,
            "output_manifest",
            self.output_manifest or root / "data/manifests" / f"{self.release_id}.csv",
        )
        object.__setattr__(
            self,
            "audit_root",
            self.audit_root or root / "outputs/audit" / self.release_id,
        )


@dataclass
class ImageEvidence:
    file_sha256: str
    pixel_sha256: str
    perceptual_hash: str
    width: int
    height: int


def _empty_exception_reasons() -> list[str]:
    return []


@dataclass
class Candidate:
    identifier: str
    source_priority: int
    source_kind: str
    semantic_sign_id: str
    split: str
    crop_path: Path
    source_crop_path: str
    original_path: str
    original_sha256: str
    crop_sha256: str
    crop_pixel_sha256: str
    perceptual_hash: str
    crop_width: int
    crop_height: int
    source_manifest: str
    source_candidate_id: str
    source_group: str
    source_video_group: str
    source_dataset: str
    source_url: str
    licence_status: str
    source_registry_status: str
    provenance_status: str
    review_decision: str
    small_object: bool
    name_en: str
    priority: str
    required_for: str
    mapping_evidence: str
    license_notes: str
    review_notes: str
    internal_exception_reasons: list[str] = dataclass_field(
        default_factory=_empty_exception_reasons
    )
    perceptual_near_match_ids: list[str] = dataclass_field(
        default_factory=_empty_exception_reasons
    )
    dedupe_component: str = ""
    dedupe_resolution: str = "pending"
    release_path: Path | None = None

    @property
    def is_internal_exception(self) -> bool:
        return bool(self.internal_exception_reasons)


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parents = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parents[value]
        if parent != value:
            self.parents[value] = self.find(parent)
        return self.parents[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[right_root] = left_root


CANONICAL_FIELDS = [
    "release_id",
    "sample_id",
    "split",
    "semantic_sign_id",
    "name_en",
    "priority",
    "required_for",
    "source_manifest",
    "source_candidate_id",
    "source_crop_path",
    "original_path",
    "dataset_image_path",
    "original_sha256",
    "crop_sha256",
    "crop_pixel_sha256",
    "perceptual_hash",
    "perceptual_near_match_ids",
    "crop_width",
    "crop_height",
    "source_group",
    "source_video_group",
    "source_dataset",
    "source_url",
    "licence_status",
    "source_registry_status",
    "provenance_status",
    "internal_academic_exception_reasons",
    "review_decision",
    "mapping_evidence",
    "license_notes",
    "review_notes",
    "small_object",
    "dedupe_component",
    "dedupe_resolution",
]

INVENTORY_FIELDS = [
    "source_manifest",
    "source_candidate_id",
    "semantic_sign_id",
    "review_decision",
    "inventory_status",
    "exclusion_reason",
    "crop_path",
    "original_path",
    "crop_sha256",
    "crop_pixel_sha256",
    "perceptual_hash",
    "source_group",
    "source_video_group",
    "source_url",
    "licence_status",
    "source_registry_status",
    "provenance_status",
    "small_object",
    "review_notes",
]

EXCLUSION_FIELDS = [
    *INVENTORY_FIELDS,
    "source_kind",
    "split",
    "dedupe_component",
    "dedupe_resolution",
    "internal_academic_exception_reasons",
    "component_label_count",
    "component_frozen_splits",
]


def default_final_classifier_release_config() -> FinalClassifierReleaseConfig:
    return FinalClassifierReleaseConfig(Path(__file__).resolve().parents[3])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(config: FinalClassifierReleaseConfig, path: Path) -> str:
    return path.resolve().relative_to(config.project_root).as_posix()


def _resolve(config: FinalClassifierReleaseConfig, value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else config.project_root / path


def _value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"true", "1", "yes"}


def is_accepted_contributor_decision(value: str) -> bool:
    """Return whether a final reviewer accepted a Stage C row."""

    decision = value.strip().casefold()
    return decision.startswith("accept") or decision.startswith("manual_accept")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_sha256(image: Image.Image) -> str:
    """Hash decoded RGB pixels and geometry, independent of file encoding."""

    rgb = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}\0".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def perceptual_hash(image: Image.Image) -> str:
    """Return the repository's 64-bit DCT perceptual hash.

    It matches the collector implementation instead of using a lower-resolution
    difference hash, which would over-group visually similar sign designs.
    """

    gray = np.asarray(
        ImageOps.exif_transpose(image).convert("L").resize(
            (32, 32), Image.Resampling.LANCZOS
        )
    )
    coefficients = cv2.dct(gray.astype(np.float32))[:8, :8]
    values = coefficients.flatten()
    median = float(np.median(values[1:]))
    bits = values > median
    return f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _image_evidence(path: Path, declared_sha256: str, *, require_declared: bool) -> ImageEvidence:
    if not path.is_file():
        raise ValueError(f"missing image: {path}")
    actual_sha256 = _sha256_file(path)
    if require_declared and not declared_sha256:
        raise ValueError(f"missing declared SHA-256: {path}")
    if declared_sha256 and actual_sha256 != declared_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            return ImageEvidence(
                file_sha256=actual_sha256,
                pixel_sha256=decoded_pixel_sha256(image),
                perceptual_hash=perceptual_hash(image),
                width=image.width,
                height=image.height,
            )
    except (OSError, ValueError) as error:
        raise ValueError(f"unreadable image: {path}") from error


def _discover_contributor_manifests(config: FinalClassifierReleaseConfig) -> list[Path]:
    directory = config.contributor_manifest_dir
    assert directory is not None
    discovered = sorted(directory.glob("stage_c_*.csv"))
    included = [
        path
        for path in discovered
        if not any(marker in path.stem for marker in ("_leads", "_review", "no_minimum"))
    ]
    specs = config.contributor_manifest_specs or CONTRIBUTOR_MANIFEST_SPECS
    unknown = [path.name for path in included if path.name not in specs]
    if unknown:
        raise ValueError(
            "Finalized Stage C manifest(s) need an explicit semantic mapping: "
            + ", ".join(unknown)
        )
    missing = sorted(set(specs) - {path.name for path in included})
    if missing:
        raise ValueError("Expected contributor manifest(s) missing: " + ", ".join(missing))
    return included


def _specific_group(value: str) -> str:
    normalized = value.strip()
    lowered = normalized.casefold()
    if not normalized or any(marker in lowered for marker in GENERIC_GROUP_MARKERS):
        return ""
    return normalized


def _connection_keys(candidate: Candidate) -> tuple[str, ...]:
    keys: list[str] = []
    if candidate.source_kind in {"frozen", "v2"}:
        group = _specific_group(candidate.source_group)
        if group:
            keys.append(f"existing:{group}")
    else:
        source_group = _specific_group(candidate.source_group)
        video_group = _specific_group(candidate.source_video_group)
        if source_group:
            keys.append(f"contributor-source:{source_group}")
        if video_group:
            keys.append(f"contributor-video:{video_group}")
    return tuple(keys)


def _candidate_from_v2_row(
    config: FinalClassifierReleaseConfig,
    row: dict[str, str],
    *,
    source_kind: str,
) -> Candidate:
    v2_root = config.v2_root
    assert v2_root is not None
    asset = _resolve(config, row["dataset_image_path"])
    evidence = _image_evidence(asset, row.get("crop_sha256", ""), require_declared=True)
    if not asset.resolve().is_relative_to(v2_root.resolve()):
        raise ValueError(f"v2 dataset path is outside its release root: {asset}")
    source_group = _value(row, "leakage_group", "source_group", "dedupe_key")
    licence_notes = row.get("license_notes", "")
    source_manifest_path = config.v2_manifest if source_kind == "v2" else config.frozen_manifest
    assert source_manifest_path is not None
    return Candidate(
        identifier=f"{source_kind}:{row['sample_id']}",
        source_priority=0 if source_kind == "frozen" else 1,
        source_kind=source_kind,
        semantic_sign_id=row["semantic_sign_id"],
        split=row["split"],
        crop_path=asset,
        source_crop_path=row.get("source_crop_path", row["dataset_image_path"]),
        original_path=row.get("source_crop_path", row["dataset_image_path"]),
        original_sha256=row.get("crop_sha256", ""),
        crop_sha256=evidence.file_sha256,
        crop_pixel_sha256=evidence.pixel_sha256,
        perceptual_hash=evidence.perceptual_hash,
        crop_width=evidence.width,
        crop_height=evidence.height,
        source_manifest=_relative(config, source_manifest_path),
        source_candidate_id=row["sample_id"],
        source_group=source_group,
        source_video_group="",
        source_dataset=row.get("source_dataset", ""),
        source_url=row.get("source_url", ""),
        licence_status="frozen_release_provenance",
        source_registry_status="existing_registry",
        provenance_status="frozen_release_verified",
        review_decision=row.get("review_decision", "accept") or "accept",
        small_object=False,
        name_en=row.get("name_en", ""),
        priority=row.get("priority", ""),
        required_for=row.get("required_for", ""),
        mapping_evidence=row.get("mapping_evidence", ""),
        license_notes=licence_notes,
        review_notes=row.get("review_notes", ""),
    )


def _contributor_inventory_row(
    config: FinalClassifierReleaseConfig,
    manifest: Path,
    row: dict[str, str],
    target: str,
    *,
    status: str,
    exclusion_reason: str = "",
    candidate: Candidate | None = None,
) -> dict[str, object]:
    return {
        "source_manifest": _relative(config, manifest),
        "source_candidate_id": _value(row, "candidate_id", "record_id"),
        "semantic_sign_id": target,
        "review_decision": _value(row, "final_review_decision", "review_decision"),
        "inventory_status": status,
        "exclusion_reason": exclusion_reason,
        "crop_path": candidate.source_crop_path if candidate else row.get("crop_path", ""),
        "original_path": candidate.original_path if candidate else row.get("original_path", ""),
        "crop_sha256": candidate.crop_sha256 if candidate else row.get("crop_sha256", ""),
        "crop_pixel_sha256": candidate.crop_pixel_sha256 if candidate else _value(row, "crop_pixel_sha256"),
        "perceptual_hash": candidate.perceptual_hash if candidate else _value(row, "perceptual_hash", "perceptual_hash_dhash"),
        "source_group": candidate.source_group if candidate else row.get("source_group", ""),
        "source_video_group": candidate.source_video_group if candidate else row.get("source_video_group", ""),
        "source_url": candidate.source_url if candidate else _value(row, "source_page_url", "source_url"),
        "licence_status": candidate.licence_status if candidate else row.get("rights_status", ""),
        "source_registry_status": candidate.source_registry_status if candidate else "",
        "provenance_status": candidate.provenance_status if candidate else "",
        "small_object": str(candidate.small_object).lower() if candidate else row.get("small_object", ""),
        "review_notes": candidate.review_notes if candidate else _value(row, "reviewer_notes", "notes"),
    }


def _candidate_from_contributor_row(
    config: FinalClassifierReleaseConfig,
    manifest: Path,
    target: str,
    row: dict[str, str],
) -> Candidate:
    candidate_id = _value(row, "candidate_id", "record_id")
    if not candidate_id:
        raise ValueError("missing candidate ID")
    crop_text = row.get("crop_path", "").strip()
    original_text = row.get("original_path", "").strip()
    if not crop_text or not original_text:
        raise ValueError("missing original or crop path")
    crop_path = _resolve(config, crop_text)
    original_path = _resolve(config, original_text)
    crop_evidence = _image_evidence(crop_path, row.get("crop_sha256", ""), require_declared=True)
    _image_evidence(original_path, row.get("original_sha256", ""), require_declared=True)

    rights_status = row.get("rights_status", "").strip().casefold()
    exception_reasons: list[str] = []
    if rights_status == "stated_license":
        licence_status = "licence_documented"
        registry_status = "row_level_licence_documented"
        provenance_status = "accepted_with_row_level_provenance"
    else:
        licence_status = "internal_academic_exception"
        registry_status = "internal_academic_exception"
        provenance_status = "internal_academic_exception_not_licence_cleared"
        exception_reasons.append("source_rights_not_licence_cleared")
    if not _value(row, "source_page_url", "source_url"):
        exception_reasons.append("source_page_url_missing")
    if not _specific_group(row.get("source_group", "")) and not _specific_group(
        row.get("source_video_group", "")
    ):
        exception_reasons.append("specific_source_or_video_group_missing")
    if exception_reasons and not config.allow_internal_academic_exceptions:
        raise ValueError("internal academic policy exception is not enabled")

    return Candidate(
        identifier=f"contributor:{manifest.name}:{candidate_id}",
        source_priority=2,
        source_kind="contributor",
        semantic_sign_id=target,
        split="train",
        crop_path=crop_path,
        source_crop_path=_relative(config, crop_path),
        original_path=_relative(config, original_path),
        original_sha256=row.get("original_sha256", ""),
        crop_sha256=crop_evidence.file_sha256,
        crop_pixel_sha256=crop_evidence.pixel_sha256,
        perceptual_hash=crop_evidence.perceptual_hash,
        crop_width=crop_evidence.width,
        crop_height=crop_evidence.height,
        source_manifest=_relative(config, manifest),
        source_candidate_id=candidate_id,
        source_group=row.get("source_group", "").strip(),
        source_video_group=row.get("source_video_group", "").strip(),
        source_dataset=_value(
            row, "source_original_dataset", "website_dataset_name", "source_type"
        ),
        source_url=_value(row, "source_page_url", "source_url", "direct_image_url"),
        licence_status=licence_status,
        source_registry_status=registry_status,
        provenance_status=provenance_status,
        review_decision=_value(row, "final_review_decision", "review_decision"),
        small_object=_truthy(row.get("small_object", "")),
        name_en="",
        priority="",
        required_for="",
        mapping_evidence=(
            f"Explicit Phase A manifest mapping to {target}; "
            + _value(row, "target_class_evidence", "website_displayed_class_label")
        ).strip(),
        license_notes=_value(row, "stated_license", "rights_status"),
        review_notes=_value(row, "reviewer_notes", "notes"),
        internal_exception_reasons=exception_reasons,
    )


def _load_contributors(
    config: FinalClassifierReleaseConfig,
) -> tuple[list[Candidate], list[dict[str, object]], list[dict[str, object]]]:
    candidates: list[Candidate] = []
    inventory: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for manifest in _discover_contributor_manifests(config):
        specs = config.contributor_manifest_specs or CONTRIBUTOR_MANIFEST_SPECS
        target = specs[manifest.name].semantic_sign_id
        for row in _read_csv(manifest):
            decision = _value(row, "final_review_decision", "review_decision")
            if not is_accepted_contributor_decision(decision):
                inventory.append(
                    _contributor_inventory_row(
                        config,
                        manifest,
                        row,
                        target,
                        status="excluded",
                        exclusion_reason="review_decision_not_accepted",
                    )
                )
                continue
            try:
                candidate = _candidate_from_contributor_row(config, manifest, target, row)
            except ValueError as error:
                inventory_row = _contributor_inventory_row(
                    config,
                    manifest,
                    row,
                    target,
                    status="excluded",
                    exclusion_reason=str(error),
                )
                inventory.append(inventory_row)
                exclusions.append(inventory_row)
                continue
            candidates.append(candidate)
            inventory.append(
                _contributor_inventory_row(
                    config, manifest, row, target, status="accepted_pending_global_audit", candidate=candidate
                )
            )
    return candidates, inventory, exclusions


def _build_duplicate_components(
    candidates: list[Candidate], threshold: int
) -> dict[str, list[Candidate]]:
    union_find = UnionFind(candidate.identifier for candidate in candidates)
    hash_index: dict[str, str] = {}
    pixel_index: dict[str, str] = {}
    group_index: dict[str, str] = {}
    for candidate in candidates:
        for index, value in ((hash_index, candidate.crop_sha256), (pixel_index, candidate.crop_pixel_sha256)):
            prior = index.setdefault(value, candidate.identifier)
            union_find.union(prior, candidate.identifier)
        for group in _connection_keys(candidate):
            prior = group_index.setdefault(group, candidate.identifier)
            union_find.union(prior, candidate.identifier)

    by_label: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_label[candidate.semantic_sign_id].append(candidate)
    for label_candidates in by_label.values():
        sorted_candidates = sorted(label_candidates, key=lambda candidate: candidate.identifier)
        for left_index, left in enumerate(sorted_candidates):
            for right in sorted_candidates[left_index + 1 :]:
                if hamming_distance(left.perceptual_hash, right.perceptual_hash) <= threshold:
                    # Perceptual similarity is an audit signal, not enough on its own
                    # to discard a separately reviewed physical sign. Exact bytes,
                    # decoded pixels, and documented connected groups remain hard links.
                    left.perceptual_near_match_ids.append(right.identifier)
                    right.perceptual_near_match_ids.append(left.identifier)

    components: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        components[union_find.find(candidate.identifier)].append(candidate)
    normalized: dict[str, list[Candidate]] = {}
    for number, (_, members) in enumerate(
        sorted(components.items(), key=lambda item: min(member.identifier for member in item[1])), start=1
    ):
        component_id = f"component_{number:05d}"
        for candidate in members:
            candidate.dedupe_component = component_id
        normalized[component_id] = members
    return normalized


def _candidate_score(candidate: Candidate) -> tuple[int, int, int, str, str]:
    return (
        candidate.source_priority,
        int(candidate.is_internal_exception),
        int(candidate.small_object),
        candidate.source_manifest,
        candidate.source_candidate_id,
    )


def _resolve_duplicate_components(
    components: dict[str, list[Candidate]],
) -> tuple[list[Candidate], list[dict[str, object]], list[dict[str, object]]]:
    retained: list[Candidate] = []
    decisions: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for _component_id, members in components.items():
        frozen = [member for member in members if member.source_kind == "frozen"]
        frozen_splits = {member.split for member in frozen}
        source_labels = {member.semantic_sign_id for member in members}
        if len(frozen_splits) > 1:
            for member in members:
                member.dedupe_resolution = "retained_frozen_cross_split_violation" if member in frozen else "held_cross_split_violation"
            retained.extend(frozen)
        elif frozen:
            for member in frozen:
                member.dedupe_resolution = "retained_frozen"
            retained.extend(frozen)
            for member in members:
                if member in frozen:
                    continue
                member.dedupe_resolution = "held_related_to_frozen_evaluation" if frozen_splits & {"validation", "test"} else "excluded_related_to_frozen_training"
                exclusions.append(_exclusion_row(member))
        else:
            winner = min(members, key=_candidate_score)
            winner.dedupe_resolution = "retained_component_winner"
            retained.append(winner)
            for member in members:
                if member is winner:
                    continue
                member.dedupe_resolution = f"excluded_component_winner:{winner.identifier}"
                exclusions.append(_exclusion_row(member))
        for member in members:
            decision = _decision_row(member)
            decision["component_label_count"] = len(source_labels)
            decision["component_frozen_splits"] = ";".join(sorted(frozen_splits))
            decisions.append(decision)
    return retained, decisions, exclusions


def _decision_row(candidate: Candidate) -> dict[str, object]:
    return {
        "dedupe_component": candidate.dedupe_component,
        "source_kind": candidate.source_kind,
        "source_manifest": candidate.source_manifest,
        "source_candidate_id": candidate.source_candidate_id,
        "semantic_sign_id": candidate.semantic_sign_id,
        "split": candidate.split,
        "crop_sha256": candidate.crop_sha256,
        "crop_pixel_sha256": candidate.crop_pixel_sha256,
        "perceptual_hash": candidate.perceptual_hash,
        "perceptual_near_match_ids": ";".join(candidate.perceptual_near_match_ids),
        "source_group": candidate.source_group,
        "source_video_group": candidate.source_video_group,
        "licence_status": candidate.licence_status,
        "dedupe_resolution": candidate.dedupe_resolution,
        "internal_academic_exception_reasons": ";".join(candidate.internal_exception_reasons),
    }


def _exclusion_row(candidate: Candidate) -> dict[str, object]:
    row = _decision_row(candidate)
    row["exclusion_reason"] = candidate.dedupe_resolution
    return row


def _canonical_row(config: FinalClassifierReleaseConfig, candidate: Candidate) -> dict[str, object]:
    assert candidate.release_path is not None
    return {
        "release_id": config.release_id,
        "sample_id": candidate.source_candidate_id,
        "split": candidate.split,
        "semantic_sign_id": candidate.semantic_sign_id,
        "name_en": candidate.name_en,
        "priority": candidate.priority,
        "required_for": candidate.required_for,
        "source_manifest": candidate.source_manifest,
        "source_candidate_id": candidate.source_candidate_id,
        "source_crop_path": candidate.source_crop_path,
        "original_path": candidate.original_path,
        "dataset_image_path": _relative(config, candidate.release_path),
        "original_sha256": candidate.original_sha256,
        "crop_sha256": candidate.crop_sha256,
        "crop_pixel_sha256": candidate.crop_pixel_sha256,
        "perceptual_hash": candidate.perceptual_hash,
        "perceptual_near_match_ids": ";".join(candidate.perceptual_near_match_ids),
        "crop_width": candidate.crop_width,
        "crop_height": candidate.crop_height,
        "source_group": candidate.source_group,
        "source_video_group": candidate.source_video_group,
        "source_dataset": candidate.source_dataset,
        "source_url": candidate.source_url,
        "licence_status": candidate.licence_status,
        "source_registry_status": candidate.source_registry_status,
        "provenance_status": candidate.provenance_status,
        "internal_academic_exception_reasons": ";".join(candidate.internal_exception_reasons),
        "review_decision": candidate.review_decision,
        "mapping_evidence": candidate.mapping_evidence,
        "license_notes": candidate.license_notes,
        "review_notes": candidate.review_notes,
        "small_object": str(candidate.small_object).lower(),
        "dedupe_component": candidate.dedupe_component,
        "dedupe_resolution": candidate.dedupe_resolution,
    }


def _safe_prepare_output(path: Path, root: Path, *, overwrite: bool) -> None:
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ValueError(f"refusing to write outside project root: {path}")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_candidate(config: FinalClassifierReleaseConfig, candidate: Candidate) -> None:
    output_root = config.output_root
    assert output_root is not None
    if candidate.source_kind in {"frozen", "v2"}:
        v2_root = config.v2_root
        assert v2_root is not None
        relative = candidate.crop_path.resolve().relative_to(v2_root.resolve())
        destination = output_root / relative
    else:
        suffix = candidate.crop_path.suffix.lower() or ".jpg"
        destination = (
            output_root
            / "train"
            / candidate.semantic_sign_id
            / f"contributor_{candidate.source_candidate_id}_{candidate.crop_sha256[:10]}{suffix}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"release output collision: {destination}")
    shutil.copy2(candidate.crop_path, destination)
    if _sha256_file(destination) != candidate.crop_sha256:
        raise ValueError(f"copy checksum mismatch: {destination}")
    candidate.release_path = destination


def _verify_v2_against_frozen(
    frozen_rows: list[dict[str, str]], v2_rows: list[dict[str, str]]
) -> tuple[set[str], set[str]]:
    frozen_by_sample = {row["sample_id"]: row for row in frozen_rows}
    if len(frozen_by_sample) != len(frozen_rows):
        raise ValueError("frozen release has duplicate sample IDs")
    v2_by_sample = {row["sample_id"]: row for row in v2_rows}
    missing = sorted(set(frozen_by_sample) - set(v2_by_sample))
    if missing:
        raise ValueError(f"v2 release is missing frozen samples: {missing[:5]}")
    for sample_id, frozen in frozen_by_sample.items():
        v2 = v2_by_sample[sample_id]
        for compared_field in ("split", "semantic_sign_id", "crop_sha256"):
            if frozen.get(compared_field, "") != v2.get(compared_field, ""):
                raise ValueError(f"v2 changed frozen {compared_field} for {sample_id}")
    labels = {row["semantic_sign_id"] for row in frozen_rows}
    if len(labels) != 78:
        raise ValueError(f"frozen release must contain exactly 78 labels, found {len(labels)}")
    return set(frozen_by_sample), labels


def _coverage_rows(
    config: FinalClassifierReleaseConfig, retained: list[Candidate]
) -> tuple[list[dict[str, object]], set[str]]:
    tracker_path = config.progress_tracker
    assert tracker_path is not None
    tracker = _read_csv(tracker_path)
    counts = Counter(candidate.semantic_sign_id for candidate in retained)
    exceptions = Counter(
        candidate.semantic_sign_id for candidate in retained if candidate.is_internal_exception
    )
    rows: list[dict[str, object]] = []
    below_minimum: set[str] = set()
    for row in tracker:
        label = row["semantic_sign_id"]
        minimum = int(row.get("minimum_clean_crops", "0") or 0)
        total = counts[label]
        gap = max(0, minimum - total)
        if row.get("priority") == "must" and gap:
            below_minimum.add(label)
        rows.append(
            {
                "semantic_sign_id": label,
                "priority": row.get("priority", ""),
                "minimum_clean_crops": minimum,
                "v3_total_count": total,
                "v3_internal_academic_exception_count": exceptions[label],
                "gap_to_minimum": gap,
                "meets_minimum": str(total >= minimum).lower(),
            }
        )
    return rows, below_minimum


def _update_progress_tracker(
    config: FinalClassifierReleaseConfig,
    coverage: list[dict[str, object]],
    *,
    overwrite: bool,
) -> None:
    tracker_path = config.progress_tracker
    assert tracker_path is not None
    original = _read_csv(tracker_path)
    coverage_by_label = {str(row["semantic_sign_id"]): row for row in coverage}
    additions = [
        "last_release_id",
        "production_78_v3_total_count",
        "production_78_v3_internal_exception_count",
    ]
    fields = list(original[0]) if original else []
    fields.extend(column for column in additions if column not in fields)
    updated: list[dict[str, object]] = []
    for row in original:
        result: dict[str, object] = dict(row)
        item = coverage_by_label[row["semantic_sign_id"]]
        total = int(str(item["v3_total_count"]))
        minimum = int(str(item["minimum_clean_crops"]))
        result["last_release_id"] = config.release_id
        result["production_78_v3_total_count"] = total
        result["production_78_v3_internal_exception_count"] = item[
            "v3_internal_academic_exception_count"
        ]
        result["gap_to_minimum"] = max(0, minimum - total)
        if row.get("priority") == "must":
            result["status"] = "v3_candidate_with_policy_exceptions"
            result["next_action"] = (
                "Resolve recorded Phase A provenance/licence exceptions before any external "
                "data or model release; locked validation/test remain unchanged."
            )
        updated.append(result)
    _write_csv(tracker_path, updated, fields)


def build_final_classifier_release(
    config: FinalClassifierReleaseConfig | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create the auditable v3 classifier release and return its audit report.

    Invalid contributor rows are retained in audit outputs but are not copied into
    the release.  A corrupt frozen/v2 input is a hard stop because it would make
    preservation of the locked evaluation data impossible to prove.
    """

    config = config or default_final_classifier_release_config()
    frozen_manifest = config.frozen_manifest
    v2_manifest = config.v2_manifest
    output_root = config.output_root
    output_manifest = config.output_manifest
    audit_root = config.audit_root
    assert frozen_manifest is not None and v2_manifest is not None
    assert output_root is not None and output_manifest is not None and audit_root is not None

    if output_manifest.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_manifest}")
    if audit_root.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {audit_root}")
    frozen_rows = _read_csv(frozen_manifest)
    v2_rows = _read_csv(v2_manifest)
    frozen_sample_ids, labels = _verify_v2_against_frozen(frozen_rows, v2_rows)

    existing: list[Candidate] = []
    for row in v2_rows:
        kind = "frozen" if row["sample_id"] in frozen_sample_ids else "v2"
        if kind == "v2" and row.get("split") != "train":
            raise ValueError(f"v2-added sample is not train-only: {row['sample_id']}")
        existing.append(_candidate_from_v2_row(config, row, source_kind=kind))
    contributors, inventory, initial_exclusions = _load_contributors(config)
    invalid_labels = sorted({candidate.semantic_sign_id for candidate in contributors} - labels)
    if invalid_labels:
        raise ValueError("contributors map outside frozen 78-label ontology: " + ", ".join(invalid_labels))

    all_candidates = existing + contributors
    components = _build_duplicate_components(all_candidates, config.perceptual_hamming_threshold)
    retained, duplicate_decisions, duplicate_exclusions = _resolve_duplicate_components(components)
    retained.sort(key=lambda candidate: (candidate.split, candidate.semantic_sign_id, candidate.source_candidate_id))

    retained_frozen = [candidate for candidate in retained if candidate.source_kind == "frozen"]
    expected_frozen_eval = [row for row in frozen_rows if row["split"] in {"validation", "test"}]
    actual_frozen_eval = [candidate for candidate in retained_frozen if candidate.split in {"validation", "test"}]
    frozen_eval_preserved = {
        (row["sample_id"], row["split"], row["crop_sha256"]) for row in expected_frozen_eval
    } == {
        (candidate.source_candidate_id, candidate.split, candidate.crop_sha256)
        for candidate in actual_frozen_eval
    }
    if not frozen_eval_preserved:
        raise ValueError("global dedupe attempted to change frozen validation/test membership")
    if any(candidate.source_kind == "contributor" and candidate.split != "train" for candidate in retained):
        raise ValueError("a contributor record was assigned outside train")

    coverage, below_minimum = _coverage_rows(config, retained)
    component_cross_split = [
        decision
        for decision in duplicate_decisions
        if decision["component_frozen_splits"] == "test;validation"
        or decision["component_frozen_splits"] == "train;validation"
        or decision["component_frozen_splits"] == "test;train"
        or decision["component_frozen_splits"] == "test;train;validation"
    ]
    policy_exceptions = [
        candidate for candidate in retained if candidate.is_internal_exception
    ]

    _safe_prepare_output(output_root, config.project_root, overwrite=overwrite)
    _safe_prepare_output(audit_root, config.project_root, overwrite=overwrite)
    for candidate in retained:
        _copy_candidate(config, candidate)
    canonical_rows = [_canonical_row(config, candidate) for candidate in retained]
    _write_csv(output_manifest, canonical_rows, CANONICAL_FIELDS)
    labels_path = output_root / "labels.json"
    _write_json(labels_path, sorted(labels))

    for row in inventory:
        matching = next(
            (
                candidate
                for candidate in contributors
                if candidate.source_manifest == row["source_manifest"]
                and candidate.source_candidate_id == row["source_candidate_id"]
            ),
            None,
        )
        if matching is not None:
            row["inventory_status"] = (
                "retained" if matching in retained else "excluded_after_global_audit"
            )
            row["exclusion_reason"] = "" if matching in retained else matching.dedupe_resolution
    all_exclusions = [*initial_exclusions, *duplicate_exclusions]
    _write_csv(audit_root / "contributor_inventory.csv", inventory, INVENTORY_FIELDS)
    duplicate_fields = [*_decision_row(retained[0]),
        "component_label_count",
        "component_frozen_splits",
    ]
    _write_csv(audit_root / "duplicate_decisions.csv", duplicate_decisions, duplicate_fields)
    _write_csv(audit_root / "exclusions.csv", all_exclusions, EXCLUSION_FIELDS)
    coverage_fields = list(coverage[0]) if coverage else []
    _write_csv(audit_root / "per_class_coverage.csv", coverage, coverage_fields)

    normal_phase_a_gate_passed = (
        frozen_eval_preserved
        and not component_cross_split
        and not initial_exclusions
        and not below_minimum
    )
    coverage_exception_approved = bool(
        below_minimum and config.allow_must_have_coverage_exception
    )
    phase_a_training_ready = (
        frozen_eval_preserved
        and not component_cross_split
        and not initial_exclusions
        and (not below_minimum or coverage_exception_approved)
    )
    metadata = {
        "dataset_id": config.release_id,
        "base_release": FROZEN_RELEASE_ID,
        "v2_release": V2_RELEASE_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "labels": len(labels),
        "split_counts": dict(Counter(candidate.split for candidate in retained)),
        "total_images": len(retained),
        "release_status": "production_candidate_with_policy_exceptions",
        # This release is reviewed and eligible for the explicitly approved
        # internal Phase-B workflow.  It is intentionally *not* equivalent to
        # a licence-cleared public release.
        "annotation_status": "approved_with_internal_academic_exception",
        "coursework_images_included": 0,
        "phase_a_gate_status": (
            "passed"
            if normal_phase_a_gate_passed
            else "passed_with_approved_coverage_exception"
            if phase_a_training_ready
            else "blocked"
        ),
        "phase_b_authorized": phase_a_training_ready,
        "internal_academic_only": True,
        "internal_runtime_promotion_eligible": True,
        "external_data_or_model_release_allowed": False,
        "dvc_remote_push_allowed": False,
        "publication_prohibited": True,
        "perceptual_hamming_threshold": config.perceptual_hamming_threshold,
        "frozen_validation_and_test_preserved": True,
        "contributor_split": "train_only",
        "source_registry_exception_status": "internal_academic_exception",
        "coverage_exception": {
            "approved": coverage_exception_approved,
            "classes_below_minimum": sorted(below_minimum),
            "owner_note": config.coverage_exception_note,
        },
    }
    _write_json(output_root / "dataset_metadata.json", metadata)
    _update_progress_tracker(config, coverage, overwrite=overwrite)

    audit: dict[str, Any] = {
        "schema_version": "2.0",
        **metadata,
        "inputs": {
            "frozen_manifest": _relative(config, frozen_manifest),
            "v2_manifest": _relative(config, v2_manifest),
            "contributor_manifests": [
                _relative(config, path) for path in _discover_contributor_manifests(config)
            ],
        },
        "gates": {
            "exactly_78_labels": len(labels) == 78,
            "frozen_validation_and_test_preserved": frozen_eval_preserved,
            "contributors_train_only": True,
            "must_have_coverage": not below_minimum,
            "must_have_coverage_exception_approved": coverage_exception_approved,
            "phase_a_training_ready": phase_a_training_ready,
            "frozen_cross_split_component_count": len(component_cross_split),
            "invalid_contributor_rows": len(initial_exclusions),
        },
        "counts": {
            "frozen_rows": len(frozen_rows),
            "v2_rows": len(v2_rows),
            "accepted_contributor_rows_before_global_audit": len(contributors),
            "retained_contributor_rows": sum(
                candidate.source_kind == "contributor" for candidate in retained
            ),
            "internal_academic_exception_rows_retained": len(policy_exceptions),
            "perceptual_near_match_pairs": sum(
                len(candidate.perceptual_near_match_ids) for candidate in all_candidates
            )
            // 2,
            "exclusions": len(all_exclusions),
        },
        "must_have_coverage_gaps": sorted(below_minimum),
        "artifacts": {
            "canonical_manifest": _relative(config, output_manifest),
            "dataset_root": _relative(config, output_root),
            "labels": _relative(config, labels_path),
            "inventory": _relative(config, audit_root / "contributor_inventory.csv"),
            "duplicates": _relative(config, audit_root / "duplicate_decisions.csv"),
            "exclusions": _relative(config, audit_root / "exclusions.csv"),
            "coverage": _relative(config, audit_root / "per_class_coverage.csv"),
        },
    }
    _write_json(audit_root / "release_audit.json", audit)
    return audit
