"""Build the assignment-only Phase C detector release.

The teammate payload is treated as an immutable, quarantined input.  This
module deliberately does not use its supplied split assignment: it verifies
the multipart archives, validates the decoded YOLO data, produces review
queues, and creates a fresh one-class data release only after those queues
have been decided.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps

from roadsign_assist.paths import PROJECT_ROOT

RELEASE_ID = "detector_production_assignment_v1_20260829"
SEED = 2513
NEGATIVE_TARGET = 120
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
HARD_REVIEW_FLAGS = {"tiny_box", "large_box", "extreme_aspect_ratio"}
CANONICAL_FIELDS = [
    "sample_id",
    "split",
    "source_kind",
    "source_id",
    "source_image_id",
    "layout_root_id",
    "source_path",
    "dataset_image_path",
    "dataset_label_path",
    "sha256",
    "pixel_sha256",
    "dhash64",
    "perceptual_hash",
    "group_id",
    "bbox_count",
    "is_negative",
    "annotation_status",
    "review_decision",
    "review_reasons",
    "licence_status",
    "use_policy",
]
REVIEW_FIELDS = [
    "sample_id",
    "source_kind",
    "source_path",
    "layout_root_id",
    "source_image_id",
    "review_reasons",
    "review_decision",
    "reviewer_notes",
]


class ReviewRequiredError(RuntimeError):
    """Raised after reproducible review queues have been generated."""


@dataclass(frozen=True)
class DetectorReleaseConfig:
    project_root: Path = PROJECT_ROOT
    release_id: str = RELEASE_ID
    bundle_root: Path = Path(r"C:\MiniProject-data-audit\Ultimate_Datasets_v1_DVC_Transfer_96MiB")
    quarantine_root: Path = Path(
        r"C:\MiniProject-data-audit\_validation\phase_c_detector_assignment_v1"
    )
    emtd_root: Path = Path("data/processed/emtd_detection")
    negative_root: Path = Path("data/raw/phase_c_no_sign_frames")
    output_root: Path = Path("data/processed") / RELEASE_ID
    output_manifest: Path = Path("data/manifests") / f"{RELEASE_ID}.csv"
    audit_root: Path = Path("outputs/audit") / RELEASE_ID
    review_root: Path = Path("data/manifests") / f"{RELEASE_ID}_reviews"
    seed: int = SEED
    negative_target: int = NEGATIVE_TARGET
    reuse_verified_quarantine: bool = False

    def __post_init__(self) -> None:
        root = self.project_root.resolve()
        object.__setattr__(self, "project_root", root)
        object.__setattr__(self, "bundle_root", self.bundle_root.resolve())
        object.__setattr__(self, "quarantine_root", self.quarantine_root.resolve())
        def resolve_project_path(value: Path) -> Path:
            return (value if value.is_absolute() else root / value).resolve()

        object.__setattr__(self, "emtd_root", resolve_project_path(self.emtd_root))
        object.__setattr__(self, "negative_root", resolve_project_path(self.negative_root))
        object.__setattr__(self, "output_root", resolve_project_path(self.output_root))
        object.__setattr__(self, "output_manifest", resolve_project_path(self.output_manifest))
        object.__setattr__(self, "audit_root", resolve_project_path(self.audit_root))
        object.__setattr__(self, "review_root", resolve_project_path(self.review_root))


@dataclass
class Candidate:
    sample_id: str
    source_path: Path
    source_kind: str
    source_id: str
    source_image_id: str
    layout_root_id: str
    sha256: str
    pixel_sha256: str
    dhash64: str
    perceptual_hash: str
    width: int
    height: int
    boxes: list[tuple[float, float, float, float]]
    group_id: str
    review_reasons: set[str] = field(default_factory=lambda: set[str]())
    review_decision: str = "auto_accept"
    is_negative: bool = False
    source_relative_path: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_hash(path: Path) -> tuple[str, str, str, int, int]:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = image.size
        pixels = image.tobytes()
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bits = gray[:, 1:] > gray[:, :-1]
        dhash = "".join("1" if bit else "0" for bit in bits.flat)
        small_gray = np.asarray(
            image.convert("L").resize((16, 16), Image.Resampling.LANCZOS), dtype=np.uint8
        )
        coarse_rgb = np.asarray(
            image.resize((16, 16), Image.Resampling.LANCZOS), dtype=np.uint8
        ) // 16
        perceptual_hex = hashlib.sha256(
            np.packbits(small_gray > np.median(small_gray)).tobytes() + coarse_rgb.tobytes()
        ).hexdigest()
    return hashlib.sha256(pixels).hexdigest(), dhash, perceptual_hex, width, height


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_empty_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")
        if not path.is_dir():
            raise ValueError(f"Expected a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _checksum_entries(bundle_root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw in (bundle_root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, relative = raw.split(maxsplit=1)
        entries[relative.strip()] = digest.lower()
    return entries


def verify_bundle_checksums(bundle_root: Path) -> dict[str, Any]:
    """Verify all materialized supplier checksums before archive reassembly."""
    expected = _checksum_entries(bundle_root)
    rows: list[dict[str, str]] = []
    for relative, digest in sorted(expected.items()):
        path = bundle_root / relative
        if not path.is_file():
            rows.append({"path": relative, "status": "missing", "expected": digest, "actual": ""})
            continue
        actual = _sha256(path)
        rows.append(
            {
                "path": relative,
                "status": "pass" if actual == digest else "fail",
                "expected": digest,
                "actual": actual,
            }
        )
    unexpected_missing = [
        row["path"]
        for row in rows
        if row["status"] == "missing"
        and not (row["path"].endswith(".tar") or row["expected"] == hashlib.sha256(b"").hexdigest())
    ]
    failed = [row["path"] for row in rows if row["status"] == "fail"]
    if unexpected_missing or failed:
        raise ValueError(
            "Supplier checksum validation failed: "
            f"missing={unexpected_missing}, mismatched={failed}"
        )
    return {
        "entries": rows,
        "passed_materialized_entries": sum(row["status"] == "pass" for row in rows),
        "expected_missing_entries": [row["path"] for row in rows if row["status"] == "missing"],
    }


def _safe_extract(archive: tarfile.TarFile, target: Path) -> None:
    target_resolved = target.resolve()
    for member in archive.getmembers():
        destination = (target / member.name).resolve()
        if not destination.is_relative_to(target_resolved):
            raise ValueError(f"Archive member escapes quarantine: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Archive links are not allowed: {member.name}")
    archive.extractall(target, filter="data")


def _reuse_verified_reassembly(
    config: DetectorReleaseConfig,
    *,
    checksum: dict[str, Any],
    expected: dict[str, str],
    archive_root: Path,
) -> dict[str, Any]:
    """Validate an existing quarantine without deleting or re-extracting it."""
    audit_path = config.quarantine_root / "reassembly_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing reassembly audit for reusable quarantine: {audit_path}")
    previous = json.loads(audit_path.read_text(encoding="utf-8"))
    previous_archives = {row.get("split"): row for row in previous.get("archives", [])}
    archives: list[dict[str, Any]] = []
    for split, name in (("train", "train.tar"), ("validation", "validation.tar"), ("test", "test.tar")):
        output = config.quarantine_root / "reassembled" / name
        parts = sorted(archive_root.glob(f"{name}.part*"), key=lambda path: path.name)
        expected_digest = expected[f"detector_generic_yolo/{name}"]
        if not output.is_file() or _sha256(output) != expected_digest:
            raise ValueError(f"Reusable quarantine archive is invalid: {output}")
        split_root = config.quarantine_root / "extracted" / split
        if not (split_root / "images").is_dir() or not (split_root / "labels").is_dir():
            raise ValueError(f"Reusable quarantine extraction is incomplete: {split_root}")
        previous_row = previous_archives.get(split, {})
        archives.append(
            {
                "split": split,
                "parts": [part.name for part in parts],
                "sha256": expected_digest,
                "archive_path": output.as_posix(),
                "extraction_root": split_root.as_posix(),
                **({"prior_audit_sha256": previous_row["sha256"]} if previous_row else {}),
            }
        )
    return {"checksum": checksum, "archives": archives}


def reassemble_and_extract_detector_archives(
    config: DetectorReleaseConfig,
    *,
    overwrite: bool,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Concatenate verified part files, validate final tar hashes, then extract."""
    checksum = verify_bundle_checksums(config.bundle_root)
    archive_root = config.bundle_root / "detector_generic_yolo"
    expected = _checksum_entries(config.bundle_root)
    if reuse_existing:
        return _reuse_verified_reassembly(
            config,
            checksum=checksum,
            expected=expected,
            archive_root=archive_root,
        )
    reassembled = config.quarantine_root / "reassembled"
    extracted = config.quarantine_root / "extracted"
    _prepare_empty_directory(config.quarantine_root, overwrite=overwrite)
    reassembled.mkdir()
    extracted.mkdir()
    archives: list[dict[str, Any]] = []
    for split, name in (("train", "train.tar"), ("validation", "validation.tar"), ("test", "test.tar")):
        parts = sorted(archive_root.glob(f"{name}.part*"), key=lambda path: path.name)
        if not parts:
            raise FileNotFoundError(f"No archive parts found for {name}")
        output = reassembled / name
        with output.open("wb") as destination:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        actual = _sha256(output)
        expected_digest = expected[f"detector_generic_yolo/{name}"]
        if actual != expected_digest:
            raise ValueError(f"Reassembled {name} SHA-256 mismatch: {actual} != {expected_digest}")
        split_root = extracted / split
        split_root.mkdir()
        with tarfile.open(output, mode="r") as archive:
            _safe_extract(archive, split_root)
        archives.append(
            {
                "split": split,
                "parts": [part.name for part in parts],
                "sha256": actual,
                "archive_path": output.as_posix(),
                "extraction_root": split_root.as_posix(),
            }
        )
    report = {"checksum": checksum, "archives": archives}
    _write_json(config.quarantine_root / "reassembly_audit.json", report)
    return report


def _parse_yolo(path: Path, *, require_boxes: bool) -> list[tuple[float, float, float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number} must contain five YOLO fields")
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} has invalid YOLO values") from exc
        if class_id < 0 or not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"{path}:{line_number} has invalid class or non-finite geometry")
        if width <= 0 or height <= 0 or x - width / 2 < 0 or y - height / 2 < 0 or x + width / 2 > 1 or y + height / 2 > 1:
            raise ValueError(f"{path}:{line_number} box is out of normalized bounds")
        boxes.append((x, y, width, height))
    if require_boxes and not boxes:
        raise ValueError(f"Positive image has an empty label file: {path}")
    return boxes


def _label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    return Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _teammate_indexes(bundle_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    inventory = _load_jsonl(bundle_root / "manifests/canonical_samples.jsonl")
    by_hash = {
        str(row["image_sha256"]): row
        for row in inventory
        if row.get("task") == "detector" and row.get("image_sha256")
    }
    groups = {
        str(row["source_image_id"]): str(row["group"])
        for row in _load_jsonl(bundle_root / "manifests/release_splits.jsonl")
        if row.get("task") == "detector" and row.get("source_image_id") and row.get("group")
    }
    return by_hash, groups


def _review_reasons(boxes: list[tuple[float, float, float, float]]) -> set[str]:
    reasons: set[str] = set()
    for _, _, width, height in boxes:
        area = width * height
        aspect = max(width / height, height / width)
        if area < 0.001:
            reasons.add("tiny_box")
        if area > 0.50:
            reasons.add("large_box")
        if aspect > 8:
            reasons.add("extreme_aspect_ratio")
    return reasons


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big") / float(2**64)


def _select_stratified_review(candidates: list[Candidate], seed: int) -> set[str]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.layout_root_id or "unknown"].append(candidate)
    selected: set[str] = set()
    for _layout, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: (_stable_fraction(row.sample_id, seed), row.sample_id))
        selected.update(row.sample_id for row in ordered[: max(1, math.ceil(len(rows) * 0.10))])
    return selected


def _candidate_from_image(
    *,
    image: Path,
    source_kind: str,
    source_id: str,
    source_image_id: str,
    layout_root_id: str,
    group_id: str,
    require_boxes: bool,
    source_relative_path: str,
) -> Candidate:
    sha256 = _sha256(image)
    pixel_sha256, dhash64, perceptual_hash, width, height = _pixel_hash(image)
    boxes = _parse_yolo(_label_for_image(image), require_boxes=require_boxes)
    sample_id = hashlib.sha256(
        f"{source_kind}:{sha256}".encode()
    ).hexdigest()[:24]
    return Candidate(
        sample_id=sample_id,
        source_path=image,
        source_kind=source_kind,
        source_id=source_id,
        source_image_id=source_image_id,
        layout_root_id=layout_root_id,
        sha256=sha256,
        pixel_sha256=pixel_sha256,
        dhash64=dhash64,
        perceptual_hash=perceptual_hash,
        width=width,
        height=height,
        boxes=boxes,
        group_id=group_id,
        review_reasons=_review_reasons(boxes),
        source_relative_path=source_relative_path,
    )


def _scan_teammate(config: DetectorReleaseConfig) -> tuple[list[Candidate], list[dict[str, str]]]:
    hashes, groups = _teammate_indexes(config.bundle_root)
    extracted = config.quarantine_root / "extracted"
    candidates: list[Candidate] = []
    exclusions: list[dict[str, str]] = []
    for image in sorted(path for path in extracted.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        try:
            sha256 = _sha256(image)
            source = hashes.get(sha256)
            if source is None:
                raise ValueError("image hash is absent from teammate canonical detector manifest")
            source_image_id = str(source.get("source_image_id", ""))
            candidate = _candidate_from_image(
                image=image,
                source_kind="teammate",
                source_id=str(source.get("source_id", "owner_drive")),
                source_image_id=source_image_id,
                layout_root_id=str(source.get("layout_root_id", "")),
                group_id=groups.get(source_image_id, f"single:{source_image_id}"),
                require_boxes=True,
                source_relative_path=image.relative_to(extracted).as_posix(),
            )
            candidates.append(candidate)
        except (OSError, ValueError) as exc:
            exclusions.append({"source_path": image.as_posix(), "reason": str(exc)})
    return candidates, exclusions


def _scan_emtd(config: DetectorReleaseConfig) -> tuple[list[Candidate], list[dict[str, str]]]:
    root = config.emtd_root
    if not root.is_dir():
        return [], [{"source_path": root.as_posix(), "reason": "EMTD root is unavailable"}]
    candidates: list[Candidate] = []
    exclusions: list[dict[str, str]] = []
    for image in sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        try:
            digest = _sha256(image)
            candidates.append(
                _candidate_from_image(
                    image=image,
                    source_kind="emtd",
                    source_id="emtd_zenodo_1217105",
                    source_image_id=digest[:24],
                    layout_root_id="emtd",
                    group_id=f"emtd:{digest}",
                    require_boxes=True,
                    source_relative_path=image.relative_to(root).as_posix(),
                )
            )
        except (OSError, ValueError) as exc:
            exclusions.append({"source_path": image.as_posix(), "reason": str(exc)})
    return candidates, exclusions


def _scan_negatives(config: DetectorReleaseConfig) -> tuple[list[Candidate], list[dict[str, str]]]:
    root = config.negative_root
    if not root.is_dir():
        return [], [{"source_path": root.as_posix(), "reason": "negative frame root is unavailable"}]
    candidates: list[Candidate] = []
    exclusions: list[dict[str, str]] = []
    for image in sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        try:
            sha256 = _sha256(image)
            pixel_sha256, dhash64, perceptual_hash, width, height = _pixel_hash(image)
            sample_id = hashlib.sha256(f"negative:{sha256}".encode()).hexdigest()[:24]
            candidates.append(
                Candidate(
                    sample_id=sample_id,
                    source_path=image,
                    source_kind="negative",
                    source_id="assignment_negative_frames",
                    source_image_id=sha256[:24],
                    layout_root_id="negative",
                    sha256=sha256,
                    pixel_sha256=pixel_sha256,
                    dhash64=dhash64,
                    perceptual_hash=perceptual_hash,
                    width=width,
                    height=height,
                    boxes=[],
                    group_id=f"negative:{sha256}",
                    review_reasons={"verify_no_traffic_sign"},
                    is_negative=True,
                    source_relative_path=image.relative_to(root).as_posix(),
                )
            )
        except OSError as exc:
            exclusions.append({"source_path": image.as_posix(), "reason": str(exc)})
    return candidates, exclusions


def audit_approved_negatives_against_teammate(config: DetectorReleaseConfig) -> dict[str, Any]:
    """Block an approved no-sign frame that collides with a teammate positive.

    EMTD remains unapproved diagnostic data at this point, so it cannot yet be
    treated as a production positive.  It is checked again when the approved
    EMTD merge is attempted during final materialization.
    """
    teammate, teammate_exclusions = _scan_teammate(config)
    negatives, negative_exclusions = _scan_negatives(config)
    file_index: dict[str, list[str]] = defaultdict(list)
    pixel_index: dict[str, list[str]] = defaultdict(list)
    perceptual_index: dict[str, list[str]] = defaultdict(list)
    for candidate in teammate:
        file_index[candidate.sha256].append(candidate.sample_id)
        pixel_index[candidate.pixel_sha256].append(candidate.sample_id)
        perceptual_index[candidate.perceptual_hash].append(candidate.sample_id)
    conflicts: list[dict[str, str]] = []
    for candidate in negatives:
        for kind, index, value in (
            ("file_sha256", file_index, candidate.sha256),
            ("decoded_pixel_sha256", pixel_index, candidate.pixel_sha256),
            ("perceptual_hash", perceptual_index, candidate.perceptual_hash),
        ):
            for positive_sample_id in index.get(value, []):
                conflicts.append(
                    {
                        "negative_sample_id": candidate.sample_id,
                        "positive_sample_id": positive_sample_id,
                        "match_kind": kind,
                    }
                )
    report = {
        "negative_candidate_count": len(negatives),
        "teammate_positive_candidate_count": len(teammate),
        "teammate_exclusions": teammate_exclusions,
        "negative_exclusions": negative_exclusions,
        "conflicts": conflicts,
        "passed": not conflicts,
        "use_policy": "assignment_only_no_redistribution_no_shared_dvc",
    }
    _write_json(config.audit_root / "approved_negative_vs_teammate_audit.json", report)
    if conflicts:
        raise ValueError(
            f"Approved no-sign candidates collide with teammate positives: {len(conflicts)} matches"
        )
    return report


def _read_decisions(path: Path) -> dict[str, str]:
    return {
        sample_id: row.get("review_decision", "").strip().lower()
        for sample_id, row in _read_review_rows(path).items()
    }


def _read_review_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            str(row["sample_id"]): {key: str(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle)
            if row.get("sample_id")
        }


def record_teammate_layout_review(
    review_queue: Path,
    *,
    layout_root_ids: Iterable[str],
    decision: str,
    reviewer_notes: str,
) -> dict[str, object]:
    """Record one owner decision for fully reviewed teammate layout(s).

    The queue remains the canonical audit record.  This deliberately refuses
    to replace an existing decision and writes a recoverable pre-change copy
    beside the queue before updating it.
    """
    normalized_layouts = tuple(sorted({value.strip() for value in layout_root_ids if value.strip()}))
    normalized_decision = decision.strip().casefold()
    if not normalized_layouts:
        raise ValueError("At least one teammate layout root ID is required")
    if normalized_decision not in {"accept", "reject"}:
        raise ValueError(f"Unsupported teammate review decision: {decision}")
    if not reviewer_notes.strip():
        raise ValueError("Reviewer notes are required for a bulk layout decision")

    with review_queue.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or REVIEW_FIELDS)
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
    if set(REVIEW_FIELDS) - set(fields):
        raise ValueError(f"Unexpected teammate review queue fields: {review_queue}")

    selected = [row for row in rows if row.get("layout_root_id") in normalized_layouts]
    if not selected:
        raise ValueError(f"No teammate review rows match: {', '.join(normalized_layouts)}")
    completed = [row for row in selected if row.get("review_decision", "").strip()]
    if completed:
        raise ValueError(
            "Refusing to replace existing teammate review decisions: "
            + ", ".join(row["sample_id"] for row in completed[:5])
        )

    backup_stem = ".".join((*review_queue.stem.split("."), "before_layout_review"))
    backup = review_queue.with_name(f"{backup_stem}.csv")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite review backup: {backup}")
    shutil.copy2(review_queue, backup)
    for row in selected:
        row["review_decision"] = normalized_decision
        row["reviewer_notes"] = reviewer_notes.strip()
    _write_csv(review_queue, rows, fields)

    return {
        "decision": normalized_decision,
        "layout_root_ids": normalized_layouts,
        "updated_count": len(selected),
        "remaining_pending_count": sum(
            not row.get("review_decision", "").strip() for row in rows
        ),
        "backup": backup,
    }


def record_emtd_review(
    review_queue: Path,
    *,
    decision: str,
    reviewer_notes: str,
) -> dict[str, object]:
    """Record one owner decision for every row of the standalone EMTD queue."""
    rows = _read_review_rows(review_queue)
    if not rows:
        raise ValueError(f"EMTD review queue is empty: {review_queue}")
    if {row.get("source_kind", "") for row in rows.values()} != {"emtd"}:
        raise ValueError(f"EMTD review queue contains an unexpected source kind: {review_queue}")
    return record_teammate_layout_review(
        review_queue,
        layout_root_ids={row.get("layout_root_id", "") for row in rows.values()},
        decision=decision,
        reviewer_notes=reviewer_notes,
    )


def _write_review_queue(path: Path, candidates: Iterable[Candidate]) -> None:
    existing = _read_review_rows(path)
    rows = [
        {
            "sample_id": candidate.sample_id,
            "source_kind": candidate.source_kind,
            "source_path": candidate.source_path.as_posix(),
            "layout_root_id": candidate.layout_root_id,
            "source_image_id": candidate.source_image_id,
            "review_reasons": ";".join(sorted(candidate.review_reasons)),
            "review_decision": existing.get(candidate.sample_id, {}).get("review_decision", ""),
            "reviewer_notes": existing.get(candidate.sample_id, {}).get("reviewer_notes", ""),
        }
        for candidate in sorted(candidates, key=lambda item: item.sample_id)
    ]
    _write_csv(path, rows, REVIEW_FIELDS)


def _dedupe(candidates: list[Candidate]) -> tuple[list[Candidate], list[dict[str, str]], list[dict[str, str]]]:
    retained: list[Candidate] = []
    decisions: list[dict[str, str]] = []
    near_matches: list[dict[str, str]] = []
    seen: dict[tuple[str, str], Candidate] = {}
    for candidate in sorted(candidates, key=lambda item: (item.source_kind != "teammate", item.sample_id)):
        key = (candidate.sha256, candidate.pixel_sha256)
        winner = seen.get(key)
        if winner is None:
            seen[key] = candidate
            retained.append(candidate)
            decisions.append(
                {"sample_id": candidate.sample_id, "status": "retained", "reason": "unique"}
            )
        else:
            decisions.append(
                {
                    "sample_id": candidate.sample_id,
                    "status": "excluded",
                    "reason": f"exact_or_decoded_duplicate_of:{winner.sample_id}",
                }
            )
    by_perceptual: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in retained:
        by_perceptual[candidate.perceptual_hash].append(candidate)
    for values in by_perceptual.values():
        if len(values) > 1:
            anchor = values[0]
            for candidate in values[1:]:
                near_matches.append(
                    {"sample_id": candidate.sample_id, "near_match_sample_id": anchor.sample_id}
                )
    return retained, decisions, near_matches


def _assign_splits(
    candidates: list[Candidate],
    seed: int,
    *,
    targets: dict[str, float] | None = None,
) -> dict[str, str]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.group_id].append(candidate)
    targets = targets or {
        "train": len(candidates) * 0.70,
        "validation": len(candidates) * 0.15,
        "test": len(candidates) * 0.15,
    }
    if set(targets) != {"train", "validation", "test"}:
        raise ValueError(f"Unexpected split targets: {sorted(targets)}")
    if not math.isclose(sum(targets.values()), len(candidates), rel_tol=0, abs_tol=1e-6):
        raise ValueError("Split targets must total the number of positive candidates")
    counts = dict.fromkeys(targets, 0)
    assignments: dict[str, str] = {}
    ordered = sorted(
        groups,
        key=lambda group: (-len(groups[group]), _stable_fraction(group, seed), group),
    )
    for group in ordered:
        size = len(groups[group])
        split = min(
            targets,
            key=lambda name: (
                sum(
                    (counts[other] + (size if other == name else 0) - targets[other]) ** 2
                    for other in targets
                ),
                _stable_fraction(f"{group}:{name}", seed),
            ),
        )
        assignments[group] = split
        counts[split] += size
    return assignments


def _copy_candidate(candidate: Candidate, split: str, output_root: Path) -> tuple[Path, Path]:
    suffix = candidate.source_path.suffix.lower()
    image_path = output_root / "images" / split / f"{candidate.sample_id}{suffix}"
    label_path = output_root / "labels" / split / f"{candidate.sample_id}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate.source_path, image_path)
    lines = [f"0 {x:.8f} {y:.8f} {width:.8f} {height:.8f}" for x, y, width, height in candidate.boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
    return image_path, label_path


def _candidate_manifest_row(config: DetectorReleaseConfig, candidate: Candidate, split: str, image: Path, label: Path) -> dict[str, str]:
    return {
        "sample_id": candidate.sample_id,
        "split": split,
        "source_kind": candidate.source_kind,
        "source_id": candidate.source_id,
        "source_image_id": candidate.source_image_id,
        "layout_root_id": candidate.layout_root_id,
        "source_path": candidate.source_path.as_posix(),
        "dataset_image_path": image.relative_to(config.project_root).as_posix(),
        "dataset_label_path": label.relative_to(config.project_root).as_posix(),
        "sha256": candidate.sha256,
        "pixel_sha256": candidate.pixel_sha256,
        "dhash64": candidate.dhash64,
        "perceptual_hash": candidate.perceptual_hash,
        "group_id": candidate.group_id,
        "bbox_count": str(len(candidate.boxes)),
        "is_negative": str(candidate.is_negative).lower(),
        "annotation_status": "approved",
        "review_decision": candidate.review_decision,
        "review_reasons": ";".join(sorted(candidate.review_reasons)),
        "licence_status": "assignment_only_waiver",
        "use_policy": "assignment_only_no_redistribution_no_shared_dvc",
    }


def build_final_detector_release(config: DetectorReleaseConfig | None = None, *, overwrite: bool = False) -> dict[str, Any]:
    """Build the final detector release, or create queues and request review."""
    config = config or DetectorReleaseConfig()
    reassembly = reassemble_and_extract_detector_archives(
        config,
        overwrite=overwrite,
        reuse_existing=config.reuse_verified_quarantine,
    )
    teammate, teammate_exclusions = _scan_teammate(config)
    emtd, emtd_exclusions = _scan_emtd(config)
    negatives, negative_exclusions = _scan_negatives(config)

    generic_review_ids = _select_stratified_review(teammate, config.seed)
    generic_review_ids.update(
        candidate.sample_id
        for candidate in teammate
        if candidate.review_reasons & HARD_REVIEW_FLAGS
    )
    generic_required = [candidate for candidate in teammate if candidate.sample_id in generic_review_ids]
    generic_queue = config.review_root / "teammate_visual_review.csv"
    emtd_queue = config.review_root / "emtd_box_review.csv"
    negative_queue = config.review_root / "negative_no_sign_review.csv"
    decisions = {
        "generic": _read_decisions(generic_queue),
        "emtd": _read_decisions(emtd_queue),
        "negative": _read_decisions(negative_queue),
    }
    accepted_values = {"accept", "accepted", "approve", "approved"}
    rejected_values = {"reject", "rejected"}
    valid_decisions = accepted_values | rejected_values
    invalid_decisions = {
        kind: sorted({value for value in rows.values() if value and value not in valid_decisions})
        for kind, rows in decisions.items()
    }
    if any(values for values in invalid_decisions.values()):
        raise ValueError(f"Invalid review decisions: {invalid_decisions}")
    missing_review = (
        any(decisions["generic"].get(candidate.sample_id) not in valid_decisions for candidate in generic_required)
        or any(decisions["emtd"].get(candidate.sample_id) not in valid_decisions for candidate in emtd)
        or any(decisions["negative"].get(candidate.sample_id) not in valid_decisions for candidate in negatives)
    )
    if missing_review:
        _write_review_queue(generic_queue, generic_required)
        _write_review_queue(emtd_queue, emtd)
        _write_review_queue(negative_queue, negatives)
        _write_json(
            config.review_root / "review_requirements.json",
            {
                "generic_samples_requiring_review": len(generic_required),
                "emtd_images_requiring_review": len(emtd),
                "negative_frames_requiring_review": len(negatives),
                "negative_target": config.negative_target,
                "accepted_values": ["accept", "accepted", "approve", "approved"],
                "rejected_values": ["reject", "rejected"],
            },
        )
        raise ReviewRequiredError(
            "Phase C review queues created. Record a review_decision for every queued row, "
            "then rerun with --overwrite."
        )

    for candidate in generic_required:
        candidate.review_decision = decisions["generic"][candidate.sample_id]
    for candidate in emtd:
        candidate.review_decision = decisions["emtd"][candidate.sample_id]
    for candidate in negatives:
        candidate.review_decision = decisions["negative"][candidate.sample_id]

    accepted_teammate = [
        candidate
        for candidate in teammate
        if candidate.sample_id not in generic_review_ids or candidate.review_decision in accepted_values
    ]
    accepted_emtd = [candidate for candidate in emtd if candidate.review_decision in accepted_values]
    accepted_negatives = [candidate for candidate in negatives if candidate.review_decision in accepted_values]
    negative_seen: set[tuple[str, str]] = set()
    unique_negatives: list[Candidate] = []
    for candidate in sorted(accepted_negatives, key=lambda row: row.sample_id):
        key = (candidate.sha256, candidate.pixel_sha256)
        if key not in negative_seen:
            negative_seen.add(key)
            unique_negatives.append(candidate)
    accepted_negatives = unique_negatives
    if len(accepted_negatives) < config.negative_target:
        raise ValueError(
            f"Need at least {config.negative_target} accepted no-sign frames; found {len(accepted_negatives)}"
        )
    accepted_negatives = sorted(accepted_negatives, key=lambda row: row.sample_id)[: config.negative_target]
    positives, dedupe_decisions, near_matches = _dedupe([*accepted_teammate, *accepted_emtd])
    negative_hashes = {candidate.sha256 for candidate in accepted_negatives} | {candidate.pixel_sha256 for candidate in accepted_negatives}
    positive_hashes = {candidate.sha256 for candidate in positives} | {candidate.pixel_sha256 for candidate in positives}
    if negative_hashes & positive_hashes:
        raise ValueError("A no-sign negative is byte or decoded-pixel equal to a positive image")
    if {candidate.perceptual_hash for candidate in accepted_negatives} & {
        candidate.perceptual_hash for candidate in positives
    }:
        raise ValueError("A no-sign negative is perceptually equal to a positive image")
    release_size = len(positives) + len(accepted_negatives)
    split_assignments = _assign_splits(
        positives,
        config.seed,
        targets={
            "train": release_size * 0.70,
            "validation": release_size * 0.15,
            "test": release_size * 0.15 - len(accepted_negatives),
        },
    )
    _prepare_empty_directory(config.output_root, overwrite=overwrite)
    config.audit_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for candidate in sorted(positives, key=lambda row: row.sample_id):
        split = split_assignments[candidate.group_id]
        image, label = _copy_candidate(candidate, split, config.output_root)
        rows.append(_candidate_manifest_row(config, candidate, split, image, label))
    for candidate in accepted_negatives:
        image, label = _copy_candidate(candidate, "test", config.output_root)
        rows.append(_candidate_manifest_row(config, candidate, "test", image, label))
    _write_csv(config.output_manifest, sorted(rows, key=lambda row: row["sample_id"]), CANONICAL_FIELDS)
    data_yaml = {
        "path": config.output_root.as_posix(),
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "names": {0: "traffic_sign"},
    }
    (config.output_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    metadata = {
        "schema_version": "1.0",
        "dataset_id": config.release_id,
        "task": "detect",
        "names": {"0": "traffic_sign"},
        "annotation_status": "approved",
        "coursework_images_included": 0,
        "release_status": "assignment_only_candidate",
        "use_policy": "assignment_only_no_redistribution_no_shared_dvc",
        "licence_status": "assignment_only_waiver",
        "dvc_remote_push_allowed": False,
        "publication_prohibited": True,
        "source_group_policy": "known_duplicate_and_connected_components_isolated",
        "split_seed": config.seed,
        "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "negative_test_images": len(accepted_negatives),
        "split_counts": dict(Counter(row["split"] for row in rows)),
    }
    _write_json(config.output_root / "dataset_metadata.json", metadata)
    audit = {
        "schema_version": "1.0",
        **metadata,
        "reassembly": reassembly,
        "gates": {
            "all_included_positive_images_valid": True,
            "review_complete": True,
            "negative_target_met": len(accepted_negatives) >= config.negative_target,
            "coursework_images_included": 0,
            "one_detector_class": True,
            "group_safe_generated_split": True,
        },
        "counts": {
            "teammate_scanned": len(teammate),
            "emtd_scanned": len(emtd),
            "positive_retained": len(positives),
            "negative_retained": len(accepted_negatives),
            "teammate_excluded": len(teammate_exclusions),
            "emtd_excluded": len(emtd_exclusions),
            "negative_excluded": len(negative_exclusions),
            "perceptual_exact_hash_groups": len(near_matches),
        },
        "artifacts": {
            "manifest": config.output_manifest.relative_to(config.project_root).as_posix(),
            "data_yaml": (config.output_root / "data.yaml").relative_to(config.project_root).as_posix(),
            "review_root": config.review_root.relative_to(config.project_root).as_posix(),
        },
    }
    _write_csv(config.audit_root / "invalid_teammate_records.csv", teammate_exclusions, ["source_path", "reason"])
    _write_csv(config.audit_root / "invalid_emtd_records.csv", emtd_exclusions, ["source_path", "reason"])
    _write_csv(config.audit_root / "invalid_negative_records.csv", negative_exclusions, ["source_path", "reason"])
    _write_csv(config.audit_root / "duplicate_decisions.csv", dedupe_decisions, ["sample_id", "status", "reason"])
    _write_csv(config.audit_root / "perceptual_duplicate_groups.csv", near_matches, ["sample_id", "near_match_sample_id"])
    _write_json(config.audit_root / "release_audit.json", audit)
    return audit
