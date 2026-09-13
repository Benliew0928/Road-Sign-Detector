"""Phase B2 normalization, privacy triage, and duplicate suppression.

All decisions produced here are automated quarantine triage. They do not grant
licence approval, semantic labels, or release eligibility.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import mimetypes
import os
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from roadsign_assist.datasets.recovery_acquisition import (
    DEFAULT_ACQUISITION_ROOT,
    DEFAULT_AUDIT_ROOT,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ObjectLedgerEntry,
    check_disk_budget,
)
from roadsign_assist.paths import project_path

NORMALIZER_VERSION = "recovery_normalizer_v1"
PREPROCESSING_VERSION = "roadsign_raw_bgr_v1"
QUALITY_METHOD = "opencv_quality_heuristics_v1"
ROAD_CONTEXT_METHOD = "source_lexical_visual_heuristic_v1"
EMBEDDING_METHOD = "gray_grid_hsv_edge_l2_v1"
PRIVACY_METHOD = "opencv_haar_face_and_plate_candidate_v1"
ALLOWED_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".webp"})
ROAD_TERMS = (
    "jalan",
    "malaysia",
    "malaysian",
    "road",
    "route",
    "street",
    "traffic",
    "highway",
    "expressway",
    "motorway",
    "sign",
    "kuala",
    "selangor",
    "penang",
    "johor",
    "sabah",
    "sarawak",
)
LOW_VALUE_TERMS = ("aerial", "diagram", "flag", "logo", "map", "satellite", "svg", "tiff")


class NormalizedCandidate(BaseModel):
    """One deterministic B2 decision row."""

    model_config = ConfigDict(extra="forbid")

    normalization_batch_id: str
    processed_at: datetime
    object_id: str
    parent_object_id: str = ""
    source_id: str
    source_batch_id: str
    group_id: str
    sequence_id: str = ""
    source_timestamp: str = ""
    raw_path: str
    raw_sha256: str
    raw_bytes: int = Field(ge=0)
    normalized_path: str = ""
    anonymized_path: str = ""
    normalized_sha256: str = ""
    normalized_bytes: int = Field(default=0, ge=0)
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    mime_type: str = ""
    preprocessing_version: str = PREPROCESSING_VERSION
    perceptual_hash: str = ""
    embedding: list[float] = Field(default_factory=lambda: list[float]())
    embedding_method: str = EMBEDDING_METHOD
    blur_laplacian_variance: float = 0.0
    brightness_mean: float = 0.0
    contrast_stddev: float = 0.0
    entropy_bits: float = 0.0
    edge_density: float = 0.0
    quality_score: float = 0.0
    quality_method: str = QUALITY_METHOD
    road_context_score: float = 0.0
    road_context_method: str = ROAD_CONTEXT_METHOD
    privacy_flags: list[str] = Field(default_factory=list)
    privacy_regions: list[list[int]] = Field(default_factory=lambda: list[list[int]]())
    privacy_method: str = PRIVACY_METHOD
    duplicate_cluster_id: str = ""
    duplicate_of_object_id: str = ""
    duplicate_method: str = ""
    rejection_reason: str = ""
    triage_state: str
    collection_state: str = "quarantined"
    licence_status: str = "unreviewed"
    permitted_use: str = "none_pending_review"
    review_status: str = "pending"


def _now() -> datetime:
    return datetime.now(UTC)


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[str, object], value)


def _string(value: object) -> str:
    return "" if value is None else str(value)


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)) and value != "":
        return int(value)
    return 0


def _load_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                yield cast(Mapping[str, object], value)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, default=str) + "\n")
        handle.flush()


def _decode_image(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    with Image.open(path) as source:
        source.load()
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
    return cast(np.ndarray[Any, np.dtype[np.uint8]], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _video_frames(
    path: Path, *, parent_object_id: str, one_frame_per_seconds: float
) -> Iterator[tuple[str, np.ndarray[Any, np.dtype[np.uint8]], float]]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        stride = max(1, round(fps * one_frame_per_seconds))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                yield (
                    f"{parent_object_id}:frame:{index}",
                    cast(np.ndarray[Any, np.dtype[np.uint8]], frame),
                    index / fps,
                )
            index += 1
    finally:
        capture.release()


def _perceptual_hash(image_bgr: np.ndarray[Any, np.dtype[np.uint8]]) -> str:
    gray = cast(
        np.ndarray[Any, np.dtype[np.uint8]],
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
    )
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    values = cv2.dct(small.astype(np.float32))[:8, :8].flatten()
    median = float(np.median(values[1:]))
    bits = values > median
    return f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _embedding(image_bgr: np.ndarray[Any, np.dtype[np.uint8]]) -> list[float]:
    resized = cv2.resize(image_bgr, (96, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    grid = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).flatten()
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hue = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    saturation = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    edges = cv2.Canny(gray, 80, 160)
    edge_grid = cv2.resize(edges, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32).flatten()
    vector = np.concatenate((grid / 255.0, hue, saturation, edge_grid / 255.0))
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return [round(float(value), 7) for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return float(np.dot(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))


def _entropy(gray: np.ndarray[Any, np.dtype[np.uint8]]) -> float:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def _quality(
    image_bgr: np.ndarray[Any, np.dtype[np.uint8]], *, filename: str, source_id: str
) -> dict[str, float]:
    gray = cast(
        np.ndarray[Any, np.dtype[np.uint8]],
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
    )
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    entropy = _entropy(gray)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    quality_score = 100.0 * (
        0.30 * min(1.0, math.log10(max(blur, 1.0)) / 3.0)
        + 0.20 * min(1.0, contrast / 64.0)
        + 0.20 * min(1.0, entropy / 8.0)
        + 0.15 * min(1.0, edge_density / 0.18)
        + 0.15 * max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    )
    text = f"{filename} {source_id}".lower()
    lexical = min(1.0, sum(term in text for term in ROAD_TERMS) / 3.0)
    lexical -= min(0.8, 0.25 * sum(term in text for term in LOW_VALUE_TERMS))
    bottom = hsv[hsv.shape[0] // 2 :, :, :]
    neutral_fraction = float(
        np.mean((bottom[:, :, 1] < 100) & (bottom[:, :, 2] > 25) & (bottom[:, :, 2] < 235))
    )
    visual = min(1.0, 1.4 * neutral_fraction + 2.5 * edge_density)
    source_prior = 0.65 if source_id.startswith("kartaview") else 0.0
    road_context = max(0.0, min(1.0, max(source_prior, 0.55 * lexical + 0.45 * visual)))
    return {
        "blur_laplacian_variance": round(blur, 4),
        "brightness_mean": round(brightness, 4),
        "contrast_stddev": round(contrast, 4),
        "entropy_bits": round(entropy, 4),
        "edge_density": round(edge_density, 6),
        "quality_score": round(quality_score, 3),
        "road_context_score": round(road_context, 4),
    }


def _privacy_regions(
    image_bgr: np.ndarray[Any, np.dtype[np.uint8]], *, enabled: bool
) -> tuple[list[str], list[list[int]]]:
    if not enabled:
        return ["privacy_scan_disabled"], []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1280.0 / max(gray.shape))
    scan = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
    cascade_root = Path(cv2.__file__ or "").parent / "data"
    detectors = (
        ("possible_face", cascade_root / "haarcascade_frontalface_default.xml", 1.1, 5),
        ("possible_plate", cascade_root / "haarcascade_russian_plate_number.xml", 1.1, 4),
    )
    flags: list[str] = []
    regions: list[list[int]] = []
    for flag, cascade_path, scale_factor, neighbors in detectors:
        detector = cv2.CascadeClassifier(str(cascade_path))
        detections = detector.detectMultiScale(
            scan, scaleFactor=scale_factor, minNeighbors=neighbors, minSize=(20, 12)
        )
        if len(detections):
            flags.append(flag)
        for x, y, width, height in detections:
            regions.append(
                [
                    round(int(x) / scale),
                    round(int(y) / scale),
                    round(int(width) / scale),
                    round(int(height) / scale),
                ]
            )
    return flags, regions


def _encode_jpeg(path: Path, image_bgr: np.ndarray[Any, np.dtype[np.uint8]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError("OpenCV JPEG encoding failed")
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(encoded.tobytes())
    sha256 = _hash(temporary)
    size = temporary.stat().st_size
    os.replace(temporary, path)
    return sha256, size


def _anonymize(
    image_bgr: np.ndarray[Any, np.dtype[np.uint8]], regions: list[list[int]]
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    output = image_bgr.copy()
    for x, y, width, height in regions:
        x0, y0 = max(0, x), max(0, y)
        x1 = min(output.shape[1], x0 + width)
        y1 = min(output.shape[0], y0 + height)
        if x1 <= x0 or y1 <= y0:
            continue
        region = output[y0:y1, x0:x1]
        kernel = max(9, (min(region.shape[:2]) // 3) | 1)
        output[y0:y1, x0:x1] = cv2.GaussianBlur(region, (kernel, kernel), 0)
    return output


def _spent_hashes(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row.get("sha256", "").strip().lower()
            for row in csv.DictReader(handle)
            if row.get("sha256", "").strip()
        }


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    arrow: Any = pa
    parquet: Any = pq
    table: Any = arrow.Table.from_pylist(rows)
    parquet.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _source_rows(
    ledger: Path, *, source_id: str, source_batch_id: str
) -> list[Mapping[str, object]]:
    latest: dict[str, Mapping[str, object]] = {}
    for row in _load_jsonl(ledger):
        if _string(row.get("source_id")) != source_id:
            continue
        if _string(row.get("batch_id")) != source_batch_id:
            continue
        raw_path = project_path(_string(row.get("raw_path")))
        if raw_path.is_file():
            latest[_string(row.get("object_id"))] = row
    return sorted(latest.values(), key=lambda row: _string(row.get("object_id")))


def normalize_quarantine_batch(
    *,
    source_id: str,
    source_batch_id: str,
    normalization_batch_id: str,
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    spent_denylist: str
    | Path = "_archive/recovery_acquisition_v1/manifests/spent_data_denylist.csv",
    max_objects: int = 10_000,
    minimum_free_gib: float = 15.0,
    maximum_acquisition_gib: float = 10.0,
    one_frame_per_seconds: float = 1.0,
    privacy_scan: bool = True,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    """Normalize one acquired batch and produce immutable B2 decision artifacts."""
    if not 0 < max_objects <= 10_000:
        raise ValueError("max_objects must be within 1..10000")
    if one_frame_per_seconds <= 0:
        raise ValueError("one_frame_per_seconds must be positive")
    root = project_path(acquisition_root)
    audit = project_path(audit_root) / "normalization" / f"{normalization_batch_id}.json"
    parquet_path = root / "manifests" / "normalization" / f"{normalization_batch_id}.parquet"
    json_path = root / "manifests" / "normalization" / f"{normalization_batch_id}.json"
    progress_path = root / "manifests" / "normalization" / f"{normalization_batch_id}.jsonl"
    if json_path.exists():
        if resume:
            result: object = json.loads(json_path.read_text(encoding="utf-8"))
            return dict(_mapping(result)) | {"resume_reused": True}
        raise FileExistsError(f"Refusing to overwrite normalization manifest: {json_path}")
    object_ledger = root / "manifests" / "object_events.jsonl"
    inputs = _source_rows(object_ledger, source_id=source_id, source_batch_id=source_batch_id)[
        :max_objects
    ]
    estimated_bytes = sum(_integer(row.get("byte_size")) for row in inputs)
    budget = check_disk_budget(
        root,
        acquisition_root=root,
        minimum_free_gib=minimum_free_gib,
        maximum_acquisition_gib=maximum_acquisition_gib,
        planned_additional_bytes=estimated_bytes,
    )
    preflight: dict[str, object] = {
        "normalization_batch_id": normalization_batch_id,
        "source_id": source_id,
        "source_batch_id": source_batch_id,
        "candidate_objects": len(inputs),
        "estimated_maximum_output_bytes": estimated_bytes,
        "disk_allowed": budget.allowed,
        "disk_reasons": list(budget.reasons),
        "dry_run": dry_run,
    }
    if dry_run:
        return preflight
    if not budget.allowed:
        raise RuntimeError(f"Disk budget guard blocked normalization: {', '.join(budget.reasons)}")

    existing_rows = {
        _string(row.get("object_id")): NormalizedCandidate.model_validate(row)
        for row in _load_jsonl(progress_path)
        if _string(row.get("normalization_batch_id")) == normalization_batch_id
    }
    rows = dict(existing_rows)
    denylist = _spent_hashes(project_path(spent_denylist))
    prior_raw_hashes: dict[str, str] = {}
    for ledger_row in _load_jsonl(object_ledger):
        if _string(ledger_row.get("batch_id")) == source_batch_id:
            continue
        sha256 = _string(ledger_row.get("sha256")).lower()
        if sha256:
            prior_raw_hashes.setdefault(sha256, _string(ledger_row.get("object_id")))
    accepted = [row for row in rows.values() if row.triage_state == "normalized_candidate"]
    # Repeated autonomous batches must not admit the same view through a new ID.
    # Use only completed prior normalization reports; never consume a concurrent
    # batch's partial progress as a frozen deduplication decision.
    prior_candidates: dict[str, NormalizedCandidate] = {}
    for prior_report in sorted((root / "manifests" / "normalization").glob("*.json")):
        if prior_report == json_path:
            continue
        prior_progress = prior_report.with_suffix(".jsonl")
        if not prior_progress.is_file():
            continue
        for previous in _load_jsonl(prior_progress):
            if previous.get("triage_state") == "normalized_candidate":
                candidate = NormalizedCandidate.model_validate(previous)
                prior_candidates[candidate.object_id] = candidate
    accepted.extend(prior_candidates.values())
    normalized_dir = root / "normalized" / source_id / normalization_batch_id
    anonymized_dir = root / "anonymized" / source_id / normalization_batch_id

    def process_frame(
        ledger_row: Mapping[str, object],
        object_id: str,
        image_bgr: np.ndarray[Any, np.dtype[np.uint8]],
        *,
        parent_object_id: str = "",
        frame_seconds: float | None = None,
    ) -> NormalizedCandidate:
        raw_path = project_path(_string(ledger_row.get("raw_path")))
        raw_sha256 = _string(ledger_row.get("sha256")).lower()
        filename = _string(ledger_row.get("filename"))
        height, width = image_bgr.shape[:2]
        qualities = _quality(image_bgr, filename=filename, source_id=source_id)
        phash = _perceptual_hash(image_bgr)
        embedding = _embedding(image_bgr)
        privacy_flags, regions = _privacy_regions(image_bgr, enabled=privacy_scan)
        rejection = ""
        duplicate_of = ""
        duplicate_method = ""
        if raw_sha256 in denylist:
            rejection = "spent_evaluation_denylisted"
        elif raw_sha256 in prior_raw_hashes:
            rejection = "exact_duplicate_of_existing_object"
            duplicate_of = prior_raw_hashes[raw_sha256]
            duplicate_method = "raw_sha256"
        elif min(width, height) < 128 or width * height < 65_536:
            rejection = "thumbnail_only"
        elif qualities["quality_score"] < 15.0:
            rejection = "irrecoverable_quality"
        else:
            for previous in accepted:
                distance = _hamming(phash, previous.perceptual_hash)
                similarity = _cosine(embedding, previous.embedding)
                same_sequence = bool(
                    _string(ledger_row.get("sequence_id"))
                    and _string(ledger_row.get("sequence_id")) == previous.sequence_id
                )
                if same_sequence and (distance <= 10 or similarity >= 0.995):
                    rejection = "near_identical_adjacent_frame"
                    duplicate_of = previous.object_id
                    duplicate_method = f"sequence_phash_{distance}_cosine_{similarity:.6f}"
                    break
                if distance <= 4 or (distance <= 10 and similarity >= 0.998):
                    rejection = "perceptual_near_duplicate"
                    duplicate_of = previous.object_id
                    duplicate_method = f"phash_{distance}_cosine_{similarity:.6f}"
                    break
        normalized_path = ""
        anonymized_path = ""
        normalized_sha256 = ""
        normalized_bytes = 0
        triage_state = "rejected_auto" if rejection else "normalized_candidate"
        if not rejection:
            suffix = f"_t{frame_seconds:.3f}" if frame_seconds is not None else ""
            output = normalized_dir / f"{_stable_id(object_id)[:16]}{suffix}.jpg"
            normalized_sha256, normalized_bytes = _encode_jpeg(output, image_bgr)
            normalized_path = str(output)
            if regions:
                anonymized = anonymized_dir / output.name
                _encode_jpeg(anonymized, _anonymize(image_bgr, regions))
                anonymized_path = str(anonymized)
        duplicate_cluster_id = (
            _stable_id("duplicate", duplicate_of or object_id) if duplicate_of else ""
        )
        return NormalizedCandidate(
            normalization_batch_id=normalization_batch_id,
            processed_at=_now(),
            object_id=object_id,
            parent_object_id=parent_object_id,
            source_id=source_id,
            source_batch_id=source_batch_id,
            group_id=_string(ledger_row.get("group_id")),
            sequence_id=_string(ledger_row.get("sequence_id")),
            source_timestamp=_string(ledger_row.get("source_timestamp")),
            raw_path=str(raw_path),
            raw_sha256=raw_sha256,
            raw_bytes=_integer(ledger_row.get("byte_size")),
            normalized_path=normalized_path,
            anonymized_path=anonymized_path,
            normalized_sha256=normalized_sha256,
            normalized_bytes=normalized_bytes,
            width=width,
            height=height,
            mime_type="image/jpeg" if normalized_path else _string(ledger_row.get("mime_type")),
            perceptual_hash=phash,
            embedding=embedding,
            blur_laplacian_variance=qualities["blur_laplacian_variance"],
            brightness_mean=qualities["brightness_mean"],
            contrast_stddev=qualities["contrast_stddev"],
            entropy_bits=qualities["entropy_bits"],
            edge_density=qualities["edge_density"],
            quality_score=qualities["quality_score"],
            road_context_score=qualities["road_context_score"],
            privacy_flags=privacy_flags,
            privacy_regions=regions,
            duplicate_cluster_id=duplicate_cluster_id,
            duplicate_of_object_id=duplicate_of,
            duplicate_method=duplicate_method,
            rejection_reason=rejection,
            triage_state=triage_state,
            licence_status=_string(ledger_row.get("licence_status")) or "unreviewed",
        )

    for ledger_row in inputs:
        parent_id = _string(ledger_row.get("object_id"))
        if parent_id in rows:
            continue
        raw_path = project_path(_string(ledger_row.get("raw_path")))
        extension = raw_path.suffix.lower()
        produced: list[NormalizedCandidate] = []
        try:
            if extension in VIDEO_EXTENSIONS:
                for frame_id, frame, seconds in _video_frames(
                    raw_path,
                    parent_object_id=parent_id,
                    one_frame_per_seconds=one_frame_per_seconds,
                ):
                    if frame_id in rows:
                        continue
                    produced.append(
                        process_frame(
                            ledger_row,
                            frame_id,
                            frame,
                            parent_object_id=parent_id,
                            frame_seconds=seconds,
                        )
                    )
            elif extension in IMAGE_EXTENSIONS and extension in ALLOWED_IMAGE_EXTENSIONS:
                produced.append(process_frame(ledger_row, parent_id, _decode_image(raw_path)))
            else:
                produced.append(
                    NormalizedCandidate(
                        normalization_batch_id=normalization_batch_id,
                        processed_at=_now(),
                        object_id=parent_id,
                        source_id=source_id,
                        source_batch_id=source_batch_id,
                        group_id=_string(ledger_row.get("group_id")),
                        sequence_id=_string(ledger_row.get("sequence_id")),
                        raw_path=str(raw_path),
                        raw_sha256=_string(ledger_row.get("sha256")),
                        raw_bytes=_integer(ledger_row.get("byte_size")),
                        mime_type=_string(ledger_row.get("mime_type"))
                        or mimetypes.guess_type(raw_path.name)[0]
                        or "application/octet-stream",
                        privacy_flags=["privacy_scan_not_applicable"],
                        rejection_reason="unsupported_media",
                        triage_state="rejected_auto",
                        licence_status=_string(ledger_row.get("licence_status")) or "unreviewed",
                    )
                )
        except (OSError, UnidentifiedImageError, ValueError, cv2.error) as exc:
            produced.append(
                NormalizedCandidate(
                    normalization_batch_id=normalization_batch_id,
                    processed_at=_now(),
                    object_id=parent_id,
                    source_id=source_id,
                    source_batch_id=source_batch_id,
                    group_id=_string(ledger_row.get("group_id")),
                    sequence_id=_string(ledger_row.get("sequence_id")),
                    raw_path=str(raw_path),
                    raw_sha256=_string(ledger_row.get("sha256")),
                    raw_bytes=_integer(ledger_row.get("byte_size")),
                    mime_type=_string(ledger_row.get("mime_type")),
                    privacy_flags=["privacy_scan_failed"],
                    rejection_reason=f"decode_failure:{type(exc).__name__}",
                    triage_state="rejected_auto",
                    licence_status=_string(ledger_row.get("licence_status")) or "unreviewed",
                )
            )
        for candidate in produced:
            payload = candidate.model_dump(mode="json")
            _append_jsonl(progress_path, payload)
            rows[candidate.object_id] = candidate
            if candidate.triage_state == "normalized_candidate":
                accepted.append(candidate)
            normalized_event = ObjectLedgerEntry(
                event_id=_stable_id(
                    "normalize", normalization_batch_id, candidate.object_id, candidate.raw_sha256
                ),
                observed_at=candidate.processed_at,
                object_id=candidate.object_id,
                parent_object_id=candidate.parent_object_id,
                source_id=source_id,
                batch_id=normalization_batch_id,
                group_id=candidate.group_id,
                sequence_id=candidate.sequence_id,
                filename=Path(candidate.normalized_path or candidate.raw_path).name,
                sha256=candidate.normalized_sha256 or candidate.raw_sha256,
                byte_size=candidate.normalized_bytes or candidate.raw_bytes,
                mime_type=candidate.mime_type,
                width=candidate.width,
                height=candidate.height,
                source_timestamp=candidate.source_timestamp,
                raw_path=candidate.raw_path,
                normalized_path=candidate.normalized_path,
                privacy_flags=candidate.privacy_flags,
                quality_scores={
                    "blur_laplacian_variance": candidate.blur_laplacian_variance,
                    "quality_score": candidate.quality_score,
                    "road_context_score": candidate.road_context_score,
                },
                duplicate_cluster_id=candidate.duplicate_cluster_id,
                rejection_reason=candidate.rejection_reason,
                road_context="candidate" if candidate.road_context_score >= 0.35 else "low_score",
                collection_state="quarantined",
                licence_status=candidate.licence_status,
                permitted_use="none_pending_review",
                review_status="pending",
            )
            _append_jsonl(object_ledger, normalized_event.model_dump(mode="json"))

    ordered = [rows[key] for key in sorted(rows)]
    serial_rows = [row.model_dump(mode="json") for row in ordered]
    _write_parquet(parquet_path, serial_rows)
    states = Counter(row.triage_state for row in ordered)
    rejections = Counter(row.rejection_reason for row in ordered if row.rejection_reason)
    privacy = Counter(flag for row in ordered for flag in row.privacy_flags)
    report: dict[str, object] = preflight | {
        "dry_run": False,
        "normalizer_version": NORMALIZER_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "quality_method": QUALITY_METHOD,
        "road_context_method": ROAD_CONTEXT_METHOD,
        "embedding_method": EMBEDDING_METHOD,
        "privacy_method": PRIVACY_METHOD,
        "rows": len(ordered),
        "states": dict(sorted(states.items())),
        "rejections": dict(sorted(rejections.items())),
        "privacy_flags": dict(sorted(privacy.items())),
        "normalized_bytes": sum(row.normalized_bytes for row in ordered),
        "unique_normalized_sha256": len(
            {row.normalized_sha256 for row in ordered if row.normalized_sha256}
        ),
        "road_context_score_at_least_0_35": sum(row.road_context_score >= 0.35 for row in ordered),
        "quality_score_at_least_40": sum(row.quality_score >= 40.0 for row in ordered),
        "outputs": {
            "manifest_json": str(json_path),
            "manifest_parquet": str(parquet_path),
            "progress_jsonl": str(progress_path),
            "audit_report": str(audit),
            "normalized_directory": str(normalized_dir),
            "anonymized_directory": str(anonymized_dir),
        },
        "exit_condition": {
            "passed": bool(ordered)
            and all(row.raw_sha256 and row.group_id for row in ordered)
            and all(
                row.rejection_reason
                or (row.normalized_path and row.perceptual_hash and row.embedding)
                for row in ordered
            ),
            "basis": "Every input has a traceable grouped decision; retained candidates are normalized, quality-scored, privacy-triaged, and deduplicated.",
        },
        "resume_reused": False,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "EMBEDDING_METHOD",
    "NORMALIZER_VERSION",
    "PREPROCESSING_VERSION",
    "PRIVACY_METHOD",
    "QUALITY_METHOD",
    "ROAD_CONTEXT_METHOD",
    "NormalizedCandidate",
    "normalize_quarantine_batch",
]
