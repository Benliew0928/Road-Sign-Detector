"""Collect the owner-review queue for Phase C's no-sign test frames.

The source annotations are used only to remove frames that already contain an
annotated traffic sign.  They are not a substitute for the owner's visual
review: every resulting row remains pending until the owner confirms that the
full frame contains no traffic sign.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
from PIL import Image, ImageOps

from roadsign_assist.datasets.contact_sheet import ReviewTile, render_contact_sheet
from roadsign_assist.paths import PROJECT_ROOT

SOURCE_DATASET = "lance-format/BDD100K-enriched"
SOURCE_API_ROOT = "https://datasets-server.huggingface.co"
SOURCE_METADATA_URL = f"https://huggingface.co/api/datasets/{SOURCE_DATASET}"
SOURCE_ROWS_URL = f"{SOURCE_API_ROOT}/rows"
COLLECTION_ID = "phase_c_no_sign_bdd100k_candidates_20260829"
SEED = 2513
TARGET = 120
SCENES = ("city street", "highway", "residential")
TIMES = ("daytime", "dawn/dusk", "night")
STRATA = tuple((scene, time) for scene in SCENES for time in TIMES)

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
MANIFEST_FIELDS = [
    *REVIEW_FIELDS,
    "filename",
    "source_dataset",
    "source_revision",
    "source_split",
    "source_row_index",
    "source_annotation_categories",
    "source_duplicate_flag",
    "weather",
    "scene",
    "timeofday",
    "width",
    "height",
    "sha256",
    "pixel_sha256",
    "dhash64",
    "perceptual_hash",
    "automatic_screening_status",
    "licence_status",
    "use_policy",
]


@dataclass(frozen=True)
class RemoteCandidate:
    """A source row selected before the image is materialized locally."""

    image_id: str
    row_index: int
    image_url: str
    source_split: str
    categories: tuple[str, ...]
    weather: str
    scene: str
    timeofday: str
    width: int
    height: int

    @property
    def stratum(self) -> tuple[str, str]:
        return (self.scene, self.timeofday)

    @property
    def selection_key(self) -> str:
        return hashlib.sha256(f"{SEED}:{self.image_id}".encode()).hexdigest()


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
        dhash = "".join("1" if bit else "0" for bit in (gray[:, 1:] > gray[:, :-1]).flat)
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


def _as_mapping(value: object, *, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object for {description}")
    return cast(dict[str, Any], value)


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def candidate_from_row(raw: dict[str, Any], row_index: int) -> RemoteCandidate | None:
    """Return a candidate only when source metadata supports a no-sign review."""
    raw_categories = raw.get("ann_categories")
    categories = (
        tuple(str(value).strip().casefold() for value in cast(list[object], raw_categories))
        if isinstance(raw_categories, list)
        else ()
    )
    scene = str(raw.get("scene", "")).strip().casefold()
    timeofday = str(raw.get("timeofday", "")).strip().casefold()
    raw_image = raw.get("image_bytes")
    image = cast(dict[str, Any], raw_image) if isinstance(raw_image, dict) else None
    image_url = image.get("src") if image is not None else None
    if (
        "traffic sign" in categories
        or bool(raw.get("is_duplicate"))
        or (scene, timeofday) not in STRATA
        or not isinstance(image_url, str)
        or not str(raw.get("image_id", "")).strip()
    ):
        return None
    assert image is not None
    return RemoteCandidate(
        image_id=str(raw["image_id"]),
        row_index=row_index,
        image_url=image_url,
        source_split=str(raw.get("split", "")),
        categories=categories,
        weather=str(raw.get("weather", "")),
        scene=scene,
        timeofday=timeofday,
        width=_as_int(raw.get("width", image.get("width", 0))),
        height=_as_int(raw.get("height", image.get("height", 0))),
    )


def _page_offsets(total_rows: int, *, pages: int) -> list[int]:
    """Pick deterministic, non-overlapping pages distributed through the source."""
    if total_rows < 100:
        raise ValueError("The source dataset has too few rows")
    offsets: list[int] = []
    for index in range(pages):
        start = (total_rows * index) // pages
        end = (total_rows * (index + 1)) // pages - 100
        if end < start:
            end = start
        span = end - start + 1
        raw = int.from_bytes(
            hashlib.sha256(f"{SEED}:page:{index}".encode()).digest()[:8], "big"
        )
        offsets.append(start + raw % span)
    return offsets


def _fetch_source_candidates(client: httpx.Client, *, target: int) -> tuple[list[RemoteCandidate], str]:
    metadata = client.get(SOURCE_METADATA_URL)
    metadata.raise_for_status()
    metadata_json = _as_mapping(metadata.json(), description="source metadata")
    revision = str(metadata_json["sha"])

    info = client.get(
        f"{SOURCE_API_ROOT}/info",
        params={"dataset": SOURCE_DATASET},
    )
    info.raise_for_status()
    info_json = _as_mapping(info.json(), description="source information")
    dataset_info = _as_mapping(info_json["dataset_info"], description="dataset information")
    default = _as_mapping(dataset_info["default"], description="default dataset information")
    splits = _as_mapping(default["splits"], description="source splits")
    train = _as_mapping(splits["train"], description="source train split")
    total_rows = _as_int(train["num_examples"])
    by_id: dict[str, RemoteCandidate] = {}
    # Twelve distributed pages provide far more candidates than the 120-frame
    # intake needs, without tripping the public viewer service's request cap.
    for offset in _page_offsets(total_rows, pages=12):
        response = client.get(
            SOURCE_ROWS_URL,
            params={
                "dataset": SOURCE_DATASET,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
            },
        )
        response.raise_for_status()
        response_json = _as_mapping(response.json(), description="source row page")
        rows_value = response_json.get("rows")
        if not isinstance(rows_value, list):
            raise ValueError("Source row page did not contain a rows list")
        for item_value in cast(list[object], rows_value):
            if not isinstance(item_value, Mapping):
                continue
            item = cast(Mapping[str, Any], item_value)
            row_value = item.get("row")
            if not isinstance(row_value, dict):
                continue
            row = cast(dict[str, Any], row_value)
            candidate = candidate_from_row(
                row,
                _as_int(item.get("row_idx", offset)),
            )
            if candidate is not None:
                by_id.setdefault(candidate.image_id, candidate)
    by_stratum: dict[tuple[str, str], list[RemoteCandidate]] = {
        stratum: sorted(
            (candidate for candidate in by_id.values() if candidate.stratum == stratum),
            key=lambda candidate: candidate.selection_key,
        )
        for stratum in STRATA
    }
    if sum(len(candidates) for candidates in by_stratum.values()) < target:
        raise RuntimeError("Source did not return enough annotation-screened road frames")

    # Round-robin ordering makes the 120 candidates varied without making a
    # scarce source stratum a release blocker.  The owner remains the final
    # no-sign decision maker for every frame.
    selected: list[RemoteCandidate] = []
    index = 0
    while any(index < len(candidates) for candidates in by_stratum.values()):
        for stratum in STRATA:
            candidates = by_stratum[stratum]
            if index < len(candidates):
                selected.append(candidates[index])
        index += 1
    return selected, revision


def _prepare_new_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to mix a new candidate collection with existing files: {path}. "
            "Use --overwrite after preserving any review decisions."
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _existing_review_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    rows, _ = _read_csv(path)
    return {row["sample_id"]: row for row in rows if row.get("sample_id")}


def approve_phase_c_negative_candidates(
    *,
    project_root: Path = PROJECT_ROOT,
    manifest_path: Path = Path(f"data/manifests/{COLLECTION_ID}.csv"),
    review_path: Path = Path(
        "data/manifests/detector_production_assignment_v1_20260829_reviews/"
        "negative_no_sign_review.csv"
    ),
    owner_note: str = "Owner accepted all 120 no-sign candidates in Codex on 2026-08-29.",
) -> dict[str, Any]:
    """Record the owner's explicit acceptance of the complete 120-frame set."""
    root = project_root.resolve()
    manifest = (root / manifest_path).resolve()
    review = (root / review_path).resolve()
    manifest_rows, manifest_fields = _read_csv(manifest)
    review_rows, review_fields = _read_csv(review)
    if len(manifest_rows) != TARGET or len(review_rows) != TARGET:
        raise ValueError(
            f"Expected {TARGET} negative candidates in both manifests; "
            f"found {len(manifest_rows)} and {len(review_rows)}"
        )
    manifest_ids = {row.get("sample_id", "") for row in manifest_rows}
    review_ids = {row.get("sample_id", "") for row in review_rows}
    if "" in manifest_ids or manifest_ids != review_ids or len(manifest_ids) != TARGET:
        raise ValueError("Negative candidate and review manifests do not have matching sample IDs")
    allowed_pending = {"", "accept", "accepted", "approve", "approved"}
    for row in [*manifest_rows, *review_rows]:
        if row.get("review_decision", "").strip().casefold() not in allowed_pending:
            raise ValueError(
                "Cannot replace an existing negative review decision: "
                f"{row.get('sample_id', '')}={row.get('review_decision', '')}"
            )
    for row in manifest_rows:
        if row.get("source_kind") != "negative":
            raise ValueError(f"Unexpected source kind in candidate manifest: {row.get('source_kind', '')}")
        row["review_decision"] = "accept"
        row["reviewer_notes"] = owner_note
    for row in review_rows:
        row["review_decision"] = "accept"
        row["reviewer_notes"] = owner_note
    _write_csv(manifest, manifest_rows, manifest_fields)
    _write_csv(review, review_rows, review_fields)
    return {"accepted_count": TARGET, "manifest": manifest, "review_queue": review}


def collect_phase_c_negative_candidates(
    *,
    project_root: Path = PROJECT_ROOT,
    output_root: Path = Path("data/raw/phase_c_no_sign_frames"),
    manifest_path: Path = Path(f"data/manifests/{COLLECTION_ID}.csv"),
    review_path: Path = Path(
        "data/manifests/detector_production_assignment_v1_20260829_reviews/"
        "negative_no_sign_review.csv"
    ),
    contact_sheet_root: Path = Path(f"outputs/review/{COLLECTION_ID}"),
    target: int = TARGET,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize exactly ``target`` annotation-screened, pending-review frames."""
    if target != TARGET:
        raise ValueError(f"Phase C requires exactly {TARGET} candidates, not {target}")
    root = project_root.resolve()
    destination = (root / output_root).resolve()
    manifest = (root / manifest_path).resolve()
    review = (root / review_path).resolve()
    sheets = (root / contact_sheet_root).resolve()
    _prepare_new_directory(destination, overwrite=overwrite)
    if manifest.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest}")
    if review.exists() and review.stat().st_size > len(",".join(REVIEW_FIELDS)) + 2 and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing review decisions: {review}. Use --overwrite only before review."
        )
    existing_review_rows = _existing_review_rows(review)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    with httpx.Client(timeout=httpx.Timeout(60.0), follow_redirects=True) as client:
        remote_candidates, revision = _fetch_source_candidates(client, target=target)
        accepted: list[dict[str, str]] = []
        seen_hashes: set[str] = set()
        seen_perceptual: set[str] = set()
        counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        for candidate in remote_candidates:
            if len(accepted) >= target:
                continue
            response = client.get(candidate.image_url)
            response.raise_for_status()
            temp = destination / f".{candidate.image_id}.download"
            temp.write_bytes(response.content)
            try:
                pixel_sha256, dhash64, perceptual_hash, width, height = _pixel_hash(temp)
                sha256 = _sha256(temp)
            except OSError:
                temp.unlink(missing_ok=True)
                continue
            if sha256 in seen_hashes or pixel_sha256 in seen_hashes or perceptual_hash in seen_perceptual:
                temp.unlink(missing_ok=True)
                continue
            sample_id = hashlib.sha256(f"negative:{sha256}".encode()).hexdigest()[:24]
            filename = f"bdd100k_{candidate.image_id}.jpg"
            final_image = destination / filename
            temp.replace(final_image)
            source_path = final_image.as_posix()
            reasons = "source_annotation_no_traffic_sign;manual_full_frame_no_sign_confirmation"
            row = {
                "sample_id": sample_id,
                "source_kind": "negative",
                "source_path": source_path,
                "layout_root_id": "bdd100k_enriched",
                "source_image_id": candidate.image_id,
                "review_reasons": reasons,
                "review_decision": existing_review_rows.get(sample_id, {}).get("review_decision", ""),
                "reviewer_notes": existing_review_rows.get(sample_id, {}).get("reviewer_notes", ""),
                "filename": filename,
                "source_dataset": SOURCE_DATASET,
                "source_revision": revision,
                "source_split": candidate.source_split,
                "source_row_index": str(candidate.row_index),
                "source_annotation_categories": ";".join(candidate.categories),
                "source_duplicate_flag": "false",
                "weather": candidate.weather,
                "scene": candidate.scene,
                "timeofday": candidate.timeofday,
                "width": str(width),
                "height": str(height),
                "sha256": sha256,
                "pixel_sha256": pixel_sha256,
                "dhash64": dhash64,
                "perceptual_hash": perceptual_hash,
                "automatic_screening_status": "annotation_negative_candidate",
                "licence_status": "assignment_only_waiver",
                "use_policy": "assignment_only_no_redistribution_no_shared_dvc",
            }
            accepted.append(row)
            counts[candidate.stratum] += 1
            seen_hashes.update({sha256, pixel_sha256})
            seen_perceptual.add(perceptual_hash)

    if len(accepted) != target:
        raise RuntimeError(f"Could not materialize {target} unique candidate frames")
    accepted.sort(key=lambda row: row["sample_id"])
    _write_csv(manifest, accepted, MANIFEST_FIELDS)
    _write_csv(review, accepted, REVIEW_FIELDS)
    sheets.mkdir(parents=True, exist_ok=True)
    for index in range(0, len(accepted), 20):
        page = accepted[index : index + 20]
        render_contact_sheet(
            [
                ReviewTile(
                    label=f"{item['filename'][8:20]} | {item['scene']} | {item['timeofday']}",
                    image_path=Path(item["source_path"]),
                )
                for item in page
            ],
            sheets / f"review_{index // 20 + 1:02d}.jpg",
            columns=5,
            tile_width=240,
            tile_height=180,
        )
    audit = {
        "collection_id": COLLECTION_ID,
        "candidate_count": len(accepted),
        "selection_seed": SEED,
        "source_dataset": SOURCE_DATASET,
        "source_revision": revision,
        "source_annotation_rule": "ann_categories does not contain traffic sign",
        "automatic_screening_status": "annotation_negative_candidate",
        "manual_review_required": True,
        "licence_status": "assignment_only_waiver",
        "use_policy": "assignment_only_no_redistribution_no_shared_dvc",
        "strata": {
            f"{scene}|{timeofday}": counts[(scene, timeofday)] for scene, timeofday in STRATA
        },
    }
    (manifest.with_suffix(".json")).write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "candidate_count": len(accepted),
        "manifest": manifest,
        "review_queue": review,
        "contact_sheet_root": sheets,
        "source_revision": revision,
    }
