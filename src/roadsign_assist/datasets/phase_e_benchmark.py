"""Prepare and freeze the owner-reviewed Phase E end-to-end benchmark."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from roadsign_assist.catalogue.models import ParameterType
from roadsign_assist.catalogue.repository import catalogue_by_id
from roadsign_assist.datasets.contact_sheet import ReviewTile, render_contact_sheet
from roadsign_assist.paths import PROJECT_ROOT

BENCHMARK_ID = "phase_e_benchmark_v1_20260901"
SEED = 2513
EXTERNAL_TARGET = 60
EXTERNAL_PER_BUCKET = 20
BDD_DATASET = "lance-format/BDD100K-enriched"
BDD_REVISION = "d82c5188d392714ba8091d68014f7b9838ceadf2"
BDD_API_ROOT = "https://datasets-server.huggingface.co"
BDD_METADATA_URL = f"https://huggingface.co/api/datasets/{BDD_DATASET}"

INVENTORY_PATH = Path(f"data/manifests/{BENCHMARK_ID}_inventory.csv")
REVIEW_PATH = Path(f"data/manifests/{BENCHMARK_ID}_review.csv")
CANONICAL_PATH = Path(f"data/manifests/{BENCHMARK_ID}.csv")
SUPPORTED_LABELS_PATH = Path("models/exported/runtime/sign_classifier.labels.json")
EXTERNAL_ROOT = Path("data/raw/phase_e_bdd100k_positive")
REVIEW_ROOT = Path(f"outputs/review/{BENCHMARK_ID}")
AUDIT_ROOT = Path(f"outputs/audit/{BENCHMARK_ID}")

FIELDS = [
    "benchmark_id",
    "instance_id",
    "image_id",
    "image_path",
    "expected_kind",
    "domain",
    "source_kind",
    "source_id",
    "source_revision",
    "source_row_index",
    "source_image_id",
    "condition_bucket",
    "scene",
    "weather",
    "timeofday",
    "image_width",
    "image_height",
    "x1",
    "y1",
    "x2",
    "y2",
    "area_ratio",
    "small_sign",
    "very_small_sign",
    "source_semantic_hint",
    "semantic_sign_id",
    "parameter_value",
    "parameter_unit",
    "bbox_review_decision",
    "review_decision",
    "reviewer",
    "reviewed_at",
    "reviewer_notes",
    "sha256",
    "pixel_sha256",
    "dhash64",
    "crop_sha256",
    "crop_pixel_sha256",
    "crop_dhash64",
    "licence_status",
    "use_policy",
    "evaluation_only",
    "included_in_training",
]
IMMUTABLE_REVIEW_FIELDS = (
    "benchmark_id",
    "instance_id",
    "image_id",
    "image_path",
    "expected_kind",
    "domain",
    "source_kind",
    "source_id",
    "source_revision",
    "source_row_index",
    "source_image_id",
    "condition_bucket",
    "scene",
    "weather",
    "timeofday",
    "image_width",
    "image_height",
    "sha256",
    "pixel_sha256",
    "dhash64",
    "licence_status",
    "use_policy",
    "evaluation_only",
    "included_in_training",
)


@dataclass(frozen=True)
class BDDCandidate:
    image_id: str
    row_index: int
    image_url: str
    source_split: str
    weather: str
    scene: str
    timeofday: str
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float], ...]

    @property
    def bucket(self) -> str:
        if self.timeofday.casefold() == "night":
            return "night"
        if self.weather.casefold() not in {"", "clear", "undefined"}:
            return "adverse_weather"
        return "daytime_clear"

    @property
    def selection_key(self) -> str:
        return hashlib.sha256(f"{SEED}:{self.image_id}".encode()).hexdigest()


def _resolve(root: Path, path: Path | str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _project_rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_hashes(image: Image.Image) -> tuple[str, str, str]:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    gray = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = "".join("1" if bit else "0" for bit in (gray[:, 1:] > gray[:, :-1]).flat)
    return pixel_sha, bits, f"{int(bits, 2):016x}"


def _path_hashes(path: Path) -> tuple[str, str, str, int, int]:
    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
        pixel_sha, bits, hex_hash = _image_hashes(image)
    return pixel_sha, bits, hex_hash, width, height


def _hamming(left: str, right: str) -> int:
    def parsed(value: str) -> int:
        stripped = value.strip().casefold()
        return (
            int(stripped, 2)
            if set(stripped) <= {"0", "1"} and len(stripped) == 64
            else int(stripped, 16)
        )

    return (parsed(left) ^ parsed(right)).bit_count()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_int(value: object) -> int:
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return 0


def _as_mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object for {description}")
    return cast(dict[str, Any], value)


def _page_offsets(total: int, pages: int = 24) -> list[int]:
    offsets: list[int] = []
    for index in range(pages):
        start = (total * index) // pages
        end = max(start, (total * (index + 1)) // pages - 100)
        raw = int.from_bytes(
            hashlib.sha256(f"{SEED}:phase-e-page:{index}".encode()).digest()[:8], "big"
        )
        offsets.append(start + raw % (end - start + 1))
    return offsets


def _bdd_candidate(raw: dict[str, Any], row_index: int) -> BDDCandidate | None:
    categories = raw.get("ann_categories")
    boxes = raw.get("ann_bboxes")
    image_value = raw.get("image_bytes")
    if not isinstance(categories, list) or not isinstance(boxes, list):
        return None
    category_values = cast(list[object], categories)
    box_values = cast(list[object], boxes)
    if (
        len(category_values) != len(box_values)
        or not isinstance(image_value, dict)
        or bool(raw.get("is_duplicate"))
    ):
        return None
    image_mapping = cast(dict[str, object], image_value)
    image_url = image_mapping.get("src")
    image_id = str(raw.get("image_id", "")).strip()
    if not isinstance(image_url, str) or not image_id:
        return None
    traffic_boxes: list[tuple[float, float, float, float]] = []
    for category, raw_box in zip(category_values, box_values, strict=True):
        if str(category).strip().casefold() != "traffic sign" or not isinstance(raw_box, list):
            continue
        raw_box_values = cast(list[object], raw_box)
        if len(raw_box_values) != 4:
            continue
        x1, y1, x2, y2 = (float(cast(Any, value)) for value in raw_box_values)
        if x2 > x1 and y2 > y1:
            traffic_boxes.append((x1, y1, x2, y2))
    if not traffic_boxes:
        return None
    return BDDCandidate(
        image_id=image_id,
        row_index=row_index,
        image_url=image_url,
        source_split=str(raw.get("split", "")),
        weather=str(raw.get("weather", "")).strip().casefold(),
        scene=str(raw.get("scene", "")).strip().casefold(),
        timeofday=str(raw.get("timeofday", "")).strip().casefold(),
        width=_as_int(raw.get("width", image_mapping.get("width"))),
        height=_as_int(raw.get("height", image_mapping.get("height"))),
        boxes=tuple(traffic_boxes),
    )


def _fetch_bdd_candidates(client: httpx.Client) -> dict[str, list[BDDCandidate]]:
    metadata = client.get(BDD_METADATA_URL)
    metadata.raise_for_status()
    revision = str(_as_mapping(metadata.json(), "BDD metadata").get("sha", ""))
    if revision != BDD_REVISION:
        raise RuntimeError(
            f"BDD100K source revision changed: expected {BDD_REVISION}, found {revision}. "
            "Do not silently rebuild the evaluation set from a different source revision."
        )
    info = client.get(f"{BDD_API_ROOT}/info", params={"dataset": BDD_DATASET})
    info.raise_for_status()
    payload = _as_mapping(info.json(), "BDD info")
    dataset_info = _as_mapping(payload["dataset_info"], "BDD dataset info")
    default = _as_mapping(dataset_info["default"], "BDD default config")
    splits = _as_mapping(default["splits"], "BDD splits")
    train = _as_mapping(splits["train"], "BDD train split")
    total = _as_int(train["num_examples"])
    unique: dict[str, BDDCandidate] = {}
    for offset in _page_offsets(total):
        response = client.get(
            f"{BDD_API_ROOT}/rows",
            params={
                "dataset": BDD_DATASET,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
            },
        )
        response.raise_for_status()
        rows = _as_mapping(response.json(), "BDD rows").get("rows")
        if not isinstance(rows, list):
            raise ValueError("BDD rows response is missing rows")
        for item_value in cast(list[object], rows):
            if not isinstance(item_value, dict):
                continue
            item = cast(dict[str, object], item_value)
            row_value = item.get("row")
            if not isinstance(row_value, dict):
                continue
            candidate = _bdd_candidate(
                cast(dict[str, Any], row_value), _as_int(item.get("row_idx"))
            )
            if candidate is not None:
                unique.setdefault(candidate.image_id, candidate)
    buckets = {
        name: sorted(
            (candidate for candidate in unique.values() if candidate.bucket == name),
            key=lambda candidate: candidate.selection_key,
        )
        for name in ("daytime_clear", "night", "adverse_weather")
    }
    short = {
        name: len(rows) for name, rows in buckets.items() if len(rows) < EXTERNAL_PER_BUCKET * 2
    }
    if short:
        raise RuntimeError(f"BDD100K scan did not provide enough positive candidates: {short}")
    return buckets


def _existing_hashes(root: Path) -> dict[str, object]:
    detector = _read_csv(root / "data/manifests/detector_production_assignment_v1_20260829.csv")
    classifier = _read_csv(root / "data/manifests/classifier_production_78_v3_20260829.csv")
    negatives = _read_csv(root / "data/manifests/phase_c_no_sign_bdd100k_candidates_20260829.csv")
    return {
        "detector_sha": {row["sha256"] for row in detector if row.get("sha256")},
        "detector_pixel": {row["pixel_sha256"] for row in detector if row.get("pixel_sha256")},
        "detector_dhash": [row["dhash64"] for row in detector if row.get("dhash64")],
        "classifier_sha": {row["crop_sha256"] for row in classifier if row.get("crop_sha256")},
        "classifier_pixel": {
            row["crop_pixel_sha256"] for row in classifier if row.get("crop_pixel_sha256")
        },
        "classifier_dhash": [
            row["perceptual_hash"] for row in classifier if row.get("perceptual_hash")
        ],
        "production_source_ids": {
            value
            for row in [*detector, *classifier, *negatives]
            for key in (
                "sample_id",
                "source_image_id",
                "source_candidate_id",
                "source_group",
            )
            if (value := row.get(key))
        },
    }


def _near_any(value: str, candidates: Iterable[str], threshold: int = 6) -> bool:
    return any(_hamming(value, candidate) <= threshold for candidate in candidates)


def _external_collision_reasons(rows: list[dict[str, str]], hashes: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    checked_images: set[str] = set()
    for row in rows:
        if row.get("domain") != "external_non_malaysian":
            continue
        image_id = row.get("image_id", "")
        if image_id not in checked_images:
            checked_images.add(image_id)
            if row.get("source_image_id") in cast(set[str], hashes["production_source_ids"]):
                reasons.append(f"production_source_id_collision:{image_id}")
            for release in ("detector", "classifier"):
                if row.get("sha256") in cast(set[str], hashes[f"{release}_sha"]) or row.get(
                    "pixel_sha256"
                ) in cast(set[str], hashes[f"{release}_pixel"]):
                    reasons.append(f"{release}_frame_exact_or_pixel_collision:{image_id}")
                elif _near_any(row.get("dhash64", ""), cast(list[str], hashes[f"{release}_dhash"])):
                    reasons.append(f"{release}_frame_perceptual_collision:{image_id}")
        instance_id = row.get("instance_id", "")
        for release in ("detector", "classifier"):
            if row.get("crop_sha256") in cast(set[str], hashes[f"{release}_sha"]) or row.get(
                "crop_pixel_sha256"
            ) in cast(set[str], hashes[f"{release}_pixel"]):
                reasons.append(f"{release}_crop_exact_or_pixel_collision:{instance_id}")
            elif _near_any(
                row.get("crop_dhash64", ""), cast(list[str], hashes[f"{release}_dhash"])
            ):
                reasons.append(f"{release}_crop_perceptual_collision:{instance_id}")
    return reasons


def _crop_hashes(
    image: Image.Image, box: tuple[float, float, float, float]
) -> tuple[str, str, str]:
    width, height = image.size
    x1, y1, x2, y2 = box
    crop = image.crop(
        (
            max(0, math.floor(x1)),
            max(0, math.floor(y1)),
            min(width, math.ceil(x2)),
            min(height, math.ceil(y2)),
        )
    )
    pixel, _, hex_hash = _image_hashes(crop)
    encoded = io.BytesIO()
    crop.save(encoded, format="PNG", optimize=False)
    return hashlib.sha256(encoded.getvalue()).hexdigest(), pixel, hex_hash


def _download_external_rows(
    root: Path, destination: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    destination.mkdir(parents=True, exist_ok=True)
    hashes = _existing_hashes(root)
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    selected_images: set[str] = set()
    selected_pixels: set[str] = set()
    selected_dhash: list[str] = []
    counts: defaultdict[str, int] = defaultdict(int)
    with httpx.Client(timeout=httpx.Timeout(60.0), follow_redirects=True) as client:
        buckets = _fetch_bdd_candidates(client)
        for bucket in ("daytime_clear", "night", "adverse_weather"):
            for candidate in buckets[bucket]:
                if counts[bucket] >= EXTERNAL_PER_BUCKET:
                    break
                if candidate.image_id in cast(set[str], hashes["production_source_ids"]):
                    exclusions.append(
                        {
                            "source_image_id": candidate.image_id,
                            "reason": "production_source_id_collision",
                        }
                    )
                    continue
                response = client.get(candidate.image_url)
                response.raise_for_status()
                temp = destination / f".{candidate.image_id}.download"
                temp.write_bytes(response.content)
                try:
                    pixel, bits, _, width, height = _path_hashes(temp)
                    byte_sha = _sha256(temp)
                    with Image.open(temp) as raw:
                        image = ImageOps.exif_transpose(raw).convert("RGB")
                        crop_hashes = [_crop_hashes(image, box) for box in candidate.boxes]
                except OSError:
                    temp.unlink(missing_ok=True)
                    exclusions.append(
                        {"source_image_id": candidate.image_id, "reason": "invalid_image"}
                    )
                    continue
                collision = ""
                if byte_sha in cast(set[str], hashes["detector_sha"]) or pixel in cast(
                    set[str], hashes["detector_pixel"]
                ):
                    collision = "detector_exact_or_pixel_collision"
                elif _near_any(bits, cast(list[str], hashes["detector_dhash"])):
                    collision = "detector_perceptual_collision"
                elif byte_sha in cast(set[str], hashes["classifier_sha"]) or pixel in cast(
                    set[str], hashes["classifier_pixel"]
                ):
                    collision = "classifier_full_frame_exact_or_pixel_collision"
                elif _near_any(bits, cast(list[str], hashes["classifier_dhash"])):
                    collision = "classifier_full_frame_perceptual_collision"
                elif (
                    byte_sha in selected_images
                    or pixel in selected_pixels
                    or _near_any(bits, selected_dhash)
                ):
                    collision = "phase_e_external_duplicate"
                else:
                    for crop_sha, crop_pixel, crop_dhash in crop_hashes:
                        if crop_sha in cast(set[str], hashes["detector_sha"]) or crop_pixel in cast(
                            set[str], hashes["detector_pixel"]
                        ):
                            collision = "detector_crop_exact_or_pixel_collision"
                            break
                        if _near_any(crop_dhash, cast(list[str], hashes["detector_dhash"])):
                            collision = "detector_crop_perceptual_collision"
                            break
                        if crop_sha in cast(
                            set[str], hashes["classifier_sha"]
                        ) or crop_pixel in cast(set[str], hashes["classifier_pixel"]):
                            collision = "classifier_crop_exact_or_pixel_collision"
                            break
                        if _near_any(crop_dhash, cast(list[str], hashes["classifier_dhash"])):
                            collision = "classifier_crop_perceptual_collision"
                            break
                if collision:
                    temp.unlink(missing_ok=True)
                    exclusions.append({"source_image_id": candidate.image_id, "reason": collision})
                    continue
                suffix = Path(httpx.URL(candidate.image_url).path).suffix or ".jpg"
                image_path = destination / f"bdd100k_{candidate.image_id}{suffix}"
                temp.replace(image_path)
                selected_images.add(byte_sha)
                selected_pixels.add(pixel)
                selected_dhash.append(bits)
                image_id = hashlib.sha256(f"phase-e-bdd:{byte_sha}".encode()).hexdigest()[:24]
                for index, (box, crop_values) in enumerate(
                    zip(candidate.boxes, crop_hashes, strict=True), start=1
                ):
                    x1, y1, x2, y2 = box
                    crop_sha, crop_pixel, crop_dhash = crop_values
                    area = (x2 - x1) * (y2 - y1) / max(1.0, width * height)
                    rows.append(
                        {
                            "benchmark_id": BENCHMARK_ID,
                            "instance_id": f"{image_id}:box-{index:03d}",
                            "image_id": image_id,
                            "image_path": _project_rel(root, image_path),
                            "expected_kind": "sign",
                            "domain": "external_non_malaysian",
                            "source_kind": "bdd100k",
                            "source_id": BDD_DATASET,
                            "source_revision": BDD_REVISION,
                            "source_row_index": candidate.row_index,
                            "source_image_id": candidate.image_id,
                            "condition_bucket": bucket,
                            "scene": candidate.scene,
                            "weather": candidate.weather,
                            "timeofday": candidate.timeofday,
                            "image_width": width,
                            "image_height": height,
                            "x1": f"{x1:.6f}",
                            "y1": f"{y1:.6f}",
                            "x2": f"{x2:.6f}",
                            "y2": f"{y2:.6f}",
                            "area_ratio": f"{area:.10f}",
                            "small_sign": str(area <= 0.01).lower(),
                            "very_small_sign": str(area <= 0.001).lower(),
                            "source_semantic_hint": "",
                            "semantic_sign_id": "",
                            "parameter_value": "",
                            "parameter_unit": "",
                            "bbox_review_decision": "",
                            "review_decision": "",
                            "reviewer": "",
                            "reviewed_at": "",
                            "reviewer_notes": "",
                            "sha256": byte_sha,
                            "pixel_sha256": pixel,
                            "dhash64": bits,
                            "crop_sha256": crop_sha,
                            "crop_pixel_sha256": crop_pixel,
                            "crop_dhash64": crop_dhash,
                            "licence_status": "assignment_only_waiver",
                            "use_policy": "assignment_only_no_redistribution_no_shared_dvc",
                            "evaluation_only": "true",
                            "included_in_training": "false",
                        }
                    )
                counts[bucket] += 1
            if counts[bucket] != EXTERNAL_PER_BUCKET:
                raise RuntimeError(
                    f"Could not materialize {EXTERNAL_PER_BUCKET} unique {bucket} frames"
                )
    return rows, {
        "selected_images": sum(counts.values()),
        "selected_boxes": len(rows),
        "bucket_counts": dict(counts),
        "exclusions": exclusions,
    }


def _read_yolo_boxes(
    label_path: Path, width: int, height: int
) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Invalid Phase E one-class label: {label_path}: {raw}")
        cx, cy, box_width, box_height = (float(value) for value in parts[1:])
        x1 = (cx - box_width / 2) * width
        y1 = (cy - box_height / 2) * height
        x2 = (cx + box_width / 2) * width
        y2 = (cy + box_height / 2) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def _emtd_hints(root: Path) -> dict[str, list[dict[str, str]]]:
    hints: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(root / "data/annotations/emtd_boxes.csv"):
        hints[row["sample_id"]].append(row)
    return dict(hints)


def _in_domain_rows(root: Path) -> list[dict[str, object]]:
    manifest = _read_csv(root / "data/manifests/detector_production_assignment_v1_20260829.csv")
    selected = [
        row
        for row in manifest
        if row.get("split") == "test"
        and row.get("layout_root_id") in {"source/EMTD_YOLO_33", "emtd", "negative"}
    ]
    if len(selected) != 317:
        raise ValueError(f"Expected 317 locked-test Phase E images, found {len(selected)}")
    hints = _emtd_hints(root)
    rows: list[dict[str, object]] = []
    positive_images = 0
    positive_boxes = 0
    multi_sign_images = 0
    for source in sorted(selected, key=lambda row: row["sample_id"]):
        image_path = _resolve(root, source["dataset_image_path"])
        label_path = _resolve(root, source["dataset_label_path"])
        pixel, bits, _, width, height = _path_hashes(image_path)
        if _sha256(image_path) != source["sha256"] or pixel != source["pixel_sha256"]:
            raise ValueError(f"Locked detector image hash mismatch: {image_path}")
        boxes = _read_yolo_boxes(label_path, width, height)
        if source["layout_root_id"] == "negative":
            if boxes:
                raise ValueError(f"No-sign image has labels: {label_path}")
            rows.append(
                {
                    "benchmark_id": BENCHMARK_ID,
                    "instance_id": f"{source['sample_id']}:no-sign",
                    "image_id": source["sample_id"],
                    "image_path": source["dataset_image_path"],
                    "expected_kind": "no_sign",
                    "domain": "in_domain_locked_test",
                    "source_kind": source["source_kind"],
                    "source_id": source["source_id"],
                    "source_revision": "detector_production_assignment_v1_20260829",
                    "source_row_index": "",
                    "source_image_id": source["source_image_id"],
                    "condition_bucket": "no_sign",
                    "scene": "",
                    "weather": "",
                    "timeofday": "",
                    "image_width": width,
                    "image_height": height,
                    "semantic_sign_id": "no_sign",
                    "bbox_review_decision": "accept",
                    "review_decision": "accept",
                    "reviewer": "project_owner_phase_c",
                    "reviewed_at": "2026-08-29",
                    "reviewer_notes": "Inherited from the explicit Phase C owner no-sign review.",
                    "sha256": source["sha256"],
                    "pixel_sha256": pixel,
                    "dhash64": bits,
                    "licence_status": source["licence_status"],
                    "use_policy": source["use_policy"],
                    "evaluation_only": "true",
                    "included_in_training": "false",
                }
            )
            continue
        positive_images += 1
        positive_boxes += len(boxes)
        multi_sign_images += int(len(boxes) > 1)
        source_hint_rows = (
            hints.get(Path(source["source_path"]).stem, [])
            if source["layout_root_id"] == "emtd"
            else []
        )
        for index, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1) / max(1.0, width * height)
            with Image.open(image_path) as raw:
                crop_sha, crop_pixel, crop_dhash = _crop_hashes(
                    ImageOps.exif_transpose(raw).convert("RGB"), box
                )
            hint = source_hint_rows[index - 1] if index <= len(source_hint_rows) else {}
            rows.append(
                {
                    "benchmark_id": BENCHMARK_ID,
                    "instance_id": f"{source['sample_id']}:box-{index:03d}",
                    "image_id": source["sample_id"],
                    "image_path": source["dataset_image_path"],
                    "expected_kind": "sign",
                    "domain": "in_domain_locked_test",
                    "source_kind": source["source_kind"],
                    "source_id": source["source_id"],
                    "source_revision": "detector_production_assignment_v1_20260829",
                    "source_row_index": "",
                    "source_image_id": source["source_image_id"],
                    "condition_bucket": "full_road",
                    "scene": "",
                    "weather": "",
                    "timeofday": "",
                    "image_width": width,
                    "image_height": height,
                    "x1": f"{x1:.6f}",
                    "y1": f"{y1:.6f}",
                    "x2": f"{x2:.6f}",
                    "y2": f"{y2:.6f}",
                    "area_ratio": f"{area:.10f}",
                    "small_sign": str(area <= 0.01).lower(),
                    "very_small_sign": str(area <= 0.001).lower(),
                    "source_semantic_hint": hint.get("semantic_sign_id", ""),
                    "semantic_sign_id": "",
                    "parameter_value": "",
                    "parameter_unit": "",
                    "bbox_review_decision": "accept",
                    "review_decision": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "reviewer_notes": "",
                    "sha256": source["sha256"],
                    "pixel_sha256": pixel,
                    "dhash64": bits,
                    "crop_sha256": crop_sha,
                    "crop_pixel_sha256": crop_pixel,
                    "crop_dhash64": crop_dhash,
                    "licence_status": source["licence_status"],
                    "use_policy": source["use_policy"],
                    "evaluation_only": "true",
                    "included_in_training": "false",
                }
            )
    if (positive_images, positive_boxes, multi_sign_images) != (197, 351, 94):
        raise ValueError(
            "Locked Phase E full-road inventory changed: "
            f"found {positive_images} images, {positive_boxes} boxes, {multi_sign_images} multi-sign images"
        )
    return rows


def _render_review_package(root: Path, rows: list[dict[str, object]], review_root: Path) -> None:
    overlays = review_root / "overlays"
    crops = review_root / "crops"
    overlays.mkdir(parents=True, exist_ok=True)
    crops.mkdir(parents=True, exist_ok=True)
    by_image: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["expected_kind"] == "sign":
            by_image[str(row["image_id"])].append(row)
    overlay_tiles: list[ReviewTile] = []
    crop_tiles: list[ReviewTile] = []
    for image_id, image_rows in sorted(by_image.items()):
        image_path = _resolve(root, str(image_rows[0]["image_path"]))
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        drawn = image.copy()
        draw = ImageDraw.Draw(drawn)
        for row in image_rows:
            box = tuple(float(cast(Any, row[key])) for key in ("x1", "y1", "x2", "y2"))
            x1, y1, x2, y2 = box
            label = str(row["instance_id"]).rsplit(":", 1)[-1]
            draw.rectangle(box, outline=(255, 48, 48), width=max(2, round(min(image.size) / 300)))
            draw.text((x1 + 2, max(0, y1 - 14)), label, fill=(255, 48, 48))
            pad_x = (x2 - x1) * 0.20
            pad_y = (y2 - y1) * 0.20
            crop = image.crop(
                (
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(image.width, x2 + pad_x),
                    min(image.height, y2 + pad_y),
                )
            )
            crop_path = crops / f"{str(row['instance_id']).replace(':', '_')}.jpg"
            crop.save(crop_path, quality=92)
            crop_tiles.append(ReviewTile(label=str(row["instance_id"]), image_path=crop_path))
        overlay_path = overlays / f"{image_id}.jpg"
        drawn.save(overlay_path, quality=92)
        overlay_tiles.append(
            ReviewTile(label=f"{image_id} | {image_rows[0]['domain']}", image_path=overlay_path)
        )
    for index in range(0, len(overlay_tiles), 20):
        render_contact_sheet(
            overlay_tiles[index : index + 20],
            review_root / f"frames_{index // 20 + 1:02d}.jpg",
            columns=5,
            tile_width=260,
            tile_height=180,
        )
    for index in range(0, len(crop_tiles), 30):
        render_contact_sheet(
            crop_tiles[index : index + 30],
            review_root / f"signs_{index // 30 + 1:02d}.jpg",
            columns=6,
            tile_width=190,
            tile_height=160,
        )


def prepare_phase_e_benchmark(
    *,
    project_root: Path = PROJECT_ROOT,
    overwrite: bool = False,
    collect_external: bool = True,
) -> dict[str, object]:
    root = project_root.resolve()
    inventory_path = _resolve(root, INVENTORY_PATH)
    review_path = _resolve(root, REVIEW_PATH)
    canonical_path = _resolve(root, CANONICAL_PATH)
    review_root = _resolve(root, REVIEW_ROOT)
    audit_root = _resolve(root, AUDIT_ROOT)
    external_root = _resolve(root, EXTERNAL_ROOT)
    if canonical_path.exists():
        raise FileExistsError(f"The immutable Phase E benchmark already exists: {canonical_path}")
    if review_path.exists():
        existing = _read_csv(review_path)
        if any(
            row.get("expected_kind") == "sign" and row.get("review_decision", "").strip()
            for row in existing
        ):
            raise FileExistsError("Refusing to overwrite Phase E owner review decisions")
        if not overwrite:
            raise FileExistsError(f"Phase E review inventory already exists: {review_path}")
    for path in (review_root, audit_root, external_root):
        if path.exists() and any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Refusing to mix Phase E preparation artifacts: {path}")
            shutil.rmtree(path)
    rows = _in_domain_rows(root)
    external_audit: dict[str, object] = {
        "selected_images": 0,
        "selected_boxes": 0,
        "bucket_counts": {},
    }
    if collect_external:
        external_rows, external_audit = _download_external_rows(root, external_root)
        rows.extend(external_rows)
    rows.sort(key=lambda row: str(row["instance_id"]))
    _write_csv(inventory_path, rows)
    _write_csv(review_path, rows)
    _render_review_package(root, rows, review_root)
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "benchmark_id": BENCHMARK_ID,
        "status": "owner_review_required",
        "selection_seed": SEED,
        "locked_test_images": 317,
        "locked_test_positive_images": 197,
        "locked_test_no_sign_images": 120,
        "locked_test_boxes": 351,
        "locked_test_multi_sign_images": 94,
        "external": external_audit,
        "image_count": len({str(row["image_id"]) for row in rows}),
        "sign_count": sum(row["expected_kind"] == "sign" for row in rows),
        "unresolved_review_rows": sum(
            row["expected_kind"] == "sign" and not str(row["review_decision"]).strip()
            for row in rows
        ),
        "review_rows": len(rows),
        "review_path": _project_rel(root, review_path),
        "review_root": _project_rel(root, review_root),
        "predictions_rendered_for_review": False,
        "evaluation_only": True,
        "included_in_training": False,
        "licence_status": "assignment_only_waiver",
        "use_policy": "assignment_only_no_redistribution_no_shared_dvc",
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_json(audit_root / "preparation_audit.json", summary)
    return summary


def validate_phase_e_review(
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    root = project_root.resolve()
    review_path = _resolve(root, REVIEW_PATH)
    canonical_path = _resolve(root, CANONICAL_PATH)
    audit_root = _resolve(root, AUDIT_ROOT)
    if not review_path.is_file():
        raise FileNotFoundError(review_path)
    inventory_path = _resolve(root, INVENTORY_PATH)
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    review_sha = _sha256(review_path)
    freeze_path = audit_root / "freeze_audit.json"
    if canonical_path.exists():
        if not freeze_path.is_file():
            raise RuntimeError("Canonical Phase E manifest exists without a freeze audit")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("review_sha256") != review_sha:
            raise RuntimeError("Owner review changed after the Phase E benchmark was frozen")
        return freeze
    rows = _read_csv(review_path)
    inventory_rows = _read_csv(inventory_path)
    inventory_by_instance = {row["instance_id"]: row for row in inventory_rows}
    catalogue = catalogue_by_id()
    supported_labels_path = _resolve(root, SUPPORTED_LABELS_PATH)
    raw_supported_labels: object = json.loads(
        supported_labels_path.read_text(encoding="utf-8")
    )
    if not isinstance(raw_supported_labels, list):
        raise ValueError("Phase E requires the exact 78-class classifier label list")
    supported_labels = cast(list[object], raw_supported_labels)
    if len(supported_labels) != 78:
        raise ValueError("Phase E requires the exact 78-class classifier label list")
    if not all(isinstance(label, str) for label in supported_labels):
        raise ValueError("Phase E classifier labels must be semantic string identifiers")
    allowed_semantics = {str(label) for label in supported_labels} | {
        "out_of_ontology",
        "unreadable",
    }
    blockers: list[str] = []
    exclusions: list[dict[str, str]] = []
    accepted: list[dict[str, str]] = []
    seen_instances: set[str] = set()
    image_hashes: dict[str, tuple[str, str]] = {}
    if len(inventory_by_instance) != len(inventory_rows) or {
        row.get("instance_id", "") for row in rows
    } != set(inventory_by_instance):
        blockers.append("review_row_set_changed_from_preparation_inventory")
    blockers.extend(_external_collision_reasons(rows, _existing_hashes(root)))
    for row in rows:
        instance_id = row.get("instance_id", "")
        if not instance_id or instance_id in seen_instances:
            blockers.append(f"duplicate_or_missing_instance:{instance_id}")
            continue
        seen_instances.add(instance_id)
        inventory_row = inventory_by_instance.get(instance_id)
        if inventory_row is None or any(
            row.get(field, "") != inventory_row.get(field, "") for field in IMMUTABLE_REVIEW_FIELDS
        ):
            blockers.append(f"immutable_review_field_changed:{instance_id}")
            continue
        image_path = _resolve(root, row["image_path"])
        if not image_path.is_file():
            blockers.append(f"missing_image:{instance_id}")
            continue
        actual = image_hashes.get(row["image_id"])
        if actual is None:
            actual = (_sha256(image_path), _path_hashes(image_path)[0])
            image_hashes[row["image_id"]] = actual
        if actual != (row["sha256"], row["pixel_sha256"]):
            blockers.append(f"image_hash_mismatch:{instance_id}")
        decision = row.get("review_decision", "").strip().casefold()
        if row.get("expected_kind") == "no_sign":
            if (
                decision not in {"accept", "accepted", "approve", "approved"}
                or row.get("semantic_sign_id") != "no_sign"
            ):
                blockers.append(f"invalid_no_sign_review:{instance_id}")
            else:
                accepted.append(row)
            continue
        if decision in {"reject", "rejected"}:
            if row.get("domain") == "in_domain_locked_test":
                blockers.append(f"locked_test_box_rejected:{instance_id}")
            else:
                exclusions.append(
                    {"instance_id": instance_id, "reason": "owner_rejected_external_box"}
                )
            continue
        if decision not in {"accept", "accepted", "approve", "approved"}:
            blockers.append(f"unresolved_review:{instance_id}")
            continue
        if row.get("bbox_review_decision", "").strip().casefold() not in {
            "accept",
            "accepted",
            "approve",
            "approved",
            "adjusted",
        }:
            blockers.append(f"unresolved_bbox_review:{instance_id}")
        if row.get("semantic_sign_id", "").strip() not in allowed_semantics:
            blockers.append(f"invalid_semantic_label:{instance_id}")
        if not row.get("reviewer", "").strip() or not row.get("reviewed_at", "").strip():
            blockers.append(f"missing_reviewer_evidence:{instance_id}")
        try:
            width = float(row["image_width"])
            height = float(row["image_height"])
            x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError
        except (KeyError, ValueError):
            blockers.append(f"invalid_bbox:{instance_id}")
            continue
        semantic_id = row.get("semantic_sign_id", "").strip()
        definition = catalogue.get(semantic_id)
        numeric_types = {
            ParameterType.SPEED,
            ParameterType.HEIGHT,
            ParameterType.WIDTH,
            ParameterType.WEIGHT,
            ParameterType.AXLE_WEIGHT,
            ParameterType.DISTANCE,
        }
        if definition is not None and definition.parameter_type in numeric_types:
            try:
                parameter = float(row.get("parameter_value", ""))
                if parameter <= 0 or not row.get("parameter_unit", "").strip():
                    raise ValueError
            except ValueError:
                blockers.append(f"missing_or_invalid_numeric_parameter:{instance_id}")
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            crop_sha, crop_pixel, crop_dhash = _crop_hashes(image, (x1, y1, x2, y2))
        row["crop_sha256"] = crop_sha
        row["crop_pixel_sha256"] = crop_pixel
        row["crop_dhash64"] = crop_dhash
        area = (x2 - x1) * (y2 - y1) / max(1.0, width * height)
        row["area_ratio"] = f"{area:.10f}"
        row["small_sign"] = str(area <= 0.01).lower()
        row["very_small_sign"] = str(area <= 0.001).lower()
        accepted.append(row)
    if blockers:
        _write_json(
            audit_root / "review_validation_failure.json",
            {
                "benchmark_id": BENCHMARK_ID,
                "status": "blocked",
                "blocker_count": len(blockers),
                "blockers": blockers,
            },
        )
        raise ValueError(
            f"Phase E owner review is incomplete or invalid ({len(blockers)} blockers)"
        )
    in_domain = [row for row in accepted if row["domain"] == "in_domain_locked_test"]
    external = [row for row in accepted if row["domain"] == "external_non_malaysian"]
    in_domain_signs = [row for row in in_domain if row["expected_kind"] == "sign"]
    in_domain_no_sign = [row for row in in_domain if row["expected_kind"] == "no_sign"]
    external_image_ids = {row["image_id"] for row in external}
    if (
        len({row["image_id"] for row in in_domain}) != 317
        or len({row["image_id"] for row in in_domain_signs}) != 197
        or len(in_domain_signs) != 351
        or len(in_domain_no_sign) != 120
    ):
        raise ValueError("Owner review no longer preserves the locked Phase E in-domain benchmark")
    if len(external_image_ids) != EXTERNAL_TARGET:
        raise ValueError(
            f"Owner review must retain 60 BDD100K positive frames; found {len(external_image_ids)}"
        )
    accepted.sort(key=lambda row: row["instance_id"])
    _write_csv(canonical_path, accepted)
    freeze: dict[str, object] = {
        "schema_version": "1.0",
        "benchmark_id": BENCHMARK_ID,
        "status": "frozen",
        "review_sha256": review_sha,
        "inventory_sha256": _sha256(inventory_path),
        "preparation_audit_sha256": _sha256(audit_root / "preparation_audit.json"),
        "canonical_sha256": _sha256(canonical_path),
        "canonical_rows": len(accepted),
        "images": len({row["image_id"] for row in accepted}),
        "sign_boxes": sum(row["expected_kind"] == "sign" for row in accepted),
        "no_sign_images": sum(row["expected_kind"] == "no_sign" for row in accepted),
        "external_exclusions": exclusions,
        "evaluation_only": True,
        "included_in_training": False,
        "test_retuning_allowed": False,
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    _write_json(freeze_path, freeze)
    return freeze


def phase_e_benchmark_status(*, project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    root = project_root.resolve()
    review_path = _resolve(root, REVIEW_PATH)
    canonical_path = _resolve(root, CANONICAL_PATH)
    if canonical_path.is_file():
        freeze_path = _resolve(root, AUDIT_ROOT / "freeze_audit.json")
        if not freeze_path.is_file():
            raise RuntimeError("Canonical Phase E manifest exists without a freeze audit")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("canonical_sha256") != _sha256(canonical_path):
            raise RuntimeError("Immutable Phase E canonical manifest hash changed after freeze")
        rows = _read_csv(canonical_path)
        status: dict[str, object] = {
            "benchmark_id": BENCHMARK_ID,
            "status": "frozen",
            "rows": len(rows),
            "images": len({row["image_id"] for row in rows}),
            "canonical_path": _project_rel(root, canonical_path),
        }
        return status
    if not review_path.is_file():
        return {"benchmark_id": BENCHMARK_ID, "status": "not_prepared"}
    rows = _read_csv(review_path)
    unresolved = sum(
        row.get("expected_kind") == "sign" and not row.get("review_decision", "").strip()
        for row in rows
    )
    status = {
        "benchmark_id": BENCHMARK_ID,
        "status": "owner_review_required" if unresolved else "ready_to_validate",
        "rows": len(rows),
        "images": len({row["image_id"] for row in rows}),
        "unresolved_sign_rows": unresolved,
        "review_path": _project_rel(root, review_path),
    }
    return status
