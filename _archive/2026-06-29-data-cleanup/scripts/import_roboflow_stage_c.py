from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DETECTION_ID = "roboflow_malaysia_road_sign_v1"
SOURCE_CLASSIFICATION_ID = "roboflow_dr_samsudin_malaysia_road_sign"

DETECTION_ROOT = PROJECT_ROOT / "data/raw/roboflow/malaysia_road_sign_dataset_v1/extracted_full"
CLASSIFICATION_ROOT = PROJECT_ROOT / "data/raw/roboflow/dr_samsudin_malaysia_road_sign/extracted"
STAGING_ROOT = PROJECT_ROOT / "data/staging/roboflow"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/roboflow_source_manifest.csv"
CLASS_MAPPING_PATH = PROJECT_ROOT / "data/manifests/roboflow_class_mapping.csv"
AUDIT_PATH = PROJECT_ROOT / "outputs/audit/roboflow_import_audit.json"
CONTACT_SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_roboflow_import_sheets"

MIN_CROP_SIDE = 32
MIN_CROP_AREA = 1024
CROP_PADDING_FRACTION = 0.08
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class MappingDecision:
    semantic_sign_id: str
    action: str
    confidence: str
    reason: str

    @property
    def include(self) -> bool:
        return self.action in {"map_existing", "new_class_candidate"}


DETECTION_MAPPING: dict[str, MappingDecision] = {
    "Bicycle lane": MappingDecision("bicycles_only", "map_existing", "medium", "Blue bicycle-only/lane sign is visually closest to existing bicycles_only."),
    "Bumps": MappingDecision("road_hump", "map_existing", "high", "Same road hump/bump warning meaning."),
    "Bumps ahead": MappingDecision("road_hump", "map_existing", "high", "Same road hump/bump warning meaning."),
    "Bus stop": MappingDecision("bus_stop", "map_existing", "high", "Same informational bus stop sign."),
    "Camera operation zone": MappingDecision("camera_enforcement", "map_existing", "high", "Same enforcement camera warning/information sign."),
    "Chevron -left-": MappingDecision("chevron_left", "map_existing", "high", "Left chevron alignment marker."),
    "Chevron -right-": MappingDecision("chevron_right", "map_existing", "high", "Right chevron alignment marker."),
    "Children": MappingDecision("children_crossing", "map_existing", "high", "Children/school crossing warning."),
    "Construction sign": MappingDecision("roadworks", "map_existing", "medium", "Temporary construction/roadworks sign."),
    "Cow nearby": MappingDecision("animal_crossing", "map_existing", "high", "Animal/cattle crossing warning."),
    "Crossroad": MappingDecision("crossroads", "map_existing", "high", "Crossroads warning."),
    "Crossroad on the left": MappingDecision("side_road_left", "map_existing", "medium", "Side/cross road entering from left."),
    "Crossroad on the right": MappingDecision("side_road_right", "map_existing", "medium", "Side/cross road entering from right."),
    "Crosswind area": MappingDecision("crosswind", "map_existing", "high", "Crosswind warning."),
    "Double Bend to Left Ahead": MappingDecision("double_curve", "map_existing", "medium", "Existing catalogue keeps double-curve as one semantic class."),
    "Double Bend to Right Ahead": MappingDecision("double_curve", "map_existing", "medium", "Existing catalogue keeps double-curve as one semantic class."),
    "Expressway signs 1": MappingDecision("expressway", "map_existing", "medium", "Expressway information/direction sign family."),
    "Expressway signs 2": MappingDecision("expressway", "map_existing", "medium", "Expressway information/direction sign family."),
    "Flagman ahead": MappingDecision("flagman_ahead", "map_existing", "high", "Same temporary flagman warning."),
    "Give way": MappingDecision("give_way", "map_existing", "high", "Same give-way sign."),
    "Height limit": MappingDecision("height_restriction", "map_existing", "high", "Same height restriction sign."),
    "Horn Prohibited": MappingDecision("no_horn", "map_existing", "high", "Same no-horn prohibition."),
    "Left Bend Ahead": MappingDecision("curve_left", "map_existing", "high", "Left curve warning."),
    "Level crossing with gates ahead": MappingDecision("railway_crossing", "map_existing", "high", "Railway/level crossing warning."),
    "Motorcycles only": MappingDecision("motorcycles_only", "new_class_candidate", "high", "Mandatory motorcycles-only sign is not in the approved catalogue."),
    "Narrow bridge": MappingDecision("narrow_bridge", "new_class_candidate", "high", "Bridge-specific narrow warning is distinct from generic road narrows."),
    "No Stopping": MappingDecision("no_stopping", "map_existing", "high", "Same no-stopping prohibition."),
    "No U-turns": MappingDecision("no_u_turn", "map_existing", "high", "Same no-U-turn prohibition."),
    "No entry": MappingDecision("no_entry", "map_existing", "high", "Same no-entry sign."),
    "No left turn": MappingDecision("no_left_turn", "map_existing", "high", "Same no-left-turn prohibition."),
    "No overtaking": MappingDecision("no_overtaking", "map_existing", "high", "Same no-overtaking sign."),
    "No parking": MappingDecision("no_stopping", "map_existing", "medium", "Source crops show the red-X no-stopping style sign, not a single-slash no-parking sign."),
    "No right turn": MappingDecision("no_right_turn", "map_existing", "high", "Same no-right-turn prohibition."),
    "Obstruction": MappingDecision("obstruction_ahead", "map_existing", "high", "Same obstruction-ahead warning."),
    "Other dangers nearby": MappingDecision("general_caution", "map_existing", "high", "General caution/other danger warning."),
    "Parking area": MappingDecision("parking", "map_existing", "high", "Same parking information sign; known outlier filenames are filtered."),
    "Pass either side": MappingDecision("", "exclude_noisy_mixed_source", "high", "Source class mixes speed-limit signs and pass-either-side signs."),
    "Pass on the left": MappingDecision("keep_left", "map_existing", "high", "Mandatory pass/keep left."),
    "Pass on the right": MappingDecision("keep_right", "map_existing", "high", "Mandatory pass/keep right."),
    "Right Bend Ahead": MappingDecision("curve_right", "map_existing", "high", "Right curve warning."),
    "Road cones": MappingDecision("", "exclude_non_sign", "high", "Traffic cone object, not a road sign class."),
    "Road narrows on the left": MappingDecision("road_narrows_left", "map_existing", "high", "Same road-narrows-left warning."),
    "Road narrows on the right": MappingDecision("road_narrows_right", "map_existing", "high", "Same road-narrows-right warning."),
    "Road work": MappingDecision("roadworks", "map_existing", "high", "Same roadworks warning."),
    "Roadway diverges": MappingDecision("", "exclude_noisy_mixed_source", "high", "Source class is visually mixed with obstruction/caution signs."),
    "Roundabout ahead": MappingDecision("roundabout_ahead", "map_existing", "high", "Same roundabout-ahead warning."),
    "Slippery road": MappingDecision("slippery_road", "map_existing", "high", "Same slippery-road warning."),
    "Speed limit": MappingDecision("maximum_speed", "map_existing", "high", "Same maximum-speed sign; numeric value remains OCR/parameter work."),
    "Stop": MappingDecision("stop", "map_existing", "high", "Same stop sign."),
    "T-junction": MappingDecision("t_junction", "map_existing", "high", "Same T-junction warning."),
    "Towing area": MappingDecision("tow_away_zone", "new_class_candidate", "medium", "Tow-away/towing zone sign is distinct from no parking/no stopping."),
    "Traffic from Left Merges Ahead": MappingDecision("merge_left", "map_existing", "medium", "Traffic merging from left."),
    "Traffic from Right Merges Ahead": MappingDecision("merge_right", "map_existing", "medium", "Traffic merging from right."),
    "Traffic merging from the left": MappingDecision("merge_left", "map_existing", "high", "Same merge-left warning."),
    "Traffic merging from the right": MappingDecision("merge_right", "map_existing", "high", "Same merge-right warning."),
    "Traffic merging to the left": MappingDecision("lane_merges_left", "new_class_candidate", "medium", "Lane merges left, direction differs from traffic merging from left."),
    "Traffic signals ahead": MappingDecision("traffic_signal_ahead", "map_existing", "high", "Same traffic-signal-ahead warning."),
    "Train Gate": MappingDecision("railway_crossing", "map_existing", "high", "Railway gate/crossing warning."),
    "U turn": MappingDecision("permitted_u_turn", "map_existing", "high", "Permitted U-turn sign; known speed-limit outlier filenames are filtered."),
    "Weight limit": MappingDecision("weight_restriction", "map_existing", "high", "Same weight restriction sign."),
    "Zebra crossing": MappingDecision("pedestrian_crossing", "map_existing", "high", "Same pedestrian/zebra crossing warning."),
    "pedestrian crossing opt1": MappingDecision("pedestrian_crossing", "map_existing", "high", "Same pedestrian crossing warning."),
}


CLASSIFICATION_MAPPING: dict[str, MappingDecision] = {
    "Bumper": MappingDecision("road_hump", "map_existing", "high", "Same road hump/bump warning."),
    "Bumper 2": MappingDecision("road_hump", "map_existing", "high", "Same road hump/bump warning."),
    "Give Way": MappingDecision("give_way", "map_existing", "high", "Same give-way sign."),
    "Give Way 2": MappingDecision("give_way", "map_existing", "high", "Same give-way sign with Bahasa text."),
    "Height Limit": MappingDecision("height_restriction", "map_existing", "high", "Same height restriction sign."),
    "Height Limit 2": MappingDecision("height_restriction", "map_existing", "high", "Same height restriction sign."),
    "Narrow Road": MappingDecision("road_narrows_both", "map_existing", "high", "Generic road-narrows warning."),
    "Narrow Road 2": MappingDecision("road_narrows_both", "map_existing", "medium", "Generic road-narrows warning; non-Malaysian illustration retained as public mixed source."),
    "No Entry": MappingDecision("no_entry", "map_existing", "high", "Same no-entry sign."),
    "No Honking": MappingDecision("no_horn", "map_existing", "high", "Same no-horn prohibition."),
    "No Left Turn": MappingDecision("no_left_turn", "map_existing", "high", "Same no-left-turn prohibition."),
    "No Lorry": MappingDecision("no_heavy_vehicle", "map_existing", "high", "Same no-heavy-vehicle/no-lorry sign."),
    "No Motorcycle": MappingDecision("no_motorcycle", "map_existing", "high", "Same no-motorcycle prohibition."),
    "No Overtake": MappingDecision("no_overtaking", "map_existing", "high", "Same no-overtaking sign."),
    "No Parking": MappingDecision("no_parking", "map_existing", "high", "Same no-parking sign."),
    "No Right Turn": MappingDecision("no_right_turn", "map_existing", "high", "Same no-right-turn prohibition."),
    "No Stop": MappingDecision("no_stopping", "map_existing", "high", "Same no-stopping prohibition."),
    "No U-Turn": MappingDecision("no_u_turn", "map_existing", "high", "Same no-U-turn prohibition."),
    "OKU Parking": MappingDecision("disabled_parking", "new_class_candidate", "high", "Accessible/OKU parking is distinct from generic parking."),
    "One Way": MappingDecision("one_way", "map_existing", "high", "Same one-way sign."),
    "Parking": MappingDecision("parking", "map_existing", "high", "Same parking information sign."),
    "Pedestrian Crossing": MappingDecision("pedestrian_crossing", "map_existing", "high", "Same pedestrian crossing warning."),
    "Roadwork": MappingDecision("roadworks", "map_existing", "high", "Same roadworks warning."),
    "Roundabout": MappingDecision("roundabout_ahead", "map_existing", "medium", "Yellow roundabout warning, not the blue mandatory roundabout sign."),
    "Selekoh": MappingDecision("", "exclude_ambiguous_direction", "high", "Class mixes left and right chevron boards, losing the required direction."),
    "Slippery Road": MappingDecision("slippery_road", "map_existing", "high", "Same slippery-road warning."),
    "Speed Limit": MappingDecision("maximum_speed", "map_existing", "high", "Same maximum-speed sign; numeric value remains OCR/parameter work."),
    "Stop": MappingDecision("stop", "map_existing", "high", "Same stop sign."),
    "Traffic Light": MappingDecision("traffic_signal_ahead", "map_existing", "high", "Traffic-signal-ahead warning."),
    "U-Turn": MappingDecision("permitted_u_turn", "map_existing", "high", "Permitted U-turn sign."),
    "cattleCrossing": MappingDecision("animal_crossing", "map_existing", "high", "Animal/cattle crossing warning."),
    "cattleCrossing2": MappingDecision("animal_crossing", "map_existing", "high", "Animal/cattle crossing warning."),
    "crossroad": MappingDecision("crossroads", "map_existing", "high", "Same crossroads warning."),
    "deadEnd": MappingDecision("dead_end", "map_existing", "high", "Same dead-end sign."),
}


MANIFEST_FIELDS = [
    "source_id",
    "task_type",
    "split",
    "source_label",
    "source_class_index",
    "semantic_sign_id",
    "mapping_action",
    "mapping_confidence",
    "quality_status",
    "rejection_reason",
    "original_path",
    "staged_path",
    "image_width",
    "image_height",
    "bbox_xyxy",
    "crop_width",
    "crop_height",
    "sha256",
]

MAPPING_FIELDS = [
    "source_id",
    "task_type",
    "source_label",
    "source_class_index",
    "semantic_sign_id",
    "mapping_action",
    "mapping_confidence",
    "reason",
]


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def clean_generated_outputs() -> None:
    for path in (STAGING_ROOT, CONTACT_SHEET_ROOT):
        resolved = path.resolve()
        allowed = PROJECT_ROOT.resolve() in resolved.parents and "roboflow" in resolved.as_posix()
        if path.exists() and allowed:
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_yolo_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"names:\s*(\[.*\])", text, re.S)
    if not match:
        raise ValueError(f"Could not parse names from {path}")
    return list(ast.literal_eval(match.group(1)))


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    lower_stem = stem.lower()
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem.lower() == lower_stem:
            return path
    return None


def is_detection_filename_outlier(source_label: str, image_stem: str) -> bool:
    """Filter obvious source-label errors found during Stage C visual audit."""
    stem = image_stem.lower()
    if source_label == "Parking area" and stem.startswith("perintah-pilihan"):
        return True
    if source_label == "U turn" and stem.startswith("0_00004_00021"):
        return True
    if source_label == "Give way" and stem.startswith("1002_jpg"):
        return True
    return False


def stage_classification_dataset(
    manifest_rows: list[dict[str, object]],
    mapping_rows: list[dict[str, object]],
    seen_hashes: set[str],
) -> None:
    source_id = SOURCE_CLASSIFICATION_ID
    task_type = "classification_folder"
    for source_label, decision in sorted(CLASSIFICATION_MAPPING.items()):
        mapping_rows.append(mapping_row(source_id, task_type, source_label, "", decision))

    for split in ("train", "valid", "test"):
        split_root = CLASSIFICATION_ROOT / split
        for class_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
            source_label = class_root.name
            decision = CLASSIFICATION_MAPPING.get(
                source_label,
                MappingDecision("", "exclude_unmapped", "high", "No conservative mapping defined."),
            )
            for image_path in sorted(path for path in class_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                except OSError:
                    manifest_rows.append(
                        manifest_row(
                            source_id,
                            task_type,
                            split,
                            source_label,
                            "",
                            decision,
                            image_path,
                            None,
                            0,
                            0,
                            "",
                            0,
                            0,
                            "",
                            "rejected",
                            "image_open_error",
                        )
                    )
                    continue

                digest = file_sha256(image_path)
                if not decision.include:
                    status = "rejected"
                    reason = decision.action
                    staged_path = None
                elif digest in seen_hashes:
                    status = "rejected"
                    reason = "duplicate_exact_image"
                    staged_path = None
                else:
                    seen_hashes.add(digest)
                    staged_dir = STAGING_ROOT / source_id / "classification" / split / decision.semantic_sign_id
                    staged_dir.mkdir(parents=True, exist_ok=True)
                    staged_path = staged_dir / image_path.name
                    shutil.copy2(image_path, staged_path)
                    status = "accepted_public_mixed"
                    reason = ""

                manifest_rows.append(
                    manifest_row(
                        source_id,
                        task_type,
                        split,
                        source_label,
                        "",
                        decision,
                        image_path,
                        staged_path,
                        width,
                        height,
                        "",
                        width,
                        height,
                        digest,
                        status,
                        reason,
                    )
                )


def stage_detection_dataset(
    manifest_rows: list[dict[str, object]],
    mapping_rows: list[dict[str, object]],
    seen_crop_hashes: set[str],
) -> None:
    source_id = SOURCE_DETECTION_ID
    task_type = "yolov8_detection_crops"
    names = parse_yolo_names(DETECTION_ROOT / "data.yaml")
    for class_index, source_label in enumerate(names):
        decision = DETECTION_MAPPING.get(
            source_label,
            MappingDecision("", "exclude_unmapped", "high", "No conservative mapping defined."),
        )
        mapping_rows.append(mapping_row(source_id, task_type, source_label, str(class_index), decision))

    for split in ("train", "valid", "test"):
        images_dir = DETECTION_ROOT / split / "images"
        labels_dir = DETECTION_ROOT / split / "labels"
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = find_image(images_dir, label_path.stem)
            if image_path is None:
                continue
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    width, height = image.size
                    label_lines = label_path.read_text(encoding="utf-8").splitlines()
                    for line_index, line in enumerate(label_lines):
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            class_index = int(float(parts[0]))
                            cx, cy, bw, bh = (float(value) for value in parts[1:5])
                        except ValueError:
                            continue
                        source_label = names[class_index] if 0 <= class_index < len(names) else f"unknown_{class_index}"
                        decision = DETECTION_MAPPING.get(
                            source_label,
                            MappingDecision("", "exclude_unmapped", "high", "No conservative mapping defined."),
                        )
                        x1 = int(round((cx - bw / 2) * width))
                        y1 = int(round((cy - bh / 2) * height))
                        x2 = int(round((cx + bw / 2) * width))
                        y2 = int(round((cy + bh / 2) * height))
                        raw_xyxy = (x1, y1, x2, y2)
                        x1 = max(0, min(width, x1))
                        y1 = max(0, min(height, y1))
                        x2 = max(0, min(width, x2))
                        y2 = max(0, min(height, y2))
                        crop_w = max(0, x2 - x1)
                        crop_h = max(0, y2 - y1)

                        status = "accepted_public_mixed"
                        reason = ""
                        staged_path: Path | None = None
                        digest = ""

                        if not decision.include:
                            status = "rejected"
                            reason = decision.action
                        elif is_detection_filename_outlier(source_label, label_path.stem):
                            status = "rejected"
                            reason = "source_label_outlier_filename"
                        elif crop_w < MIN_CROP_SIDE or crop_h < MIN_CROP_SIDE:
                            status = "rejected"
                            reason = "tiny_crop"
                        elif crop_w * crop_h < MIN_CROP_AREA:
                            status = "rejected"
                            reason = "small_crop_area"
                        elif crop_w <= 0 or crop_h <= 0:
                            status = "rejected"
                            reason = "invalid_bbox"
                        else:
                            pad = int(round(max(crop_w, crop_h) * CROP_PADDING_FRACTION))
                            padded = (
                                max(0, x1 - pad),
                                max(0, y1 - pad),
                                min(width, x2 + pad),
                                min(height, y2 + pad),
                            )
                            crop = image.crop(padded)
                            crop_bytes = crop.tobytes()
                            digest = bytes_sha256(crop_bytes)
                            if digest in seen_crop_hashes:
                                status = "rejected"
                                reason = "duplicate_exact_crop"
                            else:
                                seen_crop_hashes.add(digest)
                                staged_dir = STAGING_ROOT / source_id / "classification_crops" / split / decision.semantic_sign_id
                                staged_dir.mkdir(parents=True, exist_ok=True)
                                staged_name = (
                                    f"{label_path.stem}_{line_index:02d}_"
                                    f"c{class_index}_{digest[:12]}.jpg"
                                )
                                staged_path = staged_dir / staged_name
                                crop.save(staged_path, quality=92)
                                crop_w, crop_h = crop.size

                        manifest_rows.append(
                            manifest_row(
                                source_id,
                                task_type,
                                split,
                                source_label,
                                str(class_index),
                                decision,
                                image_path,
                                staged_path,
                                width,
                                height,
                                ",".join(str(value) for value in raw_xyxy),
                                crop_w,
                                crop_h,
                                digest,
                                status,
                                reason,
                            )
                        )
            except OSError:
                continue


def mapping_row(
    source_id: str,
    task_type: str,
    source_label: str,
    source_class_index: str,
    decision: MappingDecision,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "task_type": task_type,
        "source_label": source_label,
        "source_class_index": source_class_index,
        "semantic_sign_id": decision.semantic_sign_id,
        "mapping_action": decision.action,
        "mapping_confidence": decision.confidence,
        "reason": decision.reason,
    }


def manifest_row(
    source_id: str,
    task_type: str,
    split: str,
    source_label: str,
    source_class_index: str,
    decision: MappingDecision,
    original_path: Path,
    staged_path: Path | None,
    image_width: int,
    image_height: int,
    bbox_xyxy: str,
    crop_width: int,
    crop_height: int,
    sha256: str,
    quality_status: str,
    rejection_reason: str,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "task_type": task_type,
        "split": split,
        "source_label": source_label,
        "source_class_index": source_class_index,
        "semantic_sign_id": decision.semantic_sign_id,
        "mapping_action": decision.action,
        "mapping_confidence": decision.confidence,
        "quality_status": quality_status,
        "rejection_reason": rejection_reason,
        "original_path": relative(original_path),
        "staged_path": relative(staged_path) if staged_path else "",
        "image_width": image_width,
        "image_height": image_height,
        "bbox_xyxy": bbox_xyxy,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "sha256": sha256,
    }


def build_audit(manifest_rows: list[dict[str, object]], mapping_rows: list[dict[str, object]]) -> dict[str, object]:
    by_source = Counter(row["source_id"] for row in manifest_rows)
    by_status = Counter(row["quality_status"] for row in manifest_rows)
    accepted_by_semantic = Counter(
        row["semantic_sign_id"]
        for row in manifest_rows
        if row["quality_status"] == "accepted_public_mixed"
    )
    rejected_by_reason = Counter(
        row["rejection_reason"]
        for row in manifest_rows
        if row["quality_status"] == "rejected"
    )
    mapping_actions = Counter(row["mapping_action"] for row in mapping_rows)
    new_class_candidates = sorted(
        {
            row["semantic_sign_id"]
            for row in mapping_rows
            if row["mapping_action"] == "new_class_candidate" and row["semantic_sign_id"]
        }
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            SOURCE_DETECTION_ID: relative(DETECTION_ROOT),
            SOURCE_CLASSIFICATION_ID: relative(CLASSIFICATION_ROOT),
        },
        "outputs": {
            "staging_root": relative(STAGING_ROOT),
            "manifest": relative(MANIFEST_PATH),
            "class_mapping": relative(CLASS_MAPPING_PATH),
            "contact_sheets": relative(CONTACT_SHEET_ROOT),
        },
        "quality_policy": {
            "minimum_crop_side": MIN_CROP_SIDE,
            "minimum_crop_area": MIN_CROP_AREA,
            "exact_duplicate_handling": "first accepted, later duplicates rejected",
            "status_meaning": {
                "accepted_public_mixed": "Mapped and staged, but still public-source data rather than final frozen training data.",
                "rejected": "Excluded from staged class folders because of mapping or quality rule.",
            },
        },
        "summary": {
            "source_rows": dict(sorted(by_source.items())),
            "quality_status_counts": dict(sorted(by_status.items())),
            "mapping_action_counts": dict(sorted(mapping_actions.items())),
            "accepted_class_count": len(accepted_by_semantic),
            "new_class_candidates": new_class_candidates,
            "rejection_counts": dict(sorted(rejected_by_reason.items())),
        },
        "accepted_by_semantic_sign_id": dict(sorted(accepted_by_semantic.items())),
    }


def write_audit(payload: dict[str, object]) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def render_contact_sheets(manifest_rows: list[dict[str, object]]) -> None:
    accepted = [row for row in manifest_rows if row["quality_status"] == "accepted_public_mixed" and row["staged_path"]]
    by_source_semantic: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in accepted:
        by_source_semantic[(str(row["source_id"]), str(row["semantic_sign_id"]))].append(row)

    font = ImageFont.load_default()
    for source_id in sorted({source for source, _semantic in by_source_semantic}):
        semantics = sorted(semantic for src, semantic in by_source_semantic if src == source_id)
        tile_w, tile_h = 210, 170
        columns = 4
        rows = (len(semantics) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * tile_w, max(1, rows) * tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        for index, semantic_id in enumerate(semantics):
            row = by_source_semantic[(source_id, semantic_id)][0]
            x = (index % columns) * tile_w
            y = (index // columns) * tile_h
            image_path = PROJECT_ROOT / str(row["staged_path"])
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((160, 112))
                    sheet.paste(image, (x + (tile_w - image.width) // 2, y + 8))
            except OSError:
                draw.rectangle([x + 8, y + 8, x + tile_w - 8, y + 120], outline="red")
            draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(210, 210, 210))
            count = len(by_source_semantic[(source_id, semantic_id)])
            draw.text((x + 6, y + 126), semantic_id[:34], fill=(0, 0, 0), font=font)
            draw.text((x + 6, y + 143), f"n={count}", fill=(70, 70, 70), font=font)
        output = CONTACT_SHEET_ROOT / f"{source_id}_accepted_classes.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, quality=92)


def main() -> None:
    if not DETECTION_ROOT.exists():
        raise FileNotFoundError(DETECTION_ROOT)
    if not CLASSIFICATION_ROOT.exists():
        raise FileNotFoundError(CLASSIFICATION_ROOT)

    clean_generated_outputs()
    manifest_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    stage_classification_dataset(manifest_rows, mapping_rows, seen_hashes=set())
    stage_detection_dataset(manifest_rows, mapping_rows, seen_crop_hashes=set())
    write_csv(MANIFEST_PATH, manifest_rows, MANIFEST_FIELDS)
    write_csv(CLASS_MAPPING_PATH, mapping_rows, MAPPING_FIELDS)
    audit = build_audit(manifest_rows, mapping_rows)
    write_audit(audit)
    render_contact_sheets(manifest_rows)

    print(f"Wrote {relative(MANIFEST_PATH)}")
    print(f"Wrote {relative(CLASS_MAPPING_PATH)}")
    print(f"Wrote {relative(AUDIT_PATH)}")
    print(f"Wrote contact sheets to {relative(CONTACT_SHEET_ROOT)}")
    print(json.dumps(audit["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
