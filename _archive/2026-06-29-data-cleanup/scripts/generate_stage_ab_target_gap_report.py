from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = PROJECT_ROOT / "configs/catalogue/malaysia_signs.v1.json"
COURSEWORK_MAPPING_PATH = PROJECT_ROOT / "configs/catalogue/coursework_draft_mapping.json"
COURSEWORK_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/coursework_manifest.csv"
P5_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/p5_label_qc_manifest.csv"
TARGET_CLASSES_PATH = PROJECT_ROOT / "data/manifests/target_sign_classes.csv"
COVERAGE_JSON_PATH = PROJECT_ROOT / "outputs/audit/target_class_coverage.json"
GAP_CSV_PATH = PROJECT_ROOT / "outputs/audit/data_gap_report.csv"
GAP_JSON_PATH = PROJECT_ROOT / "outputs/audit/data_gap_report.json"
WEAK_SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_b_weak_class_contact_sheets"


# These are project-scope defaults based on the assignment, Malaysian road-sign
# demo goal, and the user's requested live ADAS presentation direction.
COMMON_MUST_CLASSES = {
    "camera_enforcement",
    "children_crossing",
    "general_caution",
    "give_way",
    "height_restriction",
    "keep_left",
    "keep_right",
    "maximum_speed",
    "no_entry",
    "no_heavy_vehicle",
    "no_left_turn",
    "no_overtaking",
    "no_parking",
    "no_right_turn",
    "no_stopping",
    "no_u_turn",
    "pass_either_side",
    "pedestrian_crossing",
    "road_hump",
    "roadworks",
    "roundabout_ahead",
    "school_zone",
    "stop",
    "traffic_signal_ahead",
    "weight_restriction",
    "width_restriction",
}

DEFAULT_DEMO_CLASSES = {
    "camera_enforcement",
    "children_crossing",
    "give_way",
    "height_restriction",
    "maximum_speed",
    "no_entry",
    "no_parking",
    "no_stopping",
    "pedestrian_crossing",
    "road_hump",
    "roadworks",
    "roundabout_ahead",
    "school_zone",
    "stop",
    "traffic_signal_ahead",
}

COMMON_SHOULD_CLASSES = {
    "animal_crossing",
    "crossroads",
    "crosswind",
    "curve_left",
    "curve_right",
    "divided_road_begins",
    "divided_road_ends",
    "double_curve",
    "falling_rocks",
    "flagman_ahead",
    "flood_area",
    "low_clearance",
    "minimum_speed",
    "obstruction_ahead",
    "railway_crossing",
    "road_narrows_both",
    "road_narrows_left",
    "road_narrows_right",
    "side_road_left",
    "side_road_right",
    "slippery_road",
    "staggered_junction",
    "steep_ascent",
    "steep_descent",
    "t_junction",
    "temporary_speed_limit",
    "two_way_traffic",
    "uneven_road",
    "vehicle_collision_hazard",
}

OCR_VALUE_CLASSES = {
    "axle_weight_restriction",
    "end_speed_limit",
    "height_restriction",
    "low_clearance",
    "maximum_speed",
    "minimum_speed",
    "school_zone",
    "temporary_speed_limit",
    "weight_restriction",
    "width_restriction",
}

MULTILINGUAL_TEXT_CLASSES = {
    "give_way",
    "slow_text",
    "stop",
    "stop_for_checking",
}

# Classes with known crop-level noise history from the P5 owner review thread.
# Stage B still counts them, but the report keeps them out of "clean enough"
# claims until targeted QC/final freeze confirms them.
KNOWN_QC_RISK_CLASSES = {
    "animal_crossing",
    "height_restriction",
    "keep_left",
    "maximum_speed",
    "no_stopping",
    "obstruction_ahead",
    "side_road_left",
    "side_road_right",
    "vehicle_collision_hazard",
}

PRIORITY_ORDER = {"must": 0, "should": 1, "optional": 2}
STATUS_ORDER = {
    "missing_data": 0,
    "below_minimum": 1,
    "ready_count_qc_risk": 2,
    "split_gap": 3,
    "ready_minimum": 4,
    "ready_good": 5,
    "ready_strong": 6,
    "optional_no_data": 7,
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_catalogue() -> list[dict]:
    catalogue = read_json(CATALOGUE_PATH)
    return list(catalogue["entries"])


def load_coursework_summary() -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, int]]:
    mapping = read_json(COURSEWORK_MAPPING_PATH)["mappings"]
    sign_ids_by_semantic: dict[str, list[str]] = defaultdict(list)
    values_by_semantic: dict[str, list[str]] = defaultdict(list)
    for sign_id, row in mapping.items():
        semantic_id = row["semantic_sign_id"]
        sign_ids_by_semantic[semantic_id].append(sign_id)
        value = row.get("parameter_value")
        if value is not None:
            values_by_semantic[semantic_id].append(str(value))

    image_counts = Counter()
    for row in read_csv(COURSEWORK_MANIFEST_PATH):
        semantic_id = row.get("semantic_sign_id", "")
        if semantic_id:
            image_counts[semantic_id] += 1

    return sign_ids_by_semantic, values_by_semantic, dict(image_counts)


def load_p5_counts() -> tuple[Counter, dict[str, Counter], dict[str, list[dict[str, str]]], dict[str, int]]:
    label_counts: Counter = Counter()
    split_counts: dict[str, Counter] = defaultdict(Counter)
    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    tiny_counts: Counter = Counter()
    for row in read_csv(P5_MANIFEST_PATH):
        label = row.get("current_label", "")
        split = row.get("split", "")
        if not label:
            continue
        label_counts[label] += 1
        split_counts[label][split] += 1
        rows_by_label[label].append(row)
        if row.get("tiny_crop") == "true":
            tiny_counts[label] += 1
    return label_counts, split_counts, rows_by_label, dict(tiny_counts)


def choose_priority(semantic_id: str, current_count: int, assignment_ids: list[str]) -> str:
    if assignment_ids or semantic_id in COMMON_MUST_CLASSES or semantic_id in DEFAULT_DEMO_CLASSES:
        return "must"
    if semantic_id in COMMON_SHOULD_CLASSES or current_count > 0:
        return "should"
    return "optional"


def required_for(semantic_id: str, current_count: int, assignment_ids: list[str]) -> list[str]:
    groups: list[str] = []
    if assignment_ids:
        groups.append("assignment")
    if semantic_id in COMMON_MUST_CLASSES:
        groups.append("common_malaysia")
    if semantic_id in DEFAULT_DEMO_CLASSES:
        groups.append("demo_default")
    if semantic_id in COMMON_SHOULD_CLASSES:
        groups.append("secondary_malaysia")
    if semantic_id in OCR_VALUE_CLASSES:
        groups.append("ocr_value")
    if semantic_id in MULTILINGUAL_TEXT_CLASSES:
        groups.append("multilingual_text")
    if current_count > 0:
        groups.append("current_p5_dataset")
    if not groups:
        groups.append("future_catalogue")
    return groups


def target_counts(priority: str, groups: list[str]) -> tuple[int, int, int]:
    if "assignment" in groups or "demo_default" in groups:
        return 50, 150, 300
    if "common_malaysia" in groups:
        return 50, 100, 200
    if "ocr_value" in groups and priority != "optional":
        return 50, 100, 300
    if priority == "should":
        return 30, 80, 150
    return 10, 30, 50


def qc_risk_reason(semantic_id: str, current_count: int, tiny_count: int) -> str:
    reasons: list[str] = []
    if semantic_id in KNOWN_QC_RISK_CLASSES:
        reasons.append("known_noise_history")
    if current_count and tiny_count / current_count >= 0.25:
        reasons.append("high_tiny_crop_rate")
    return ";".join(reasons) if reasons else "none"


def split_status(train: int, validation: int, test: int, current_count: int) -> str:
    if current_count == 0:
        return "no_samples"
    missing = []
    if train == 0:
        missing.append("train")
    if validation == 0:
        missing.append("validation")
    if test == 0:
        missing.append("test")
    return "ok" if not missing else "missing_" + "_".join(missing)


def data_status(
    priority: str,
    current_count: int,
    train: int,
    validation: int,
    test: int,
    minimum: int,
    good: int,
    strong: int,
    risk_reason: str,
) -> str:
    if current_count == 0 and priority == "optional":
        return "optional_no_data"
    if current_count == 0:
        return "missing_data"
    if current_count < minimum:
        return "below_minimum"
    if train == 0 or validation == 0:
        return "split_gap"
    if risk_reason != "none":
        return "ready_count_qc_risk"
    if current_count >= strong:
        return "ready_strong"
    if current_count >= good:
        return "ready_good"
    return "ready_minimum"


def source_plan(
    semantic_id: str,
    priority: str,
    groups: list[str],
    current_count: int,
    minimum: int,
    good: int,
    status: str,
) -> str:
    if priority == "optional" and status == "optional_no_data":
        return "Do not collect now; exclude from final claims unless later promoted."
    if status == "ready_count_qc_risk":
        return "Run targeted QC/final split audit before training claim; collect only if QC removes too many samples."
    if status == "split_gap":
        return "Rebuild leakage-safe train/validation/test split before training."
    if status in {"ready_minimum", "ready_good", "ready_strong"}:
        return "Can be included in next classifier training after dataset freeze."

    needed = max(minimum - current_count, 0)
    base = f"Collect and annotate at least {needed} more clean crops; aim for {good} total."
    if "assignment" in groups:
        base += " Keep coursework images as external test unless explicitly moved into training."
    if "demo_default" in groups:
        base += " Capture phone/laptop-camera examples from real or printed demo signs."
    if "ocr_value" in groups:
        base += " Store OCR transcript/value/unit for each sample."
    if semantic_id == "school_zone":
        base += " Prioritize Malaysian school-zone signs because no current P5 samples exist."
    return base


def limitation(priority: str, status: str) -> str:
    if priority == "optional" and status == "optional_no_data":
        return "No final limitation if excluded from claims."
    if status in {"missing_data", "below_minimum"}:
        return "Likely unknown or wrong in the live app until data is collected and the classifier is retrained."
    if status == "ready_count_qc_risk":
        return "May train a noisy decision boundary unless QC confirms labels."
    if status == "split_gap":
        return "Metrics may be unreliable until split coverage is fixed."
    return "No major data-count limitation for the next training pass."


def build_rows() -> list[dict[str, object]]:
    catalogue = load_catalogue()
    sign_ids_by_semantic, values_by_semantic, assignment_image_counts = load_coursework_summary()
    p5_counts, split_counts, _rows_by_label, tiny_counts = load_p5_counts()

    rows: list[dict[str, object]] = []
    for entry in catalogue:
        semantic_id = entry["semantic_sign_id"]
        current_count = p5_counts.get(semantic_id, 0)
        assignment_ids = sorted(sign_ids_by_semantic.get(semantic_id, []))
        groups = required_for(semantic_id, current_count, assignment_ids)
        priority = choose_priority(semantic_id, current_count, assignment_ids)
        minimum, good, strong = target_counts(priority, groups)
        splits = split_counts.get(semantic_id, Counter())
        train = splits.get("train", 0)
        validation = splits.get("validation", 0)
        test = splits.get("test", 0)
        tiny_count = tiny_counts.get(semantic_id, 0)
        risk = qc_risk_reason(semantic_id, current_count, tiny_count)
        status = data_status(priority, current_count, train, validation, test, minimum, good, strong, risk)
        groups_text = ";".join(groups)

        rows.append(
            {
                "semantic_sign_id": semantic_id,
                "name_en": entry.get("names", {}).get("en", ""),
                "category": entry.get("category", ""),
                "severity": entry.get("severity", ""),
                "parameter_type": entry.get("parameter_type", ""),
                "priority": priority,
                "required_for": groups_text,
                "assignment_sign_ids": ";".join(assignment_ids),
                "assignment_image_count": assignment_image_counts.get(semantic_id, 0),
                "assignment_parameter_values": ";".join(sorted(set(values_by_semantic.get(semantic_id, [])))),
                "current_p5_crops": current_count,
                "train_count": train,
                "validation_count": validation,
                "test_count": test,
                "tiny_crop_count": tiny_count,
                "minimum_clean_crops": minimum,
                "good_target_crops": good,
                "strong_target_crops": strong,
                "gap_to_minimum": max(minimum - current_count, 0),
                "gap_to_good": max(good - current_count, 0),
                "split_status": split_status(train, validation, test, current_count),
                "qc_risk": risk,
                "data_status": status,
                "collection_rank": collection_rank(priority, status),
                "recommended_next_step": source_plan(
                    semantic_id, priority, groups, current_count, minimum, good, status
                ),
                "limitation_if_not_fixed": limitation(priority, status),
                "include_in_next_classifier": "yes"
                if status in {"ready_minimum", "ready_good", "ready_strong"}
                else "not_yet",
            }
        )

    rows.sort(
        key=lambda row: (
            PRIORITY_ORDER[str(row["priority"])],
            collection_rank_sort_key(str(row["collection_rank"])),
            STATUS_ORDER.get(str(row["data_status"]), 99),
            -int(row["gap_to_minimum"]),
            str(row["semantic_sign_id"]),
        )
    )
    return rows


def collection_rank(priority: str, status: str) -> str:
    if priority == "must" and status in {"missing_data", "below_minimum"}:
        return "P0_collect_now"
    if priority == "must" and status in {"ready_count_qc_risk", "split_gap"}:
        return "P1_fix_before_training"
    if priority == "should" and status in {"missing_data", "below_minimum"}:
        return "P2_collect_after_P0"
    if priority == "should" and status in {"ready_count_qc_risk", "split_gap"}:
        return "P3_fix_if_claimed"
    if priority == "optional":
        return "P4_optional"
    return "P5_ready_queue"


def collection_rank_sort_key(rank: str) -> int:
    return {
        "P0_collect_now": 0,
        "P1_fix_before_training": 1,
        "P2_collect_after_P0": 2,
        "P3_fix_if_claimed": 3,
        "P4_optional": 4,
        "P5_ready_queue": 5,
    }.get(rank, 99)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_priority = Counter(str(row["priority"]) for row in rows)
    by_status = Counter(str(row["data_status"]) for row in rows)
    must_rows = [row for row in rows if row["priority"] == "must"]
    must_action_rows = [
        row
        for row in must_rows
        if row["data_status"] in {"missing_data", "below_minimum", "ready_count_qc_risk", "split_gap"}
    ]
    p0_rows = [row for row in rows if row["collection_rank"] == "P0_collect_now"]
    ready_rows = [row for row in rows if row["include_in_next_classifier"] == "yes"]
    current_dataset_classes = [row for row in rows if int(row["current_p5_crops"]) > 0]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "catalogue": str(CATALOGUE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "coursework_mapping": str(COURSEWORK_MAPPING_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "coursework_manifest": str(COURSEWORK_MANIFEST_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "p5_manifest": str(P5_MANIFEST_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "assumptions": [
            "The target universe is the full Malaysia catalogue; assignment and common/demo signs are promoted to must.",
            "No separate final demo prop list exists yet, so demo_default classes are inferred from common ADAS presentation signs.",
            "P5 crop counts come from the current manifest after owner corrections, but known noisy classes still carry a QC risk flag.",
            "Coursework images are counted for coverage and should remain external test data unless explicitly approved for training.",
        ],
        "summary": {
            "total_target_classes": len(rows),
            "priority_counts": dict(sorted(by_priority.items())),
            "status_counts": dict(sorted(by_status.items())),
            "must_classes": len(must_rows),
            "must_classes_requiring_action": len(must_action_rows),
            "p0_collect_now_classes": len(p0_rows),
            "classes_ready_for_next_classifier_by_count": len(ready_rows),
            "classes_with_current_p5_samples": len(current_dataset_classes),
        },
        "p0_collect_now": [
            {
                "semantic_sign_id": row["semantic_sign_id"],
                "current_p5_crops": row["current_p5_crops"],
                "gap_to_minimum": row["gap_to_minimum"],
                "required_for": row["required_for"],
            }
            for row in p0_rows
        ],
        "must_fix_before_training": [
            {
                "semantic_sign_id": row["semantic_sign_id"],
                "data_status": row["data_status"],
                "qc_risk": row["qc_risk"],
                "current_p5_crops": row["current_p5_crops"],
                "recommended_next_step": row["recommended_next_step"],
            }
            for row in must_action_rows
        ],
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def render_contact_sheets(rows: list[dict[str, object]]) -> dict[str, str]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return {}

    _, _, rows_by_label, _ = load_p5_counts()
    WEAK_SHEET_ROOT.mkdir(parents=True, exist_ok=True)

    weak_classes = [
        str(row["semantic_sign_id"])
        for row in rows
        if row["priority"] in {"must", "should"}
        and int(row["current_p5_crops"]) > 0
        and row["collection_rank"] in {"P0_collect_now", "P1_fix_before_training", "P2_collect_after_P0", "P3_fix_if_claimed"}
    ]
    outputs: dict[str, str] = {}
    font = ImageFont.load_default()
    columns = 6
    tile_w = 190
    tile_h = 170
    label_h = 34
    max_tiles = 48

    for class_id in weak_classes:
        class_rows = rows_by_label.get(class_id, [])[:max_tiles]
        if not class_rows:
            continue
        sheet_rows = (len(class_rows) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * tile_w, sheet_rows * tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(class_rows):
            x = (index % columns) * tile_w
            y = (index // columns) * tile_h
            image_path = PROJECT_ROOT / row["file"]
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((tile_w - 12, tile_h - label_h - 12))
                    px = x + (tile_w - image.width) // 2
                    py = y + 6
                    sheet.paste(image, (px, py))
            except OSError:
                draw.rectangle([x + 8, y + 8, x + tile_w - 8, y + tile_h - label_h - 8], outline="red")
            draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(210, 210, 210))
            label = f"{row.get('split', '')} {row.get('instance_id', '')}"
            draw.text((x + 6, y + tile_h - label_h + 6), label[:33], fill=(20, 20, 20), font=font)

        output = WEAK_SHEET_ROOT / f"{class_id}.jpg"
        sheet.save(output, quality=90)
        outputs[class_id] = str(output.relative_to(PROJECT_ROOT)).replace("\\", "/")

    return outputs


def main() -> None:
    rows = build_rows()
    contact_sheets = render_contact_sheets(rows)
    for row in rows:
        row["weak_class_contact_sheet"] = contact_sheets.get(str(row["semantic_sign_id"]), "")

    fieldnames = [
        "semantic_sign_id",
        "name_en",
        "category",
        "severity",
        "parameter_type",
        "priority",
        "required_for",
        "assignment_sign_ids",
        "assignment_image_count",
        "assignment_parameter_values",
        "current_p5_crops",
        "train_count",
        "validation_count",
        "test_count",
        "tiny_crop_count",
        "minimum_clean_crops",
        "good_target_crops",
        "strong_target_crops",
        "gap_to_minimum",
        "gap_to_good",
        "split_status",
        "qc_risk",
        "data_status",
        "collection_rank",
        "recommended_next_step",
        "limitation_if_not_fixed",
        "include_in_next_classifier",
        "weak_class_contact_sheet",
    ]
    write_csv(TARGET_CLASSES_PATH, rows, fieldnames)
    write_csv(GAP_CSV_PATH, rows, fieldnames)

    summary = summarize(rows)
    coverage_payload = {**summary, "target_classes": rows}
    gap_payload = {**summary, "gap_report": rows, "weak_class_contact_sheets": contact_sheets}
    write_json(COVERAGE_JSON_PATH, coverage_payload)
    write_json(GAP_JSON_PATH, gap_payload)

    print(f"Wrote {TARGET_CLASSES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {COVERAGE_JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {GAP_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {GAP_JSON_PATH.relative_to(PROJECT_ROOT)}")
    if contact_sheets:
        print(f"Wrote {len(contact_sheets)} weak-class contact sheets to {WEAK_SHEET_ROOT.relative_to(PROJECT_ROOT)}")
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
