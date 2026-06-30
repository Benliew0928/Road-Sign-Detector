from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_GAP_CSV_PATH = PROJECT_ROOT / "outputs/audit/data_gap_report.csv"
ROBOFLOW_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/roboflow_source_manifest.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/audit/post_roboflow_data_gap_report.csv"
OUTPUT_JSON_PATH = PROJECT_ROOT / "outputs/audit/post_roboflow_data_gap_report.json"

NOTE = (
    "Roboflow imports are candidate public data; final training still needs "
    "dataset freeze and targeted QC."
)

FIELDNAMES = [
    "semantic_sign_id",
    "priority",
    "required_for",
    "p5_current_crops",
    "roboflow_accepted_candidate_crops",
    "candidate_total_crops",
    "minimum_clean_crops",
    "good_target_crops",
    "gap_to_minimum_after_roboflow",
    "gap_to_good_after_roboflow",
    "candidate_status",
    "roboflow_detection_crops",
    "roboflow_classification_images",
    "note",
]

PRIORITY_ORDER = {"must": 0, "should": 1, "optional": 2}
STATUS_ORDER = {
    "candidate_still_below_minimum": 0,
    "no_roboflow_gain": 1,
    "candidate_meets_minimum": 2,
    "candidate_meets_good": 3,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_roboflow_counts() -> tuple[Counter[str], Counter[str], Counter[str], int]:
    accepted = Counter()
    detection = Counter()
    classification = Counter()
    accepted_total = 0
    for row in read_csv(ROBOFLOW_MANIFEST_PATH):
        if row.get("quality_status") != "accepted_public_mixed":
            continue
        accepted_total += 1
        semantic_id = row.get("semantic_sign_id", "")
        if not semantic_id:
            continue
        accepted[semantic_id] += 1
        if row.get("task_type") == "yolov8_detection_crops":
            detection[semantic_id] += 1
        elif row.get("task_type") == "classification_folder":
            classification[semantic_id] += 1
    return accepted, detection, classification, accepted_total


def candidate_status(candidate_total: int, roboflow_count: int, minimum: int, good: int) -> str:
    if candidate_total >= good:
        return "candidate_meets_good"
    if candidate_total >= minimum:
        return "candidate_meets_minimum"
    if roboflow_count > 0:
        return "candidate_still_below_minimum"
    return "no_roboflow_gain"


def build_rows() -> list[dict[str, object]]:
    roboflow_counts, detection_counts, classification_counts, _accepted_total = load_roboflow_counts()
    rows: list[dict[str, object]] = []
    for row in read_csv(DATA_GAP_CSV_PATH):
        semantic_id = row["semantic_sign_id"]
        p5_count = int(row["current_p5_crops"])
        roboflow_count = int(roboflow_counts[semantic_id])
        candidate_total = p5_count + roboflow_count
        minimum = int(row["minimum_clean_crops"])
        good = int(row["good_target_crops"])
        rows.append(
            {
                "semantic_sign_id": semantic_id,
                "priority": row["priority"],
                "required_for": row["required_for"],
                "p5_current_crops": p5_count,
                "roboflow_accepted_candidate_crops": roboflow_count,
                "candidate_total_crops": candidate_total,
                "minimum_clean_crops": minimum,
                "good_target_crops": good,
                "gap_to_minimum_after_roboflow": max(0, minimum - candidate_total),
                "gap_to_good_after_roboflow": max(0, good - candidate_total),
                "candidate_status": candidate_status(candidate_total, roboflow_count, minimum, good),
                "roboflow_detection_crops": int(detection_counts[semantic_id]),
                "roboflow_classification_images": int(classification_counts[semantic_id]),
                "note": NOTE,
            }
        )

    rows.sort(
        key=lambda item: (
            PRIORITY_ORDER.get(str(item["priority"]), 9),
            STATUS_ORDER.get(str(item["candidate_status"]), 9),
            -int(item["gap_to_minimum_after_roboflow"]),
            str(item["semantic_sign_id"]),
        )
    )
    return rows


def main() -> None:
    rows = build_rows()
    write_csv(OUTPUT_CSV_PATH, rows)

    must_rows = [row for row in rows if row["priority"] == "must"]
    _roboflow_counts, _detection_counts, _classification_counts, accepted_total = load_roboflow_counts()
    accepted_target_total = sum(int(row["roboflow_accepted_candidate_crops"]) for row in rows)
    summary = {
        "target_classes": len(rows),
        "accepted_roboflow_candidate_samples": accepted_total,
        "accepted_roboflow_candidate_samples_for_target_classes": accepted_target_total,
        "classes_with_roboflow_gain": sum(
            1 for row in rows if int(row["roboflow_accepted_candidate_crops"]) > 0
        ),
        "must_classes_meeting_minimum_after_roboflow": sum(
            1 for row in must_rows if int(row["candidate_total_crops"]) >= int(row["minimum_clean_crops"])
        ),
        "must_classes_still_below_minimum_after_roboflow": sum(
            1 for row in must_rows if int(row["candidate_total_crops"]) < int(row["minimum_clean_crops"])
        ),
    }
    top_remaining = [
        row
        for row in must_rows
        if int(row["candidate_total_crops"]) < int(row["minimum_clean_crops"])
    ][:20]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "data_gap_report": DATA_GAP_CSV_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "roboflow_manifest": ROBOFLOW_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
        "summary": summary,
        "top_remaining_must_gaps": top_remaining,
        "output_csv": OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }
    OUTPUT_JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {OUTPUT_JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
