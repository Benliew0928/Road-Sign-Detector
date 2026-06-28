from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_GAP_PATH = PROJECT_ROOT / "outputs/audit/post_roboflow_data_gap_report.csv"
SPRINT_MANIFESTS = [
    PROJECT_ROOT / "data/manifests/stage_c_sprint_01_commons_candidates.csv",
    PROJECT_ROOT / "data/manifests/stage_c_sprint_02_synthetic_candidates.csv",
    PROJECT_ROOT / "data/manifests/stage_c_sprint_03_mandatory_direction_symbols.csv",
    PROJECT_ROOT / "data/manifests/stage_c_sprint_04_compound_mandatory_symbols.csv",
    PROJECT_ROOT / "data/manifests/stage_c_sprint_05_prohibitory_direction_symbols.csv",
    PROJECT_ROOT / "data/manifests/stage_c_sprint_06_warning_symbols.csv",
    PROJECT_ROOT / "data/manifests/stage_c_sprint_07_regulatory_text_symbols.csv",
]
MINED_MANIFESTS = [
    PROJECT_ROOT / "data/manifests/stage_c_mined_roboflow_gap_candidates.csv",
]
REFERENCE_MANIFESTS = [
    PROJECT_ROOT / "data/manifests/stage_c_online_reference_sources_01.csv",
    PROJECT_ROOT / "data/manifests/stage_c_china_reference_sources_01.csv",
]
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs/audit/post_stage_c_realistic_gap_report.csv"
OUTPUT_JSON_PATH = PROJECT_ROOT / "outputs/audit/post_stage_c_realistic_gap_report.json"

FIELDNAMES = [
    "semantic_sign_id",
    "priority",
    "required_for",
    "candidate_total_before_sprint",
    "real_sprint_candidates",
    "mined_public_candidates",
    "generated_reference_only_candidates",
    "online_reference_only_candidates",
    "realistic_candidate_total",
    "minimum_clean_crops",
    "good_target_crops",
    "gap_to_minimum_realistic",
    "gap_to_good_realistic",
    "realistic_candidate_status",
    "note",
]

PRIORITY_ORDER = {"must": 0, "should": 1, "optional": 2}
STATUS_ORDER = {
    "still_below_minimum": 0,
    "meets_minimum_pending_qc": 1,
    "meets_good_pending_qc": 2,
}

NOTE = (
    "Real/public sprint and mined candidates still require Stage D visual "
    "QC/annotation and Stage E split freeze. Locally generated symbols and "
    "online reference diagrams are reference_only and do not count toward final "
    "realistic photo coverage."
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def sprint_counts() -> tuple[Counter[str], Counter[str]]:
    real_counts: Counter[str] = Counter()
    generated_reference_counts: Counter[str] = Counter()
    for manifest in SPRINT_MANIFESTS:
        for row in read_csv(manifest):
            if row.get("download_status") == "downloaded_candidate":
                real_counts[row["semantic_sign_id"]] += 1
            elif row.get("generation_status") == "generated_candidate":
                generated_reference_counts[row["semantic_sign_id"]] += 1
    return real_counts, generated_reference_counts


def mined_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    for manifest in MINED_MANIFESTS:
        for row in read_csv(manifest):
            if row.get("mining_status") == "mined_candidate" and not row.get("review_status", "").startswith("rejected"):
                counts[row["semantic_sign_id"]] += 1
    return counts


def reference_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    for manifest in REFERENCE_MANIFESTS:
        for row in read_csv(manifest):
            if row.get("download_status") == "downloaded_reference_candidate":
                counts[row["semantic_sign_id"]] += 1
    return counts


def status(total: int, minimum: int, good: int) -> str:
    if total >= good:
        return "meets_good_pending_qc"
    if total >= minimum:
        return "meets_minimum_pending_qc"
    return "still_below_minimum"


def main() -> None:
    real_sprint_by_class, generated_reference_by_class = sprint_counts()
    mined_by_class = mined_counts()
    reference_by_class = reference_counts()
    rows: list[dict[str, object]] = []
    for row in read_csv(BASE_GAP_PATH):
        semantic_id = row["semantic_sign_id"]
        before = int(row["candidate_total_crops"])
        real_sprint = int(real_sprint_by_class[semantic_id])
        mined = int(mined_by_class[semantic_id])
        generated_reference = int(generated_reference_by_class[semantic_id])
        online_reference = int(reference_by_class[semantic_id])
        total = before + real_sprint + mined
        minimum = int(row["minimum_clean_crops"])
        good = int(row["good_target_crops"])
        rows.append(
            {
                "semantic_sign_id": semantic_id,
                "priority": row["priority"],
                "required_for": row["required_for"],
                "candidate_total_before_sprint": before,
                "real_sprint_candidates": real_sprint,
                "mined_public_candidates": mined,
                "generated_reference_only_candidates": generated_reference,
                "online_reference_only_candidates": online_reference,
                "realistic_candidate_total": total,
                "minimum_clean_crops": minimum,
                "good_target_crops": good,
                "gap_to_minimum_realistic": max(0, minimum - total),
                "gap_to_good_realistic": max(0, good - total),
                "realistic_candidate_status": status(total, minimum, good),
                "note": NOTE,
            }
        )
    rows.sort(
        key=lambda item: (
            PRIORITY_ORDER.get(str(item["priority"]), 9),
            STATUS_ORDER.get(str(item["realistic_candidate_status"]), 9),
            -int(item["gap_to_minimum_realistic"]),
            str(item["semantic_sign_id"]),
        )
    )
    write_csv(OUTPUT_CSV_PATH, rows)

    must_rows = [row for row in rows if row["priority"] == "must"]
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "base_gap_report": BASE_GAP_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sprint_manifests": [
                manifest.relative_to(PROJECT_ROOT).as_posix() for manifest in SPRINT_MANIFESTS
            ],
            "mined_manifests": [
                manifest.relative_to(PROJECT_ROOT).as_posix() for manifest in MINED_MANIFESTS
            ],
            "reference_manifests": [
                manifest.relative_to(PROJECT_ROOT).as_posix() for manifest in REFERENCE_MANIFESTS
            ],
        },
        "summary": {
            "target_classes": len(rows),
            "real_sprint_candidates": sum(real_sprint_by_class.values()),
            "mined_public_candidates": sum(mined_by_class.values()),
            "generated_reference_only_candidates": sum(generated_reference_by_class.values()),
            "online_reference_only_candidates": sum(reference_by_class.values()),
            "classes_with_real_sprint_gain": len(real_sprint_by_class),
            "classes_with_mined_public_gain": len(mined_by_class),
            "classes_with_generated_reference_only": len(generated_reference_by_class),
            "classes_with_online_reference_only": len(reference_by_class),
            "must_classes_meeting_minimum_realistic": sum(
                1 for row in must_rows if int(row["realistic_candidate_total"]) >= int(row["minimum_clean_crops"])
            ),
            "must_classes_still_below_minimum_realistic": sum(
                1 for row in must_rows if int(row["realistic_candidate_total"]) < int(row["minimum_clean_crops"])
            ),
        },
        "real_sprint_counts_by_class": dict(sorted(real_sprint_by_class.items())),
        "mined_public_counts_by_class": dict(sorted(mined_by_class.items())),
        "generated_reference_only_counts_by_class": dict(sorted(generated_reference_by_class.items())),
        "online_reference_only_counts_by_class": dict(sorted(reference_by_class.items())),
        "remaining_must_gaps": [
            row
            for row in must_rows
            if int(row["gap_to_minimum_realistic"]) > 0
        ],
        "output_csv": OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "status": "realistic_candidates_counted_generated_references_excluded",
    }
    OUTPUT_JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {OUTPUT_JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
