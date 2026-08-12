from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
AUDIT_PATH = PROJECT_ROOT / "outputs/audit/classifier_no_controlled_variants_20260812.json"


def read_tracker() -> list[dict[str, str]]:
    with TRACKER_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rows_with_gaps(
    rows: list[dict[str, str]],
    priorities: set[str],
) -> list[dict[str, str]]:
    return sorted(
        (
            row
            for row in rows
            if row["priority"] in priorities and int(row["gap_to_minimum"]) > 0
        ),
        key=lambda row: (-int(row["gap_to_minimum"]), row["semantic_sign_id"]),
    )


def print_gap_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No gaps in the selected priorities.")
        return
    print("Priority  Class                         Have  Need  Gap")
    print("--------  ----------------------------  ----  ----  ---")
    for row in rows:
        print(
            f"{row['priority']:<8}  {row['semantic_sign_id']:<28}  "
            f"{row['realistic_candidate_total']:>4}  "
            f"{row['minimum_clean_crops']:>4}  "
            f"{row['gap_to_minimum']:>3}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show active RoadSign Assist collection progress and coverage gaps."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include must, should, and optional gaps; default shows final-model blockers only.",
    )
    parser.add_argument(
        "--priority",
        choices=("must", "should", "optional"),
        help="Show gaps for one priority tier. Overrides the default.",
    )
    args = parser.parse_args()

    tracker = read_tracker()
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    accepted_rows = sum(
        1
        for row in csv.DictReader(
            (PROJECT_ROOT / "data/annotations/classifier_review_decisions.csv").open(
                encoding="utf-8-sig", newline=""
            )
        )
        if row.get("review_decision") == "accept"
    )
    priority_counts = Counter(row["priority"] for row in tracker)

    print("RoadSign Assist — active data collection progress")
    print(
        f"Clean classifier release: {audit['dataset_id']} | "
        f"{audit['samples']:,} samples | {audit['labels']} labels | "
        f"{audit['controlled_variant_samples']} controlled variants"
    )
    print(
        f"Release status: {audit['release_status']} | "
        f"reviewed accepted rows: {accepted_rows:,} | "
        f"target classes: {sum(priority_counts.values())}"
    )
    print()

    priorities = (
        {args.priority}
        if args.priority
        else {"must", "should", "optional"}
        if args.all
        else {"must"}
    )
    scope = ", ".join(sorted(priorities))
    print(f"Gaps for: {scope}")
    gaps = rows_with_gaps(tracker, priorities)
    print_gap_table(gaps)
    print()
    print(
        "This command reads the active tracker only. New images must first be "
        "reviewed, accepted, and frozen into a new versioned release before "
        "these counts should be updated."
    )


if __name__ == "__main__":
    main()
