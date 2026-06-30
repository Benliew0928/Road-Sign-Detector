from __future__ import annotations

import csv
import json
from pathlib import Path

from roadsign_assist.paths import project_path


CATALOGUE_PATH = project_path("configs/catalogue/malaysia_signs.v1.json")
REVIEW_CSV = project_path("data/manifests/p2_ontology_review.csv")
VALID_DECISIONS = {"", "approve", "reject", "needs_change"}


def main() -> None:
    if not REVIEW_CSV.exists():
        raise SystemExit(f"Review workbook does not exist: {REVIEW_CSV}")

    catalogue = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    entries_by_id = {entry["semantic_sign_id"]: entry for entry in catalogue["entries"]}
    unknown_entries: list[str] = []
    invalid_decisions: list[str] = []
    approved = 0
    rejected_or_pending = 0

    with REVIEW_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            semantic_id = row["semantic_sign_id"]
            decision = row.get("reviewer_2_decision", "").strip().casefold()
            if decision not in VALID_DECISIONS:
                invalid_decisions.append(f"{semantic_id}: {decision}")
                continue
            entry = entries_by_id.get(semantic_id)
            if entry is None:
                unknown_entries.append(semantic_id)
                continue
            if decision == "approve":
                entry["review_status"] = "approved"
                approved += 1
            elif decision in {"reject", "needs_change"}:
                entry["review_status"] = "draft"
                rejected_or_pending += 1

    if unknown_entries or invalid_decisions:
        if unknown_entries:
            print("Unknown semantic IDs in review workbook:")
            for item in unknown_entries:
                print(f"- {item}")
        if invalid_decisions:
            print("Invalid reviewer decisions; use approve, reject, needs_change, or blank:")
            for item in invalid_decisions:
                print(f"- {item}")
        raise SystemExit(1)

    CATALOGUE_PATH.write_text(
        json.dumps(catalogue, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Applied {approved} approvals; {rejected_or_pending} entries remain draft.")


if __name__ == "__main__":
    main()
