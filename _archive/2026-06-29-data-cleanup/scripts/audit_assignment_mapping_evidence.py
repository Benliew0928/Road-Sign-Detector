from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSEWORK_MANIFEST = PROJECT_ROOT / "data/manifests/coursework_manifest.csv"
CHINA_REFERENCE_MANIFEST = PROJECT_ROOT / "data/manifests/stage_c_china_reference_sources_01.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/manifests/assignment_mapping_audit.csv"
LATEST_OUTPUT_CSV = PROJECT_ROOT / "data/manifests/assignment_mapping_audit_latest.csv"
OUTPUT_JSON = PROJECT_ROOT / "outputs/audit/assignment_mapping_audit.json"
UNIQUE_SHEET = PROJECT_ROOT / "outputs/review/assignment_mapping_audit_unique_signs.jpg"
LOW_CONFIDENCE_SHEET = PROJECT_ROOT / "outputs/review/assignment_mapping_audit_low_confidence.jpg"

OWNER_CONFIRMED = {
    "sign_008": "Owner confirmed no_straight_or_left on 2026-06-26.",
    "sign_010": "Owner corrected no_straight_ahead on 2026-06-28.",
    "sign_014": "Owner corrected no_overtaking on 2026-06-28.",
    "sign_032": "Owner corrected roadway_diverges / split traffic on 2026-06-28.",
    "sign_051": "Owner corrected tractors_ahead on 2026-06-28.",
    "sign_057": "Owner confirmed stop_for_checking on 2026-06-26.",
}

EXTERNAL_REFERENCE_OVERRIDES = {
    "sign_045": {
        "status": "confirmed_online_reference",
        "source": "https://commons.wikimedia.org/wiki/File:CN_road_sign_%E8%AD%A6_23.svg",
        "note": "Residential/village area ahead verified from CN road sign warning 23.",
    }
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_with_fallback(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    try:
        write_csv(path, rows, fieldnames)
        return path
    except PermissionError:
        write_csv(LATEST_OUTPUT_CSV, rows, fieldnames)
        return LATEST_OUTPUT_CSV


def china_references() -> dict[str, dict[str, str]]:
    refs: dict[str, dict[str, str]] = {}
    if not CHINA_REFERENCE_MANIFEST.exists():
        return refs
    for row in read_csv(CHINA_REFERENCE_MANIFEST):
        if row.get("download_status") == "downloaded_reference_candidate":
            refs[row["semantic_sign_id"]] = row
    return refs


def audit_status(sign_id: str, semantic_id: str, confidence: float, refs: dict[str, dict[str, str]]) -> tuple[str, str, str]:
    if sign_id in EXTERNAL_REFERENCE_OVERRIDES:
        item = EXTERNAL_REFERENCE_OVERRIDES[sign_id]
        return item["status"], item["source"], item["note"]
    if sign_id in OWNER_CONFIRMED:
        return "confirmed_owner_review", "", OWNER_CONFIRMED[sign_id]
    if semantic_id in refs:
        ref = refs[semantic_id]
        if ref.get("mapping_confidence") == "high":
            return "confirmed_online_reference", ref.get("commons_page_url", ""), ref.get("evidence_label", "")
        return "accepted_with_reference_caution", ref.get("commons_page_url", ""), ref.get("notes", "")
    if confidence >= 0.95:
        return "confirmed_visual_high_confidence", "", "Clear standard sign shape/text in assignment image."
    if confidence >= 0.90:
        return "accepted_visual_medium_confidence", "", "Visual mapping is plausible; image quality or sign variant is less clear."
    return "needs_second_review", "", "Low confidence and no explicit evidence override."


def render_sheet(rows: list[dict[str, Any]], output_path: Path, *, columns: int = 6) -> str:
    tile_width = 250
    tile_height = 260
    label_height = 82
    sheet = Image.new(
        "RGB",
        (columns * tile_width, max(1, math.ceil(len(rows) / columns)) * tile_height),
        color=(18, 23, 22),
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        image_path = PROJECT_ROOT / "data/official/assignment_images" / row["representative_relative_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            thumbnail = ImageOps.contain(image, (tile_width - 14, tile_height - label_height - 14))
        sheet.paste(
            thumbnail,
            (
                x + (tile_width - thumbnail.width) // 2,
                y + 7 + (tile_height - label_height - 14 - thumbnail.height) // 2,
            ),
        )
        draw.rectangle((x, y + tile_height - label_height, x + tile_width, y + tile_height), fill=(32, 43, 39))
        draw.text((x + 6, y + tile_height - label_height + 5), f"{row['verified_coursework_id']}  {row['confidence']}", fill=(232, 239, 236))
        draw.text((x + 6, y + tile_height - label_height + 24), row["semantic_sign_id"][:33], fill=(232, 239, 236))
        draw.text((x + 6, y + tile_height - label_height + 44), row["audit_status"][:34], fill=(180, 198, 190))
        draw.text((x + 6, y + tile_height - label_height + 62), row["representative_relative_path"][:34], fill=(160, 178, 170))
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline=(65, 77, 73), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    manifest_rows = read_csv(COURSEWORK_MANIFEST)
    refs = china_references()
    by_sign: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        by_sign[row["verified_coursework_id"]].append(row)

    audit_rows: list[dict[str, Any]] = []
    for sign_id, rows in sorted(by_sign.items(), key=lambda item: int(item[0].split("_")[1])):
        semantic_ids = sorted({row["semantic_sign_id"] for row in rows})
        confidence_values = [float(row["confidence"]) for row in rows if row.get("confidence")]
        confidence = min(confidence_values) if confidence_values else 0.0
        representative = rows[0]
        if len(semantic_ids) != 1:
            status = "needs_second_review"
            source = ""
            note = "One coursework ID maps to multiple semantic IDs."
            semantic_id = "|".join(semantic_ids)
        else:
            semantic_id = semantic_ids[0]
            status, source, note = audit_status(sign_id, semantic_id, confidence, refs)
        audit_rows.append(
            {
                "verified_coursework_id": sign_id,
                "semantic_sign_id": semantic_id,
                "image_count": len(rows),
                "representative_relative_path": representative["relative_path"],
                "confidence": f"{confidence:.2f}",
                "audit_status": status,
                "evidence_source": source,
                "audit_note": note,
                "all_relative_paths": ";".join(row["relative_path"] for row in rows),
            }
        )

    fieldnames = [
        "verified_coursework_id",
        "semantic_sign_id",
        "image_count",
        "representative_relative_path",
        "confidence",
        "audit_status",
        "evidence_source",
        "audit_note",
        "all_relative_paths",
    ]
    written_csv = write_csv_with_fallback(OUTPUT_CSV, audit_rows, fieldnames)
    unique_sheet = render_sheet(audit_rows, UNIQUE_SHEET)
    low_rows = [
        row
        for row in audit_rows
        if row["audit_status"]
        in {
            "needs_second_review",
            "confirmed_owner_review",
            "accepted_visual_medium_confidence",
            "accepted_with_reference_caution",
        }
    ]
    low_sheet = render_sheet(low_rows, LOW_CONFIDENCE_SHEET, columns=4) if low_rows else ""
    status_counts: dict[str, int] = defaultdict(int)
    for row in audit_rows:
        status_counts[row["audit_status"]] += 1
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": COURSEWORK_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        "assignment_image_count": len(manifest_rows),
        "unique_assignment_sign_ids": len(audit_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "needs_second_review": [
            row for row in audit_rows if row["audit_status"] == "needs_second_review"
        ],
        "important_note": (
            "This audit records current evidence quality. It is stronger than a guess, "
            "but a lecturer-provided answer key would still be the highest authority."
        ),
        "outputs": {
            "audit_csv": written_csv.relative_to(PROJECT_ROOT).as_posix(),
            "unique_contact_sheet": unique_sheet,
            "attention_contact_sheet": low_sheet,
        },
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {written_csv.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {OUTPUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {unique_sheet}")
    if low_sheet:
        print(f"Wrote {low_sheet}")
    print(json.dumps(report["status_counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
