from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw

from roadsign_assist.catalogue.models import Severity
from roadsign_assist.catalogue.repository import load_catalogue, load_standards_manifest
from roadsign_assist.paths import project_path


AUDIT_DIR = project_path("outputs/audit")
REVIEW_CSV = project_path("data/manifests/p2_ontology_review.csv")
REVIEW_XLSX = project_path("data/manifests/p2_ontology_review.xlsx")
REVIEW_FALLBACK_CSV = project_path("data/manifests/p2_ontology_review_latest.csv")
REVIEW_FALLBACK_XLSX = project_path("data/manifests/p2_ontology_review_latest.xlsx")
UNRESOLVED_CSV = project_path("data/manifests/p2_unresolved_coursework_review.csv")
UNRESOLVED_SHEET = AUDIT_DIR / "p2_unresolved_coursework_signs.png"
AUDIT_JSON = AUDIT_DIR / "p2_ontology_audit.json"
UNRESOLVED_SUGGESTIONS = {
    "sign_008": {
        "relative_path": "Red Signs/008_1_0008_1_j.png",
        "suggested_semantic_id": "no_straight_or_left",
        "alternative": "needs_new_compound_prohibition",
        "reason": "Visual appears to prohibit the displayed straight/left movement combination.",
    },
    "sign_014": {
        "relative_path": "Red Signs/014_1_0028.png",
        "suggested_semantic_id": "no_lane_changing",
        "alternative": "needs_new_lane_rule",
        "reason": "Owner confirmed no lane changing.",
    },
    "sign_051": {
        "relative_path": "Yellow Signs/051_0005_j.png",
        "suggested_semantic_id": "vehicle_accident_hazard",
        "alternative": "general_caution",
        "reason": "Chinese warning sign shows a vehicle-related hazard; exact official meaning needs human confirmation.",
    },
    "sign_057": {
        "relative_path": "Red Signs/057_0003_j.png",
        "suggested_semantic_id": "stop_for_checking",
        "alternative": "stop_request",
        "reason": "Owner confirmed stop for checking.",
    },
}


def _existing_review_decisions() -> dict[str, dict[str, str]]:
    if not REVIEW_CSV.exists():
        return {}
    with REVIEW_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["semantic_sign_id"]: {
                "reviewer_2_decision": row.get("reviewer_2_decision", ""),
                "reviewer_2_notes": row.get("reviewer_2_notes", ""),
            }
            for row in csv.DictReader(handle)
        }


def _write_xlsx(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    def cell_ref(column_index: int, row_index: int) -> str:
        name = ""
        column = column_index
        while column:
            column, remainder = divmod(column - 1, 26)
            name = chr(65 + remainder) + name
        return f"{name}{row_index}"

    def inline_cell(column_index: int, row_index: int, value: object) -> str:
        text = "" if value is None else str(value)
        return (
            f'<c r="{cell_ref(column_index, row_index)}" t="inlineStr">'
            f"<is><t>{escape(text)}</t></is></c>"
        )

    sheet_rows: list[str] = []
    sheet_rows.append(
        '<row r="1">'
        + "".join(inline_cell(index, 1, field) for index, field in enumerate(fieldnames, 1))
        + "</row>"
    )
    for row_index, row in enumerate(rows, 2):
        sheet_rows.append(
            f'<row r="{row_index}">'
            + "".join(
                inline_cell(column_index, row_index, row.get(field, ""))
                for column_index, field in enumerate(fieldnames, 1)
            )
            + "</row>"
        )

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheetViews><sheetView workbookViewId=\"0\"><pane ySplit=\"1\" topLeftCell=\"A2\" "
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        "<sheetData>"
        + "".join(sheet_rows)
        + "</sheetData></worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets><sheet name=\"P2 Ontology Review\" sheetId=\"1\" r:id=\"rId1\"/></sheets>"
        "</workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as workbook_zip:
        workbook_zip.writestr("[Content_Types].xml", content_types)
        workbook_zip.writestr("_rels/.rels", root_rels)
        workbook_zip.writestr("xl/workbook.xml", workbook)
        workbook_zip.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        workbook_zip.writestr("xl/worksheets/sheet1.xml", worksheet)


def _write_unresolved_coursework_review(unresolved: dict[str, str]) -> None:
    rows: list[dict[str, str]] = []
    for sign_id, note in sorted(unresolved.items()):
        suggestion = UNRESOLVED_SUGGESTIONS.get(sign_id, {})
        rows.append(
            {
                "coursework_id": sign_id,
                "relative_path": suggestion.get("relative_path", ""),
                "current_note": note,
                "suggested_semantic_id": suggestion.get("suggested_semantic_id", ""),
                "alternative": suggestion.get("alternative", ""),
                "reason": suggestion.get("reason", ""),
                "owner_decision": "",
                "owner_notes": "",
            }
        )
    UNRESOLVED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with UNRESOLVED_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "coursework_id",
                "relative_path",
                "current_note",
                "suggested_semantic_id",
                "alternative",
                "reason",
                "owner_decision",
                "owner_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_unresolved_sheet(unresolved: dict[str, str]) -> None:
    if not unresolved:
        sheet = Image.new("RGB", (720, 180), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((24, 40), "No unresolved coursework ontology mappings.", fill=(0, 90, 0))
        draw.text(
            (24, 78),
            "The four owner-reviewed signs have been mapped in coursework_draft_mapping.json.",
            fill=(60, 60, 60),
        )
        UNRESOLVED_SHEET.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(UNRESOLVED_SHEET)
        return
    root = project_path("data/official/assignment_images")
    items = [
        (sign_id, UNRESOLVED_SUGGESTIONS[sign_id])
        for sign_id in sorted(unresolved)
        if sign_id in UNRESOLVED_SUGGESTIONS
    ]
    cell_width = 360
    cell_height = 410
    columns = 2
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (sign_id, suggestion) in enumerate(items):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        path = root / suggestion["relative_path"]
        image = Image.open(path).convert("RGB")
        image.thumbnail((260, 260), Image.Resampling.LANCZOS)
        draw.rectangle([x, y, x + cell_width - 1, y + cell_height - 1], outline=(180, 180, 180))
        draw.text((x + 12, y + 12), sign_id, fill=(0, 0, 0))
        draw.text((x + 12, y + 36), suggestion["relative_path"], fill=(70, 70, 70))
        draw.text(
            (x + 12, y + 62),
            f"Suggested: {suggestion['suggested_semantic_id']}",
            fill=(0, 80, 140),
        )
        draw.text((x + 12, y + 84), f"Alt: {suggestion['alternative']}", fill=(100, 70, 0))
        px = x + (cell_width - image.width) // 2
        py = y + 120 + (250 - image.height) // 2
        sheet.paste(image, (px, py))
    UNRESOLVED_SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(UNRESOLVED_SHEET)


def main() -> None:
    catalogue = load_catalogue()
    standards = load_standards_manifest()
    known_references = {document.reference_id: document for document in standards.documents}

    missing_registered_reference: list[str] = []
    missing_local_archive: list[str] = []
    critical_not_approved: list[str] = []
    draft_entries: list[str] = []

    for document in standards.documents:
        if document.local_archive_status != "archived":
            missing_local_archive.append(document.reference_id)
            continue
        if not document.local_archive_path or not document.local_archive_sha256:
            missing_local_archive.append(document.reference_id)
            continue
        archive_path = project_path(document.local_archive_path)
        if not archive_path.exists():
            missing_local_archive.append(document.reference_id)

    for entry in catalogue.entries:
        if entry.standard_reference.reference_id not in known_references:
            missing_registered_reference.append(entry.semantic_sign_id)
        if entry.review_status != "approved":
            draft_entries.append(entry.semantic_sign_id)
        if entry.severity is Severity.CRITICAL and entry.review_status != "approved":
            critical_not_approved.append(entry.semantic_sign_id)

    coursework_mapping = json.loads(
        project_path("configs/catalogue/coursework_draft_mapping.json").read_text(
            encoding="utf-8"
        )
    )
    unresolved_coursework = {
        sign_id: value["notes"]
        for sign_id, value in coursework_mapping["mappings"].items()
        if not value["semantic_sign_id"]
    }
    _write_unresolved_coursework_review(unresolved_coursework)
    _write_unresolved_sheet(unresolved_coursework)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    previous_decisions = _existing_review_decisions()
    review_fieldnames = [
        "semantic_sign_id",
        "category",
        "name_en",
        "name_ms",
        "name_zh",
        "base_action",
        "severity",
        "parameter_type",
        "default_parameter",
        "standard_reference",
        "review_status",
        "reviewer_2_decision",
        "reviewer_2_notes",
    ]
    review_rows: list[dict[str, object]] = []
    written_review_csv = REVIEW_CSV
    written_review_xlsx = REVIEW_XLSX
    try:
        review_handle = REVIEW_CSV.open("w", encoding="utf-8-sig", newline="")
    except PermissionError:
        written_review_csv = REVIEW_FALLBACK_CSV
        written_review_xlsx = REVIEW_FALLBACK_XLSX
        review_handle = written_review_csv.open("w", encoding="utf-8-sig", newline="")
    with review_handle as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=review_fieldnames,
        )
        writer.writeheader()
        for entry in catalogue.entries:
            preserved = previous_decisions.get(entry.semantic_sign_id, {})
            reviewer_decision = preserved.get("reviewer_2_decision", "")
            reviewer_notes = preserved.get("reviewer_2_notes", "")
            if entry.review_status == "approved" and not reviewer_decision:
                reviewer_decision = "approve"
                reviewer_notes = "Approved in catalogue review_status."
            row = {
                "semantic_sign_id": entry.semantic_sign_id,
                "category": entry.category,
                "name_en": entry.names.en,
                "name_ms": entry.names.ms,
                "name_zh": entry.names.zh,
                "base_action": entry.base_action,
                "severity": entry.severity,
                "parameter_type": entry.parameter_type,
                "default_parameter": entry.default_parameter or "",
                "standard_reference": entry.standard_reference.reference_id,
                "review_status": entry.review_status,
                "reviewer_2_decision": reviewer_decision,
                "reviewer_2_notes": reviewer_notes,
            }
            writer.writerow(row)
            review_rows.append(row)
    _write_xlsx(written_review_xlsx, review_rows, review_fieldnames)

    audit = {
        "generated_on": date.today().isoformat(),
        "catalogue_version": catalogue.catalogue_version,
        "entry_count": len(catalogue.entries),
        "standards_count": len(standards.documents),
        "machine_checks": {
            "required_fields_valid": True,
            "references_registered": not missing_registered_reference,
            "local_archives_present": not missing_local_archive,
            "critical_entries_approved": not critical_not_approved,
            "all_entries_approved": not draft_entries,
        },
        "open_items": {
            "missing_registered_reference": missing_registered_reference,
            "missing_local_archive": missing_local_archive,
            "critical_not_approved": critical_not_approved,
            "draft_entries": draft_entries,
            "unresolved_coursework_mappings": unresolved_coursework,
        },
        "review_workbook": str(written_review_csv.relative_to(project_path("."))),
        "review_workbook_xlsx": str(written_review_xlsx.relative_to(project_path("."))),
        "unresolved_coursework_review": str(UNRESOLVED_CSV.relative_to(project_path("."))),
        "unresolved_coursework_sheet": str(UNRESOLVED_SHEET.relative_to(project_path("."))),
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {AUDIT_JSON}")
    print(f"Wrote {written_review_csv}")
    print(f"Wrote {written_review_xlsx}")
    print(f"Wrote {UNRESOLVED_CSV}")
    print(f"Wrote {UNRESOLVED_SHEET}")
    print(json.dumps(audit["machine_checks"], indent=2))


if __name__ == "__main__":
    main()
