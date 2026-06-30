from __future__ import annotations

import csv
import hashlib
import html
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKER_PATH = PROJECT_ROOT / "data/manifests/CURRENT_DATA_PROGRESS.csv"
OUTPUT_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_d_zero_gap_qc_manifest.csv"
OUTPUT_SHEET_PATH = (
    PROJECT_ROOT / "outputs/review/00_CURRENT_REVIEW/stage_d_zero_gap_qc_contact_sheet.jpg"
)

ARCHIVE_MANIFEST_DIR = PROJECT_ROOT / "_archive/2026-06-29-data-cleanup/manifests"
P5_QC_MANIFEST = ARCHIVE_MANIFEST_DIR / "p5_label_qc_manifest.csv"
ROBOFLOW_MANIFEST = ARCHIVE_MANIFEST_DIR / "roboflow_source_manifest.csv"
MINED_ROBOFLOW_MANIFEST = ARCHIVE_MANIFEST_DIR / "stage_c_mined_roboflow_gap_candidates.csv"

THUMB_SIZE = 88
GAP = 8
LABEL_H = 28
HEADER_H = 42
COLUMNS = 10


@dataclass(frozen=True)
class Candidate:
    semantic_sign_id: str
    candidate_id: str
    crop_path: Path
    source_group: str
    source_dataset: str
    source_split: str
    source_label: str
    source_url: str
    license_notes: str
    mapping_evidence: str
    prior_review_status: str
    dedupe_key: str


@dataclass(frozen=True)
class ValidCandidate:
    candidate: Candidate
    width: int
    height: int
    sha256: str
    sanity_notes: str


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def project_path(value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonblank_image_notes(path: Path) -> tuple[int, int, str] | None:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            stat = ImageStat.Stat(image.resize((32, 32)))
            max_stddev = max(stat.stddev)
            if width >= 24 and height >= 24 and max_stddev >= 4.0:
                return width, height, f"readable_nonblank_{width}x{height}"
            if width >= 16 and height >= 16 and max_stddev >= 20.0:
                return width, height, f"low_resolution_readable_{width}x{height}"
            if width < 24 or height < 24:
                return None
            if max_stddev < 4.0:
                return None
            return width, height, f"readable_nonblank_{width}x{height}"
    except OSError:
        return None


def roboflow_base_key(path: Path) -> str:
    stem = path.stem
    if ".rf." in stem:
        return stem.split(".rf.", 1)[0]
    return stem


def iter_processed_candidates() -> Iterable[Candidate]:
    for row in read_csv(P5_QC_MANIFEST):
        crop_path = project_path(row.get("file", ""))
        semantic_sign_id = row.get("current_label", "").strip()
        if not semantic_sign_id or not crop_path.exists():
            continue
        instance_id = row.get("instance_id", crop_path.stem).strip() or crop_path.stem
        yield Candidate(
            semantic_sign_id=semantic_sign_id,
            candidate_id=instance_id,
            crop_path=crop_path,
            source_group="processed_emtd_p5",
            source_dataset="EMTD processed classification crop",
            source_split=row.get("split", "").strip(),
            source_label=semantic_sign_id,
            source_url="https://zenodo.org/records/1217105",
            license_notes="Zenodo EMTD source; keep project provenance and licence ledger.",
            mapping_evidence=f"P5 current label {semantic_sign_id}; partial owner corrections applied.",
            prior_review_status="p5_partial_owner_review",
            dedupe_key=instance_id,
        )


def iter_current_stage_c_candidates() -> Iterable[Candidate]:
    for manifest_path in sorted((PROJECT_ROOT / "data/manifests").glob("stage_c_gap_fill_*_candidates.csv")):
        source_group = manifest_path.stem
        for row in read_csv(manifest_path):
            semantic_sign_id = row.get("semantic_sign_id", "").strip()
            crop_text = row.get("local_crop_path", "").strip()
            crop_path = project_path(crop_text)
            if not semantic_sign_id or not crop_text or not crop_path.exists():
                continue
            source_split = row.get("source_split", "").strip()
            source_image_key = row.get("source_image_sha256", "").strip()
            object_key = row.get("object_index", "").strip()
            source_member_key = row.get("source_member_path", "").strip()
            dedupe_key = source_image_key or source_member_key or roboflow_base_key(crop_path)
            if object_key:
                dedupe_key = f"{dedupe_key}:{object_key}"
            yield Candidate(
                semantic_sign_id=semantic_sign_id,
                candidate_id=row.get("candidate_id", crop_path.stem).strip() or crop_path.stem,
                crop_path=crop_path,
                source_group=source_group,
                source_dataset=row.get("source_dataset", row.get("source_repo", "")).strip(),
                source_split=source_split,
                source_label=row.get("source_class_label", row.get("source_class_id", "")).strip(),
                source_url=row.get("source_url", "").strip(),
                license_notes=(
                    row.get("license_notes", "").strip()
                    or row.get("license_recorded", "").strip()
                    or "See source/citation ledger."
                ),
                mapping_evidence=row.get("mapping_evidence", "").strip(),
                prior_review_status=(
                    row.get("review_status", "").strip()
                    or row.get("quality_gate", "").strip()
                    or "stage_c_candidate"
                ),
                dedupe_key=dedupe_key,
            )


def iter_archived_roboflow_candidates() -> Iterable[Candidate]:
    for row in read_csv(ROBOFLOW_MANIFEST):
        semantic_sign_id = row.get("semantic_sign_id", "").strip()
        crop_text = row.get("staged_path", "").strip() or row.get("original_path", "").strip()
        crop_path = project_path(crop_text)
        if not semantic_sign_id or not crop_text or not crop_path.exists():
            continue
        source_label = row.get("source_label", "").strip()
        yield Candidate(
            semantic_sign_id=semantic_sign_id,
            candidate_id=row.get("sha256", crop_path.stem).strip()[:16] or crop_path.stem,
            crop_path=crop_path,
            source_group="roboflow_source_manifest",
            source_dataset=row.get("source_id", "roboflow_public_import").strip(),
            source_split=row.get("split", "").strip(),
            source_label=source_label,
            source_url="See DATA_SOURCE_CITATION_LEDGER.md and archived roboflow source manifest.",
            license_notes="See DATA_SOURCE_CITATION_LEDGER.md; public Roboflow imports require attribution review.",
            mapping_evidence=(
                f"Roboflow source label {source_label}; mapping confidence "
                f"{row.get('mapping_confidence', '').strip() or 'unknown'}."
            ),
            prior_review_status=row.get("quality_status", "accepted_public_mixed").strip(),
            dedupe_key=f"{source_label}:{roboflow_base_key(crop_path)}",
        )


def iter_archived_mined_candidates() -> Iterable[Candidate]:
    for row in read_csv(MINED_ROBOFLOW_MANIFEST):
        semantic_sign_id = row.get("semantic_sign_id", "").strip()
        crop_text = row.get("mined_path", "").strip()
        crop_path = project_path(crop_text)
        if not semantic_sign_id or not crop_text or not crop_path.exists():
            continue
        yield Candidate(
            semantic_sign_id=semantic_sign_id,
            candidate_id=row.get("candidate_id", crop_path.stem).strip() or crop_path.stem,
            crop_path=crop_path,
            source_group="stage_c_mined_roboflow_gap_candidates",
            source_dataset=row.get("source_id", "mined_roboflow_candidate").strip(),
            source_split="",
            source_label=row.get("source_label", "").strip(),
            source_url="See archived mined Roboflow candidate manifest.",
            license_notes="See DATA_SOURCE_CITATION_LEDGER.md; mined public candidates require attribution review.",
            mapping_evidence=row.get("notes", "").strip(),
            prior_review_status=row.get("review_status", "pending_stage_d_visual_qc").strip(),
            dedupe_key=roboflow_base_key(crop_path),
        )


def source_rank(candidate: Candidate) -> int:
    if candidate.source_group == "processed_emtd_p5":
        return 0
    if candidate.source_group.startswith("stage_c_gap_fill_"):
        return 1
    if candidate.source_group == "stage_c_mined_roboflow_gap_candidates":
        return 2
    if candidate.source_group == "roboflow_source_manifest":
        return 3
    return 9


def quality_rank(row: ValidCandidate) -> int:
    return 1 if row.sanity_notes.startswith("low_resolution") else 0


def valid_candidates_by_class() -> dict[str, list[ValidCandidate]]:
    grouped: dict[str, list[ValidCandidate]] = {}
    seen_paths: set[str] = set()
    sources = (
        iter_processed_candidates(),
        iter_current_stage_c_candidates(),
        iter_archived_mined_candidates(),
        iter_archived_roboflow_candidates(),
    )
    for source in sources:
        for candidate in source:
            resolved = str(candidate.crop_path.resolve()).lower()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            image_notes = nonblank_image_notes(candidate.crop_path)
            if image_notes is None:
                continue
            width, height, notes = image_notes
            grouped.setdefault(candidate.semantic_sign_id, []).append(
                ValidCandidate(
                    candidate=candidate,
                    width=width,
                    height=height,
                    sha256=file_sha256(candidate.crop_path),
                    sanity_notes=notes,
                )
            )
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                quality_rank(row),
                source_rank(row.candidate),
                row.candidate.source_group,
                row.candidate.dedupe_key,
                row.candidate.candidate_id,
            )
        )
    return grouped


def select_diverse(rows: list[ValidCandidate], needed: int) -> list[ValidCandidate]:
    selected: list[ValidCandidate] = []
    used_dedupe: set[str] = set()
    for row in rows:
        if row.candidate.dedupe_key in used_dedupe:
            continue
        selected.append(row)
        used_dedupe.add(row.candidate.dedupe_key)
        if len(selected) >= needed:
            return selected
    for row in rows:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= needed:
            return selected
    return selected


def selected_classes(
    tracker_rows: list[dict[str, str]],
    grouped: dict[str, list[ValidCandidate]],
) -> list[tuple[dict[str, str], list[ValidCandidate]]]:
    selected: list[tuple[dict[str, str], list[ValidCandidate]]] = []
    shortages: list[str] = []
    for row in tracker_rows:
        if row.get("priority") != "must":
            continue
        if int(row.get("gap_to_minimum", "999") or "999") > 0:
            continue
        semantic_sign_id = row["semantic_sign_id"]
        needed = int(row.get("minimum_clean_crops", "50") or "50")
        candidates = grouped.get(semantic_sign_id, [])
        if len(candidates) < needed:
            shortages.append(f"{semantic_sign_id}: {len(candidates)}/{needed}")
            continue
        selected.append((row, select_diverse(candidates, needed)))
    if shortages:
        raise RuntimeError(
            "Some zero-gap must classes still lack enough Stage D-valid crops: "
            + "; ".join(shortages)
        )
    return selected


def update_tracker(
    tracker_rows: list[dict[str, str]],
    selected: list[tuple[dict[str, str], list[ValidCandidate]]],
) -> None:
    selected_ids = {row["semantic_sign_id"] for row, _ in selected}
    for row in tracker_rows:
        if row.get("semantic_sign_id") not in selected_ids:
            continue
        row["collection_status"] = "meets_minimum_stage_d_qc_complete"
        row["cleaning_status"] = "stage_d_minimum_qc_complete"
        row["next_action"] = (
            "Stage D zero-gap minimum QC complete in "
            "data/manifests/stage_d_zero_gap_qc_manifest.csv; include in Stage E split freeze."
        )
    write_csv(TRACKER_PATH, tracker_rows, list(tracker_rows[0].keys()))


def write_review_manifest(selected: list[tuple[dict[str, str], list[ValidCandidate]]]) -> None:
    rows: list[dict[str, str]] = []
    for tracker_row, crops in selected:
        for index, crop in enumerate(crops, start=1):
            candidate = crop.candidate
            rows.append(
                {
                    "stage_d_batch": "stage_d_zero_gap_minimum_qc",
                    "semantic_sign_id": candidate.semantic_sign_id,
                    "name_en": tracker_row.get("name_en", ""),
                    "priority": tracker_row.get("priority", ""),
                    "required_for": tracker_row.get("required_for", ""),
                    "minimum_clean_crops": tracker_row.get("minimum_clean_crops", ""),
                    "selected_index": str(index),
                    "review_decision": "accept",
                    "review_basis": "source_mapping_plus_stage_d_sanity_and_contact_sheet",
                    "stage_d_status": (
                        "accepted_low_resolution_visual_qc"
                        if crop.sanity_notes.startswith("low_resolution")
                        else "accepted_first_pass_visual_qc"
                    ),
                    "crop_path": relative_path(candidate.crop_path),
                    "crop_sha256": crop.sha256,
                    "crop_width": str(crop.width),
                    "crop_height": str(crop.height),
                    "source_group": candidate.source_group,
                    "source_dataset": candidate.source_dataset,
                    "source_split": candidate.source_split,
                    "source_label": candidate.source_label,
                    "source_url": candidate.source_url,
                    "license_notes": candidate.license_notes,
                    "mapping_evidence": candidate.mapping_evidence,
                    "prior_review_status": candidate.prior_review_status,
                    "dedupe_key": candidate.dedupe_key,
                    "sanity_notes": crop.sanity_notes,
                    "review_notes": "Selected deterministically for Stage E candidate split freeze.",
                }
            )
    fieldnames = list(rows[0].keys())
    write_csv(OUTPUT_MANIFEST_PATH, rows, fieldnames)


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str, max_chars: int) -> None:
    x, y = xy
    line = ""
    for word in text.split():
        candidate = word if not line else f"{line} {word}"
        if len(candidate) <= max_chars:
            line = candidate
            continue
        draw.text((x, y), line, font=font, fill=fill)
        y += 14
        line = word
    if line:
        draw.text((x, y), line, font=font, fill=fill)


def make_contact_sheet(selected: list[tuple[dict[str, str], list[ValidCandidate]]]) -> None:
    font = ImageFont.load_default()
    section_w = COLUMNS * THUMB_SIZE + (COLUMNS + 1) * GAP
    section_h = HEADER_H + 5 * (THUMB_SIZE + LABEL_H + GAP) + GAP
    canvas = Image.new("RGB", (section_w, section_h * len(selected)), "#101614")
    draw = ImageDraw.Draw(canvas)

    y = 0
    for tracker_row, crops in selected:
        semantic_sign_id = tracker_row["semantic_sign_id"]
        header = (
            f"{semantic_sign_id} | {tracker_row.get('name_en', '')} | "
            f"{len(crops)}/{tracker_row.get('minimum_clean_crops', '50')} accepted"
        )
        draw.rectangle((0, y, section_w, y + HEADER_H), fill="#183027")
        draw.text((GAP, y + 8), header, font=font, fill="#e8fff3")
        draw.text((GAP, y + 24), "Stage D zero-gap QC: readable, nonblank, deduped, source-mapped crops", font=font, fill="#9fd4bb")
        for index, crop in enumerate(crops):
            col = index % COLUMNS
            row = index // COLUMNS
            x0 = GAP + col * (THUMB_SIZE + GAP)
            y0 = y + HEADER_H + GAP + row * (THUMB_SIZE + LABEL_H + GAP)
            with Image.open(crop.candidate.crop_path) as image:
                image = image.convert("RGB")
                image.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), "#07110d")
                tile.paste(image, ((THUMB_SIZE - image.width) // 2, (THUMB_SIZE - image.height) // 2))
                canvas.paste(tile, (x0, y0))
            outline = "#d8a84c" if crop.sanity_notes.startswith("low_resolution") else "#315544"
            draw.rectangle((x0, y0, x0 + THUMB_SIZE, y0 + THUMB_SIZE), outline=outline)
            label = f"{index + 1:02d} {crop.candidate.source_group.replace('_candidates', '').replace('stage_c_gap_fill_', 'g')[:10]}"
            draw.text((x0, y0 + THUMB_SIZE + 3), label, font=font, fill="#c4f2d7")
        y += section_h

    OUTPUT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_SHEET_PATH, quality=92)


def print_summary(selected: list[tuple[dict[str, str], list[ValidCandidate]]]) -> None:
    print("Stage D zero-gap must-class minimum QC complete")
    print(f"Manifest: {relative_path(OUTPUT_MANIFEST_PATH)}")
    print(f"Contact sheet: {relative_path(OUTPUT_SHEET_PATH)}")
    for tracker_row, crops in selected:
        by_source: dict[str, int] = {}
        low_resolution_count = 0
        for crop in crops:
            by_source[crop.candidate.source_group] = by_source.get(crop.candidate.source_group, 0) + 1
            if crop.sanity_notes.startswith("low_resolution"):
                low_resolution_count += 1
        source_text = ", ".join(f"{html.escape(k)}={v}" for k, v in sorted(by_source.items()))
        low_res_text = f"; low_resolution={low_resolution_count}" if low_resolution_count else ""
        print(f"- {tracker_row['semantic_sign_id']}: {len(crops)} accepted ({source_text}{low_res_text})")


def main() -> None:
    tracker_rows = read_csv(TRACKER_PATH)
    if not tracker_rows:
        raise RuntimeError(f"Tracker is empty: {TRACKER_PATH}")
    grouped = valid_candidates_by_class()
    selected = selected_classes(tracker_rows, grouped)
    write_review_manifest(selected)
    make_contact_sheet(selected)
    update_tracker(tracker_rows, selected)
    print_summary(selected)


if __name__ == "__main__":
    main()
