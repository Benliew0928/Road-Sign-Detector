from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOFLOW_MANIFEST = PROJECT_ROOT / "data/manifests/roboflow_source_manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data/staging/stage_c_mined_roboflow_gap_candidates"
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/stage_c_mined_roboflow_gap_candidates.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/audit/stage_c_mined_roboflow_gap_candidates.json"
SHEET_ROOT = PROJECT_ROOT / "outputs/review/stage_c_mined_roboflow_gap_candidates"


@dataclass(frozen=True)
class MiningRule:
    semantic_sign_id: str
    source_label: str
    source_id: str
    max_candidates: int
    reason: str
    filter_fn: Callable[[Path], tuple[bool, dict[str, float], str]]


FIELDNAMES = [
    "stage_id",
    "candidate_id",
    "semantic_sign_id",
    "source_id",
    "source_label",
    "source_quality_status",
    "source_rejection_reason",
    "source_original_path",
    "source_staged_path",
    "mined_path",
    "mining_status",
    "review_status",
    "filter_summary",
    "notes",
]

STAGE_ID = "stage_c_mined_roboflow_gap_candidates_01"


def image_metrics(path: Path) -> dict[str, float]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        small = image.resize((96, 96))
    pixels = list(small.getdata())
    total = len(pixels)
    red = blue = yellow = white = dark = 0
    for r, g, b in pixels:
        mx = max(r, g, b)
        mn = min(r, g, b)
        if b > 80 and b > r * 1.25 and b > g * 1.05:
            blue += 1
        if r > 110 and r > g * 1.25 and r > b * 1.25:
            red += 1
        if r > 140 and g > 120 and b < 100:
            yellow += 1
        if r > 215 and g > 215 and b > 215:
            white += 1
        if mx < 70 and mx - mn < 45:
            dark += 1
    return {
        "blue_fraction": blue / total,
        "red_fraction": red / total,
        "yellow_fraction": yellow / total,
        "white_fraction": white / total,
        "dark_fraction": dark / total,
    }


def filter_pass_either_side(path: Path) -> tuple[bool, dict[str, float], str]:
    metrics = image_metrics(path)
    # True pass-either-side signs are normally blue mandatory signs. This rejects
    # the many speed-limit/red-circle outliers in the noisy source class.
    passes = (
        metrics["blue_fraction"] >= 0.18
        and metrics["white_fraction"] >= 0.08
        and metrics["red_fraction"] <= 0.18
        and metrics["yellow_fraction"] <= 0.10
    )
    reason = "blue_mandatory_candidate" if passes else "not_blue_mandatory_shape"
    return passes, metrics, reason


def filter_yellow_warning(path: Path) -> tuple[bool, dict[str, float], str]:
    metrics = image_metrics(path)
    passes = (
        metrics["yellow_fraction"] >= 0.12
        and metrics["dark_fraction"] >= 0.04
        and metrics["red_fraction"] <= 0.12
    )
    reason = "yellow_warning_candidate" if passes else "not_yellow_warning_shape"
    return passes, metrics, reason


RULES = [
    MiningRule(
        semantic_sign_id="pass_either_side",
        source_label="Pass either side",
        source_id="roboflow_malaysia_road_sign_v1",
        max_candidates=120,
        reason=(
            "The Roboflow source label was rejected as noisy/mixed, but blue-color "
            "filtering can recover likely mandatory pass-either-side candidates."
        ),
        filter_fn=filter_pass_either_side,
    ),
    MiningRule(
        semantic_sign_id="side_road_right",
        source_label="Crossroad on the right",
        source_id="roboflow_malaysia_road_sign_v1",
        max_candidates=15,
        reason=(
            "The accepted pool left side_road_right slightly below the realistic "
            "minimum. Rejected rows from the same exact source label are mined "
            "only when they still look like yellow warning-sign candidates."
        ),
        filter_fn=filter_yellow_warning,
    )
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def source_image_path(row: dict[str, str]) -> Path | None:
    for key in ("staged_path", "original_path"):
        value = row.get(key, "")
        if value:
            path = PROJECT_ROOT / value
            if path.exists():
                return path
    return None


def copy_candidate(source_path: Path, rule: MiningRule, index: int) -> Path:
    destination = OUTPUT_ROOT / rule.semantic_sign_id / f"{STAGE_ID}_{rule.semantic_sign_id}_{index:04d}{source_path.suffix.lower()}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source_path, destination)
    return destination


def mine_rule(rule: MiningRule, source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    matching_rows = [
        row
        for row in source_rows
        if row.get("source_id") == rule.source_id and row.get("source_label") == rule.source_label
    ]
    for row in matching_rows:
        path = source_image_path(row)
        if path is None:
            continue
        passed, metrics, reason = rule.filter_fn(path)
        if not passed:
            continue
        candidate_number = len(candidates) + 1
        mined_path = copy_candidate(path, rule, candidate_number)
        candidates.append(
            {
                "stage_id": STAGE_ID,
                "candidate_id": f"MINE01-{candidate_number:04d}",
                "semantic_sign_id": rule.semantic_sign_id,
                "source_id": row.get("source_id", ""),
                "source_label": row.get("source_label", ""),
                "source_quality_status": row.get("quality_status", ""),
                "source_rejection_reason": row.get("rejection_reason", ""),
                "source_original_path": row.get("original_path", ""),
                "source_staged_path": row.get("staged_path", ""),
                "mined_path": mined_path.relative_to(PROJECT_ROOT).as_posix(),
                "mining_status": "mined_candidate",
                "review_status": "pending_stage_d_visual_qc",
                "filter_summary": json.dumps(metrics, sort_keys=True),
                "notes": f"{rule.reason} Filter reason: {reason}.",
            }
        )
        if len(candidates) >= rule.max_candidates:
            break
    return candidates


def render_sheet(rows: list[dict[str, str]], output_path: Path, *, columns: int = 6) -> str:
    tile_width = 190
    tile_height = 210
    label_height = 58
    sheet = Image.new(
        "RGB",
        (columns * tile_width, max(1, math.ceil(len(rows) / columns)) * tile_height),
        color=(18, 23, 22),
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        image_path = PROJECT_ROOT / row["mined_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            thumbnail = ImageOps.contain(
                background.convert("RGB"),
                (tile_width - 14, tile_height - label_height - 14),
            )
        sheet.paste(
            thumbnail,
            (
                x + (tile_width - thumbnail.width) // 2,
                y + 7 + (tile_height - label_height - 14 - thumbnail.height) // 2,
            ),
        )
        draw.rectangle((x, y + tile_height - label_height, x + tile_width, y + tile_height), fill=(32, 43, 39))
        draw.text((x + 6, y + tile_height - label_height + 6), row["candidate_id"], fill=(232, 239, 236))
        draw.text((x + 6, y + tile_height - label_height + 24), row["semantic_sign_id"][:26], fill=(232, 239, 236))
        draw.text((x + 6, y + tile_height - label_height + 42), row["source_rejection_reason"][:24], fill=(180, 198, 190))
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline=(65, 77, 73), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def build_sheets(rows: list[dict[str, str]]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["semantic_sign_id"], []).append(row)
    for semantic_id, class_rows in sorted(by_class.items()):
        outputs[semantic_id] = render_sheet(class_rows, SHEET_ROOT / f"{semantic_id}.jpg")
    if rows:
        outputs["_all_mined_candidates"] = render_sheet(rows, SHEET_ROOT / "_all_mined_candidates.jpg")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine exact candidates from Roboflow noisy/mixed gap classes.")
    parser.add_argument("--classes", nargs="*", default=[], help="Optional semantic_sign_id classes to mine.")
    args = parser.parse_args()

    selected = set(args.classes)
    source_rows = read_rows(ROBOFLOW_MANIFEST)
    mined_rows: list[dict[str, str]] = []
    rule_summaries = []
    for rule in RULES:
        if selected and rule.semantic_sign_id not in selected:
            continue
        rows = mine_rule(rule, source_rows)
        mined_rows.extend(rows)
        rule_summaries.append(
            {
                "semantic_sign_id": rule.semantic_sign_id,
                "source_id": rule.source_id,
                "source_label": rule.source_label,
                "mined_candidates": len(rows),
                "max_candidates": rule.max_candidates,
                "reason": rule.reason,
            }
        )

    write_csv(MANIFEST_PATH, mined_rows)
    sheets = build_sheets(mined_rows)
    status_counts = Counter(row["review_status"] for row in mined_rows)
    report = {
        "schema_version": "1.0",
        "stage_id": STAGE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": ROBOFLOW_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        "output_manifest": MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "output_root": OUTPUT_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "sheets": sheets,
        "total_mined_candidates": len(mined_rows),
        "review_status_counts": dict(status_counts),
        "rules": rule_summaries,
        "status": "mined_candidates_pending_stage_d_visual_qc",
        "important_note": (
            "These candidates are mined from a previously rejected/noisy Roboflow source class. "
            "They must be visually QCed before final dataset freeze."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    for name, path in sheets.items():
        print(f"Wrote {name}: {path}")
    print(json.dumps({"total_mined_candidates": len(mined_rows), "rules": rule_summaries}, indent=2))


if __name__ == "__main__":
    main()
