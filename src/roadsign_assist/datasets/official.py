from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from roadsign_assist.paths import MANIFEST_ROOT, OFFICIAL_ROOT

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".ppm"}
COURSEWORK_PREFIX = re.compile(r"^(?P<class_id>\d{3})(?:_|$)")


@dataclass(frozen=True)
class OfficialImage:
    image_id: str
    relative_path: str
    filename: str
    coursework_id_candidate: str
    width: int
    height: int
    mode: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_official_images() -> list[OfficialImage]:
    root = OFFICIAL_ROOT / "assignment_images"
    rows: list[OfficialImage] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        match = COURSEWORK_PREFIX.match(path.stem)
        candidate = f"sign_{match.group('class_id')}" if match else ""
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        rows.append(
            OfficialImage(
                image_id=relative.replace("/", "__").rsplit(".", 1)[0],
                relative_path=relative,
                filename=path.name,
                coursework_id_candidate=candidate,
                width=width,
                height=height,
                mode=mode,
                sha256=_sha256(path),
            )
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty manifest to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_official_manifests() -> list[OfficialImage]:
    rows = inventory_official_images()
    if len(rows) != 84:
        raise ValueError(f"Expected 84 official coursework images, found {len(rows)}")
    _write_csv(MANIFEST_ROOT / "official_images.csv", [asdict(row) for row in rows])
    _write_csv(
        MANIFEST_ROOT / "official_checksums.csv",
        [{"relative_path": row.relative_path, "sha256": row.sha256} for row in rows],
    )
    return rows


def write_coursework_review_manifest(
    mapping_path: str | Path = "configs/catalogue/coursework_draft_mapping.json",
) -> Path:
    from roadsign_assist.paths import project_path

    images = inventory_official_images()
    mapping_file = project_path(mapping_path)
    payload = json.loads(mapping_file.read_text(encoding="utf-8"))
    mappings: dict[str, dict[str, object]] = payload["mappings"]
    candidate_ids = {image.coursework_id_candidate for image in images}
    missing = sorted(candidate_ids - mappings.keys())
    extra = sorted(mappings.keys() - candidate_ids)
    if missing or extra:
        raise ValueError(f"Coursework mapping mismatch: missing={missing}, extra={extra}")
    from roadsign_assist.catalogue.repository import catalogue_by_id

    known_semantics = catalogue_by_id()
    invalid_semantics = sorted(
        {
            str(mapping["semantic_sign_id"])
            for mapping in mappings.values()
            if mapping["semantic_sign_id"]
            and str(mapping["semantic_sign_id"]) not in known_semantics
        }
    )
    if invalid_semantics:
        raise ValueError(f"Unknown semantic IDs in coursework mapping: {invalid_semantics}")

    rows: list[dict[str, object]] = []
    for image in images:
        mapping = mappings[image.coursework_id_candidate]
        rows.append(
            {
                "image_id": image.image_id,
                "relative_path": image.relative_path,
                "coursework_id_candidate": image.coursework_id_candidate,
                "verified_coursework_id": image.coursework_id_candidate,
                "semantic_sign_id": mapping["semantic_sign_id"] or "",
                "parameter_value": mapping["parameter_value"] or "",
                "reviewer_1": payload["reviewer_1"],
                "reviewer_2": payload["reviewer_2"] or "",
                "review_status": payload["review_status"],
                "confidence": mapping["confidence"],
                "notes": mapping["notes"],
            }
        )
    output = MANIFEST_ROOT / "coursework_manifest.csv"
    _write_csv(output, rows)
    return output
