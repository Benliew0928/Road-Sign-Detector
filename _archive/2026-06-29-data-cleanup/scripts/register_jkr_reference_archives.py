from __future__ import annotations

import hashlib
import json
from pathlib import Path

from roadsign_assist.paths import project_path


MANIFEST_PATH = project_path("configs/catalogue/standards_manifest.json")
ARCHIVE_ROOT = project_path("docs/references/jkr")
EXPECTED_ARCHIVES = {
    "jkr_atj_2a_85_2019": "jkr_atj_2a_85_2019.pdf",
    "jkr_atj_2b_85_2019": "jkr_atj_2b_85_2019.pdf",
    "jkr_atj_2c_85_2017": "jkr_atj_2c_85_2017.pdf",
    "jkr_road_sign_usage_2021": "jkr_road_sign_usage_2021.pdf",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    missing: list[str] = []
    for document in manifest["documents"]:
        reference_id = document["reference_id"]
        expected_name = EXPECTED_ARCHIVES.get(reference_id)
        if expected_name is None:
            continue
        archive_path = ARCHIVE_ROOT / expected_name
        if not archive_path.exists():
            missing.append(f"{reference_id}: {archive_path}")
            continue
        document["local_archive_status"] = "archived"
        document["local_archive_path"] = archive_path.relative_to(project_path(".")).as_posix()
        document["local_archive_sha256"] = _sha256(archive_path)
        document["local_archive_bytes"] = archive_path.stat().st_size
        document["notes"] = (
            document.get("notes", "").split(" PowerShell and Python direct archival attempts")[0]
            + " Local archive registered with SHA-256 evidence."
        ).strip()

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if missing:
        print("Missing local reference PDFs:")
        for item in missing:
            print(f"- {item}")
        raise SystemExit(1)
    print(f"Registered {len(EXPECTED_ARCHIVES)} local JKR reference archives.")


if __name__ == "__main__":
    main()
