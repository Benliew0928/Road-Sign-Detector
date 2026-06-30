from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from roadsign_assist.audio.advisory import (
    build_advisory_manifest,
    update_asset_metadata,
    write_manifest,
)
from roadsign_assist.paths import PROJECT_ROOT


def _voice_names(value: str) -> dict[str, str]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--voice-names must be a JSON object")
    voice_map = cast(dict[str, object], parsed)
    return {str(key): str(item) for key, item in voice_map.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build P16 offline advisory audio manifest.")
    parser.add_argument(
        "--output",
        default="apps/web/public/audio/p16/advisory_audio_manifest.json",
    )
    parser.add_argument("--public-root", default="apps/web/public")
    parser.add_argument("--update-assets", action="store_true")
    parser.add_argument("--voice-names", default="")
    parser.add_argument("--voice-names-file", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = PROJECT_ROOT / args.output
    public_root = PROJECT_ROOT / args.public_root
    manifest: dict[str, Any] = build_advisory_manifest()
    if args.update_assets:
        voice_names = args.voice_names
        if args.voice_names_file:
            voice_names = Path(args.voice_names_file).read_text(encoding="utf-8-sig")
        manifest = update_asset_metadata(manifest, public_root, _voice_names(voice_names))
    write_manifest(output, manifest)
    print(f"Wrote {len(manifest['phrases'])} advisory phrases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
