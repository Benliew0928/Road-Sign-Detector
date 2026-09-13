"""Audit offline advisory packs; --repair fixes WAV sizes without changing PCM samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inspect_wav(path: Path, repair: bool = False) -> dict:
    content = bytearray(path.read_bytes())
    if content[:4] != b"RIFF" or content[8:12] != b"WAVE":
        raise ValueError(f"Not a PCM WAV: {path}")
    offset = 12
    data_offset = None
    while offset + 8 <= len(content):
        kind = content[offset : offset + 4]
        size = struct.unpack_from("<I", content, offset + 4)[0]
        if kind == b"data":
            data_offset = offset
            break
        offset += 8 + size + size % 2
    if data_offset is None:
        raise ValueError(f"Missing data: {path}")
    declared = struct.unpack_from("<I", content, data_offset + 4)[0]
    available = len(content) - data_offset - 8
    payload_size = min(declared, available)
    payload_hash = hashlib.sha256(
        content[data_offset + 8 : data_offset + 8 + payload_size]
    ).hexdigest()
    broken = declared > available or struct.unpack_from("<I", content, 4)[0] != len(content) - 8
    if broken and repair:
        struct.pack_into("<I", content, 4, len(content) - 8)
        struct.pack_into("<I", content, data_offset + 4, payload_size)
        path.write_bytes(content)
        assert (
            hashlib.sha256(content[data_offset + 8 : data_offset + 8 + payload_size]).hexdigest()
            == payload_hash
        )
    with wave.open(str(path), "rb") as wav:
        frame_bytes = wav.getnchannels() * wav.getsampwidth()
        if payload_size == 0 or payload_size % frame_bytes:
            raise ValueError(f"Incomplete PCM frames: {path}")
        duration = payload_size / frame_bytes / wav.getframerate()
    return {
        "broken": broken,
        "duration_seconds": round(duration, 3),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "pcm_sha256": payload_hash,
    }


def audit(repair: bool = False) -> dict:
    public = ROOT / "apps/web/public"
    labels = json.loads(
        (ROOT / "models/exported/runtime/sign_classifier.labels.json").read_text(encoding="utf-8")
    )
    report = {}
    for pack in ("p16", "p16_ai"):
        path = public / "audio" / pack / "advisory_audio_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        missing = sorted(set(labels) - set(manifest["semantic_phrase_ids"]))
        errors = [f"Missing labels: {missing}"] if missing else []
        repaired = []
        count = 0
        for phrase_id in manifest["semantic_phrase_ids"].values():
            if phrase_id not in manifest["phrases"]:
                errors.append(f"Unresolved phrase: {phrase_id}")
        for phrase_id, phrase in manifest["phrases"].items():
            for language in ("en", "ms", "zh"):
                asset = phrase["assets"][language]
                info = inspect_wav(public / asset["src"].lstrip("/"), repair)
                count += 1
                if info["broken"]:
                    if repair:
                        repaired.append(
                            {
                                "phrase": phrase_id,
                                "language": language,
                                "pcm_sha256": info["pcm_sha256"],
                            }
                        )
                    else:
                        errors.append(f"Malformed WAV: {phrase_id}/{language}")
                for field in ("bytes", "sha256", "duration_seconds"):
                    if repair:
                        asset[field] = info[field]
                    elif asset[field] != info[field]:
                        errors.append(f"Metadata mismatch: {phrase_id}/{language}/{field}")
                if (
                    asset.get("fallback_src")
                    and not (public / asset["fallback_src"].lstrip("/")).is_file()
                ):
                    errors.append(f"Missing fallback: {phrase_id}/{language}")
        if repair:
            if "audio_pack" in manifest:
                manifest["audio_pack"]["status"] = "complete"
            path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        report[pack] = {
            "classes_covered": len(labels) - len(missing),
            "assets": count,
            "repaired": repaired,
            "errors": errors,
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.repair)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {pack: {**data, "repaired": len(data["repaired"])} for pack, data in result.items()},
            indent=2,
        )
    )
    raise SystemExit(int(any(data["errors"] for data in result.values())))
