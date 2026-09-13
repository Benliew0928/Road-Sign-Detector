from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

from scripts.audit_advisory_audio import audit, inspect_wav


def test_all_advisory_packs_cover_runtime_and_have_valid_audio() -> None:
    report = audit()
    for pack in report.values():
        assert pack["classes_covered"] == 78
        assert pack["assets"] == 546
        assert pack["errors"] == []


def test_header_repair_preserves_pcm_and_is_idempotent(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x02" * 24000)
    data = bytearray(stream.getvalue())
    struct.pack_into("<I", data, 4, 0xFFFFFFFF)
    struct.pack_into("<I", data, 40, 0xFFFFFFFF)
    path = tmp_path / "streamed.wav"
    path.write_bytes(data)
    before = inspect_wav(path)
    repaired = inspect_wav(path, repair=True)
    after = inspect_wav(path)
    assert before["broken"] and repaired["broken"] and not after["broken"]
    assert before["pcm_sha256"] == after["pcm_sha256"]
    assert after["duration_seconds"] == 1
    assert inspect_wav(path, repair=True) == after
