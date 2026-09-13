from __future__ import annotations

import io
import zipfile

import httpx
import pytest

from roadsign_assist.datasets.recovery_archive_ranges import index_remote_zip, read_zip_member


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_exact_ranges_decode_and_verify_zip_member(method: int) -> None:
    data = b"original image bytes" * 1024
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, "w", compression=method) as archive:
        archive.writestr("images/sample.jpg", data)
    payload = handle.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = archive.infolist()[0]
    metadata = dict(
        archive_content_url="https://example.test/archive.zip",
        archive_bytes=len(payload),
        archive_header_offset=member.header_offset,
        archive_compressed_size=member.compress_size,
        archive_compress_type=member.compress_type,
        archive_member_crc32=member.CRC,
    )
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        requested.append((start, end))
        return httpx.Response(
            206,
            content=payload[start : end + 1],
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert read_zip_member(client, metadata, len(data)) == data
        with pytest.raises(OSError, match="CRC32"):
            read_zip_member(client, metadata | {"archive_member_crc32": 0}, len(data))
    assert requested[0] == (0, 29)
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"not a range")
            )
        ) as client,
        pytest.raises(OSError, match="full download refused"),
    ):
        read_zip_member(client, metadata, len(data))


def test_remote_zip_index_reads_only_bounded_exact_ranges() -> None:
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("00000.ppm", b"P6\n1 1\n255\n\x00\x00\x00")
        archive.writestr("gt.txt", b"00000.ppm;0;0;1;1;1\n")
    payload = handle.getvalue()
    transferred = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal transferred
        assert request.headers.get("Range")
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        value = payload[start : end + 1]
        transferred += len(value)
        return httpx.Response(
            206,
            content=value,
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        members, reported = index_remote_zip(client, "https://example.test/data.zip", len(payload))

    assert [member.filename for member in members] == ["00000.ppm", "gt.txt"]
    assert members[0].file_size == 14
    assert reported == transferred
    assert reported < len(payload)
