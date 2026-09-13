"""Bounded public ZIP indexing/member reads with exact-range integrity checks."""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import httpx

MAX_RANGE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class RemoteZipMember:
    """Central-directory metadata needed for an independently verified member read."""

    filename: str
    file_size: int
    compressed_size: int
    crc32: int
    header_offset: int
    compress_type: int


class BoundedRangeReader(io.RawIOBase):
    """Seekable HTTP reader that refuses full-archive or oversized metadata reads."""

    def __init__(
        self,
        client: httpx.Client,
        url: str,
        size: int,
        *,
        maximum_transferred_bytes: int = MAX_RANGE_BYTES,
    ) -> None:
        self.client = client
        self.url = url
        self.size = size
        self.position = 0
        self.transferred = 0
        self.maximum_transferred_bytes = maximum_transferred_bytes

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if not 0 <= position <= self.size:
            raise ValueError("Out-of-archive seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        count = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        if not count:
            return b""
        if count > MAX_RANGE_BYTES or self.transferred + count > self.maximum_transferred_bytes:
            raise OSError("Metadata-only range budget exceeded")
        value = _read_range(self.client, self.url, self.position, count, self.size)
        self.transferred += count
        self.position += count
        return value


def index_remote_zip(
    client: httpx.Client,
    url: str,
    archive_bytes: int,
    *,
    maximum_metadata_bytes: int = MAX_RANGE_BYTES,
) -> tuple[list[RemoteZipMember], int]:
    """Read a remote ZIP central directory without downloading the archive body."""
    if archive_bytes <= 0:
        raise ValueError("archive_bytes must be positive")
    reader = BoundedRangeReader(
        client,
        url,
        archive_bytes,
        maximum_transferred_bytes=maximum_metadata_bytes,
    )
    with zipfile.ZipFile(reader) as archive:
        members = [
            RemoteZipMember(
                filename=member.filename,
                file_size=member.file_size,
                compressed_size=member.compress_size,
                crc32=member.CRC,
                header_offset=member.header_offset,
                compress_type=member.compress_type,
            )
            for member in archive.infolist()
            if not member.is_dir()
        ]
    return members, reader.transferred


def remote_zip_member_dict(member: RemoteZipMember) -> dict[str, object]:
    """Return stable JSON-serializable central-directory metadata."""
    return asdict(member)


def _read_range(client: httpx.Client, url: str, start: int, count: int, total: int) -> bytes:
    if not 0 <= start < start + count <= total or count > MAX_RANGE_BYTES:
        raise OSError("Invalid or oversized archive-member range")
    end = start + count - 1
    with client.stream("GET", url, headers={"Range": f"bytes={start}-{end}"}) as response:
        response.raise_for_status()
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}"
        ):
            raise OSError("Archive server did not honor exact range; full download refused")
        chunks: list[bytes] = []
        received = 0
        for chunk in response.iter_bytes(1024 * 1024):
            received += len(chunk)
            if received > count:
                raise OSError("Archive range exceeded declared length")
            chunks.append(chunk)
    if received != count:
        raise OSError("Truncated archive range")
    return b"".join(chunks)


def read_zip_member(
    client: httpx.Client, metadata: Mapping[str, object], expected_size: int
) -> bytes:
    if not 0 < expected_size <= MAX_RANGE_BYTES:
        raise OSError("Archive member exceeds bounded extraction limit")
    url = str(metadata["archive_content_url"])
    total = int(str(metadata["archive_bytes"]))
    offset = int(str(metadata["archive_header_offset"]))
    compressed_size = int(str(metadata["archive_compressed_size"]))
    header = _read_range(client, url, offset, 30, total)
    (
        signature,
        _version,
        flags,
        method,
        _time,
        _date,
        _crc,
        _compressed,
        _size,
        name_length,
        extra_length,
    ) = struct.unpack("<4s5H3I2H", header)
    if signature != b"PK\x03\x04" or flags & 1 or method not in {0, 8}:
        raise OSError("Unsupported or encrypted ZIP local header")
    if method != int(str(metadata["archive_compress_type"])):
        raise OSError("ZIP directory/local-header compression mismatch")
    data = _read_range(
        client, url, offset + 30 + name_length + extra_length, compressed_size, total
    )
    if method == 8:
        decoder = zlib.decompressobj(-15)
        try:
            value = decoder.decompress(data, expected_size + 1)
        except zlib.error as exc:
            raise OSError("Invalid ZIP deflate stream") from exc
        if not decoder.eof or decoder.unused_data or len(value) != expected_size:
            raise OSError("ZIP member size/deflate termination mismatch")
    else:
        value = data
    if len(value) != expected_size or zlib.crc32(value) != int(
        str(metadata["archive_member_crc32"])
    ):
        raise OSError("ZIP member size/CRC32 mismatch")
    return value
