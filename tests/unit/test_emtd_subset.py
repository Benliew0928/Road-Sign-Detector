from __future__ import annotations

import csv
import struct
import zlib
from pathlib import Path

from roadsign_assist.datasets.emtd_subset import (
    ArchiveEntry,
    extract_member_bytes,
    select_filenames,
)


def test_select_filenames_is_balanced_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "gt.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "Class ID", "xmin", "ymin", "xmax", "ymax"],
        )
        writer.writeheader()
        for class_id in (1, 2):
            for index in range(4):
                writer.writerow(
                    {
                        "filename": (
                            f"{class_id}-{index}.JPG" if index % 2 else f"{class_id}-{index}.jpg"
                        ),
                        "Class ID": class_id,
                        "xmin": 1,
                        "ymin": 1,
                        "xmax": 2,
                        "ymax": 2,
                    }
                )

    selected_a, classes_a = select_filenames(path, per_class=2, seed=2513)
    selected_b, classes_b = select_filenames(path, per_class=2, seed=2513)
    assert selected_a == selected_b
    assert classes_a == classes_b
    assert len(selected_a) == 4
    assert all(filename == filename.casefold() for filename in selected_a)
    assert {value for name in selected_a for value in classes_a[name]} == {1, 2}


def test_extract_member_bytes_supports_deflate() -> None:
    name = "EMTD/Detection/example.jpg"
    content = b"road-sign-image-bytes" * 16
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    compressed = compressor.compress(content) + compressor.flush()
    name_bytes = name.encode()
    local_header = struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50,
        20,
        0,
        8,
        0,
        0,
        0,
        len(compressed),
        len(content),
        len(name_bytes),
        0,
    )
    payload = local_header + name_bytes + compressed
    entry = ArchiveEntry(
        name=name,
        method=8,
        compressed_size=len(compressed),
        uncompressed_size=len(content),
        local_header_offset=0,
    )
    assert extract_member_bytes(payload, entry) == content
