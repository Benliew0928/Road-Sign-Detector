from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import struct
import time
import urllib.error
import urllib.request
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_URL = "https://zenodo.org/records/1217105/files/EMTD.zip?download=1"
LOCAL_FILE_SIGNATURE = 0x04034B50


@dataclass(frozen=True)
class ArchiveEntry:
    name: str
    method: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ArchiveEntry:
        return cls(
            name=str(value["name"]),
            method=int(value["method"]),
            compressed_size=int(value["comp"]),
            uncompressed_size=int(value["uncomp"]),
            local_header_offset=int(value["off"]),
        )


@dataclass(frozen=True)
class DownloadResult:
    filename: str
    path: Path
    compressed_size: int
    uncompressed_size: int
    sha256: str
    status: str


def _stable_key(class_id: int, filename: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{class_id}:{filename}".encode()).digest()


def select_filenames(
    ground_truth_path: Path,
    *,
    per_class: int,
    seed: int,
) -> tuple[set[str], dict[str, set[int]]]:
    if per_class < 1:
        raise ValueError("per_class must be at least one")

    by_class: dict[int, set[str]] = defaultdict(set)
    image_classes: dict[str, set[int]] = defaultdict(set)
    with ground_truth_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "Class ID", "xmin", "ymin", "xmax", "ymax"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Ground truth must contain {sorted(required)}")
        for row in reader:
            filename = row["filename"].strip().casefold()
            class_id = int(row["Class ID"])
            if not filename:
                continue
            by_class[class_id].add(filename)
            image_classes[filename].add(class_id)

    selected: set[str] = set()
    for class_id, filenames in sorted(by_class.items()):
        ordered = sorted(filenames, key=lambda value: _stable_key(class_id, value, seed))
        selected.update(ordered[:per_class])
    return selected, image_classes


def load_archive_entries(index_path: Path) -> dict[str, ArchiveEntry]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = [ArchiveEntry.from_dict(value) for value in payload["entries"]]
    return {
        PurePosixPath(entry.name).name.casefold(): entry
        for entry in entries
        if entry.name.startswith("EMTD/Detection/") and not entry.name.endswith("/")
    }


def _request_member(entry: ArchiveEntry, *, timeout: float) -> bytes:
    filename_length = len(entry.name.encode("utf-8"))
    overhead = 1024
    start = entry.local_header_offset
    end = start + 30 + filename_length + overhead + entry.compressed_size - 1
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "RoadSignAssist/0.1 academic-dataset-client",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(f"bytes {start}-"):
            raise RuntimeError(
                f"Server did not honor byte range for {entry.name}: "
                f"status={response.status}, Content-Range={content_range!r}"
            )
        maximum = 30 + filename_length + overhead + entry.compressed_size
        return response.read(maximum)


def extract_member_bytes(payload: bytes, entry: ArchiveEntry) -> bytes:
    if len(payload) < 30:
        raise ValueError(f"Truncated local header for {entry.name}")
    (signature,) = struct.unpack_from("<I", payload)
    if signature != LOCAL_FILE_SIGNATURE:
        raise ValueError(f"Invalid ZIP local header for {entry.name}")
    filename_length, extra_length = struct.unpack_from("<HH", payload, 26)
    data_start = 30 + filename_length + extra_length
    data_end = data_start + entry.compressed_size
    compressed = payload[data_start:data_end]
    if len(compressed) != entry.compressed_size:
        raise ValueError(f"Truncated compressed data for {entry.name}")
    if entry.method == 0:
        content = compressed
    elif entry.method == 8:
        content = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"Unsupported ZIP method {entry.method} for {entry.name}")
    if len(content) != entry.uncompressed_size:
        raise ValueError(
            f"Size mismatch for {entry.name}: "
            f"expected {entry.uncompressed_size}, got {len(content)}"
        )
    return content


def download_entry(
    entry: ArchiveEntry,
    output_root: Path,
    *,
    retries: int,
    timeout: float,
) -> DownloadResult:
    destination = output_root / PurePosixPath(entry.name).name
    if destination.exists() and destination.stat().st_size == entry.uncompressed_size:
        content = destination.read_bytes()
        return DownloadResult(
            filename=destination.name,
            path=destination,
            compressed_size=entry.compressed_size,
            uncompressed_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            status="cached",
        )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            payload = _request_member(entry, timeout=timeout)
            content = extract_member_bytes(payload, entry)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return DownloadResult(
                filename=destination.name,
                path=destination,
                compressed_size=entry.compressed_size,
                uncompressed_size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                status="downloaded",
            )
        except urllib.error.HTTPError as exc:
            last_error = exc
            if attempt < retries:
                retry_after = exc.headers.get("Retry-After")
                try:
                    server_delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    server_delay = 0.0
                exponential_delay = min(120.0, 5.0 * (2**attempt))
                time.sleep(max(server_delay, exponential_delay) + random.uniform(0.0, 2.0))
        except (OSError, RuntimeError, ValueError, zlib.error) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(60.0, 2.0**attempt) + random.uniform(0.0, 1.0))
    raise RuntimeError(f"Unable to download {entry.name}: {last_error}") from last_error


def write_manifest(
    path: Path,
    results: list[DownloadResult],
    image_classes: dict[str, set[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "relative_path",
                "class_ids",
                "compressed_size",
                "uncompressed_size",
                "sha256",
                "status",
            ],
        )
        writer.writeheader()
        for result in sorted(results, key=lambda value: value.filename.casefold()):
            writer.writerow(
                {
                    "filename": result.filename,
                    "relative_path": result.path.as_posix(),
                    "class_ids": " ".join(
                        str(value) for value in sorted(image_classes[result.filename.casefold()])
                    ),
                    "compressed_size": result.compressed_size,
                    "uncompressed_size": result.uncompressed_size,
                    "sha256": result.sha256,
                    "status": result.status,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a balanced EMTD subset directly from the remote ZIP archive."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/raw/emtd/metadata/GT.csv"),
    )
    parser.add_argument(
        "--archive-index",
        type=Path,
        default=Path("data/raw/emtd/metadata/archive_index.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/emtd/images"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/emtd/metadata/subset_manifest.csv"),
    )
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2513)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=240.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selected, image_classes = select_filenames(
        args.ground_truth,
        per_class=args.per_class,
        seed=args.seed,
    )
    archive_entries = load_archive_entries(args.archive_index)
    missing = sorted(selected - archive_entries.keys())
    if missing:
        raise ValueError(f"Archive index is missing {len(missing)} selected images")

    entries = [archive_entries[name] for name in sorted(selected)]
    compressed_gib = sum(entry.compressed_size for entry in entries) / 1024**3
    represented_classes = {
        class_id for filename in selected for class_id in image_classes[filename]
    }
    print(
        f"Selected {len(entries)} images across {len(represented_classes)} "
        f"classes ({compressed_gib:.2f} GiB compressed)."
    )

    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                download_entry,
                entry,
                args.output,
                retries=args.retries,
                timeout=args.timeout,
            ): entry
            for entry in entries
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            entry = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[{completed}/{len(entries)}] FAILED {entry.name}: {exc}")
                continue
            results.append(result)
            print(
                f"[{completed}/{len(entries)}] {result.status:10} "
                f"{result.filename} ({result.uncompressed_size / 1024**2:.1f} MiB)"
            )

    write_manifest(args.manifest, results, image_classes)
    failed = len(entries) - len(results)
    print(f"Wrote {len(results)} records to {args.manifest}; failures={failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
