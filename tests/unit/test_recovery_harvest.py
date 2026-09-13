from __future__ import annotations

import hashlib
import json
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import httpx
import pytest

from roadsign_assist.datasets.recovery_harvest import (
    HarvestManifest,
    HarvestObject,
    _download_one,
    _safe_filename,
    download_harvest_batch,
    harvest_kartaview,
    harvest_wikimedia,
    harvest_zenodo,
    save_harvest_manifest,
)


def test_safe_filename_bounds_windows_download_paths_and_preserves_extension() -> None:
    filename = "very long road sign name " * 30 + ".jpg"

    safe = _safe_filename(filename)

    assert len(safe) <= 80
    assert safe.endswith(".jpg")
    assert " " not in safe


def test_zenodo_harvest_preserves_size_checksum_and_download_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/records/123"
        return httpx.Response(
            200,
            json={
                "id": 123,
                "links": {"self_html": "https://zenodo.org/records/123"},
                "metadata": {
                    "title": "Dataset",
                    "license": {"id": "cc-by-4.0", "url": "https://license"},
                    "creators": [{"name": "Author"}],
                },
                "files": [
                    {
                        "key": "dataset.zip",
                        "size": 42,
                        "checksum": "md5:" + "a" * 32,
                        "links": {"self": "https://zenodo.org/files/dataset.zip"},
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        manifest = harvest_zenodo(
            123, source_id="example_source", batch_id="batch-1", client=client
        )

    assert manifest.state == "success"
    assert len(manifest.objects) == 1
    assert manifest.objects[0].expected_bytes == 42
    assert manifest.objects[0].expected_checksum_algorithm == "md5"
    assert manifest.objects[0].expected_checksum == "a" * 32
    assert manifest.objects[0].download_url.endswith("dataset.zip")


def test_kartaview_and_wikimedia_harvest_keep_provenance() -> None:
    def karta_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": {
                    "data": [
                        {
                            "id": "photo-1",
                            "fileurlProc": "https://cdn.example/photo.jpg",
                            "name": "photo.jpg",
                            "width": "1920",
                            "height": "1080",
                            "lat": "3.1",
                            "lng": "101.6",
                            "shotDate": "2026-01-01",
                            "sequenceIndex": "3",
                            "sequence": {
                                "id": "sequence-1",
                                "userId": "contributor-1",
                                "countryCode": "MY",
                                "deviceName": "Camera",
                                "platformName": "Android",
                            },
                        }
                    ]
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(karta_handler)) as client:
        karta = harvest_kartaview(
            batch_id="karta-1",
            points=(("kl", 3.1, 101.6),),
            request_delay_seconds=0,
            client=client,
        )
    assert karta.objects[0].sequence_id == "sequence-1"
    assert karta.objects[0].contributor_id == "contributor-1"
    assert karta.objects[0].coarse_location == "3.1,101.6"

    def wiki_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": {
                        "7": {
                            "pageid": 7,
                            "title": "File:Road.jpg",
                            "imageinfo": [
                                {
                                    "url": "https://upload.wikimedia.org/road.jpg",
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Road.jpg",
                                    "size": 99,
                                    "width": 640,
                                    "height": 480,
                                    "mime": "image/jpeg",
                                    "user": "Photographer",
                                    "extmetadata": {
                                        "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                        "LicenseUrl": {"value": "https://license"},
                                        "Artist": {"value": "Photographer"},
                                    },
                                }
                            ],
                        }
                    }
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(wiki_handler)) as client:
        wiki = harvest_wikimedia(
            batch_id="wiki-1",
            queries=("Malaysia road",),
            request_delay_seconds=0,
            client=client,
        )
    assert wiki.objects[0].licence_name == "CC BY-SA 4.0"
    assert wiki.objects[0].attribution == "Photographer"
    assert wiki.objects[0].expected_bytes == 99


def test_zip_member_crc_is_verified_before_publication(tmp_path: Path) -> None:
    payload = b"archive-member"
    item = HarvestObject(
        object_id="one",
        source_id="source",
        batch_id="batch",
        group_id="archive",
        source_url="https://example.test/archive",
        download_url="https://example.test/member",
        filename="member.bin",
        expected_bytes=len(payload),
        metadata={"archive_member_crc32": zlib.crc32(payload)},
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
    ) as client:
        _download_one(client, item, tmp_path / "good.bin", retries=0, request_delay_seconds=0)
        bad = item.model_copy(update={"metadata": {"archive_member_crc32": 0}})
        with pytest.raises(RuntimeError, match="CRC32 mismatch"):
            _download_one(client, bad, tmp_path / "bad.bin", retries=0, request_delay_seconds=0)
    assert not (tmp_path / "bad.bin").exists()
    assert (tmp_path / "good.bin").read_bytes() == payload


def test_cached_download_is_revalidated_and_preserved_if_corrupt(tmp_path: Path) -> None:
    destination = tmp_path / "cached.bin"
    destination.write_bytes(b"wrong")
    item = HarvestObject(
        object_id="one",
        source_id="source",
        batch_id="batch",
        group_id="archive",
        source_url="https://example.test/archive",
        download_url="https://example.test/member",
        filename="member.bin",
        expected_bytes=5,
        metadata={"archive_member_crc32": zlib.crc32(b"right")},
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("Corrupt existing files must not be silently overwritten")

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(RuntimeError, match="Existing download failed integrity"),
    ):
        _download_one(client, item, destination, retries=0, request_delay_seconds=0)
    assert destination.read_bytes() == b"wrong"


def test_rate_limit_stops_subsequent_requests(tmp_path: Path) -> None:
    manifest = HarvestManifest(
        adapter="direct_archive",
        source_id="source",
        batch_id="harvest",
        harvested_at=datetime(2026, 9, 5, tzinfo=UTC),
        source_url="https://example.test",
        state="success",
        objects=[
            HarvestObject(
                object_id=str(i),
                source_id="source",
                batch_id="harvest",
                group_id="group",
                source_url="https://example.test",
                download_url=f"https://example.test/{i}",
                filename=f"{i}.bin",
                expected_bytes=5,
            )
            for i in range(4)
        ],
    )
    saved = save_harvest_manifest(manifest, acquisition_root=tmp_path)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(429, headers={"Retry-After": "60"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_harvest_batch(
            str(saved["manifest"]),
            download_batch_id="download",
            acquisition_root=tmp_path,
            minimum_free_gib=0,
            maximum_acquisition_gib=1,
            download_workers=1,
            retries=3,
            request_delay_seconds=0,
            client=client,
        )
    assert len(calls) == 1
    assert result["rate_limited"] is True
    assert result["completed_objects"] == 0


def test_harvest_snapshot_and_download_are_resumable_and_bounded(tmp_path: Path) -> None:
    payload = b"recovery-object"
    sha = hashlib.sha256(payload).hexdigest()
    manifest = HarvestManifest(
        adapter="direct_archive",
        source_id="source",
        batch_id="harvest-1",
        harvested_at=datetime(2026, 9, 4, tzinfo=UTC),
        source_url="https://example.test/object.bin",
        state="success",
        objects=[
            HarvestObject(
                object_id="direct:object",
                source_id="source",
                batch_id="harvest-1",
                group_id="group",
                source_url="https://example.test/object.bin",
                download_url="https://example.test/object.bin",
                filename="object.bin",
                expected_bytes=len(payload),
                expected_checksum_algorithm="sha256",
                expected_checksum=sha,
            )
        ],
    )
    saved = save_harvest_manifest(manifest, acquisition_root=tmp_path)
    manifest_path = Path(str(saved["manifest"]))
    resumed = save_harvest_manifest(manifest, acquisition_root=tmp_path, resume=True)
    assert resumed["source_events_appended"] == 0
    assert len((tmp_path / "manifests/source_events.jsonl").read_text().splitlines()) == 1

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(payload))})
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        dry = download_harvest_batch(
            manifest_path,
            download_batch_id="download-dry",
            acquisition_root=tmp_path,
            max_bytes=len(payload) - 1,
            minimum_free_gib=0,
            maximum_acquisition_gib=1,
            dry_run=True,
            client=client,
        )
        assert dry["selected_objects"] == 0
        skipped = cast(list[dict[str, object]], dry["skipped"])
        assert skipped[0]["reason"] == "object_exceeds_batch_limit"

        result = download_harvest_batch(
            manifest_path,
            download_batch_id="download-1",
            acquisition_root=tmp_path,
            max_bytes=len(payload),
            minimum_free_gib=0,
            maximum_acquisition_gib=1,
            request_delay_seconds=0,
            client=client,
        )
    assert result["status"] == "success"
    assert result["completed_bytes"] == len(payload)
    completed = cast(list[dict[str, object]], result["completed"])
    downloaded = Path(str(completed[0]["path"]))
    assert downloaded.read_bytes() == payload
    event = json.loads((tmp_path / "manifests/object_events.jsonl").read_text().splitlines()[0])
    assert event["collection_state"] == "quarantined"
    assert event["permitted_use"] == "none_pending_review"
