"""Source harvesting and bounded resumable downloads for recovery Phase B1."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import threading
import time
import zlib
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from roadsign_assist.datasets.recovery_acquisition import (
    DEFAULT_ACQUISITION_ROOT,
    PLAN_ID,
    SCHEMA_VERSION,
    ObjectLedgerEntry,
    SourceLedgerEntry,
    check_disk_budget,
)
from roadsign_assist.datasets.recovery_archive_ranges import read_zip_member
from roadsign_assist.paths import project_path

ADAPTER_VERSION = "1.0"
DEFAULT_MAX_BATCH_BYTES = 2_000_000_000
DEFAULT_MAX_BATCH_OBJECTS = 10_000
USER_AGENT = (
    "RoadSignAssistRecovery/1.0 (https://github.com/benli1234/MiniProject; academic dataset audit)"
)

MALAYSIA_KARTAVIEW_POINTS: tuple[tuple[str, float, float], ...] = (
    ("kuala_lumpur", 3.1390, 101.6869),
    ("petaling_jaya", 3.1073, 101.6067),
    ("shah_alam", 3.0738, 101.5183),
    ("putrajaya", 2.9264, 101.6964),
    ("johor_bahru", 1.4927, 103.7414),
    ("george_town", 5.4141, 100.3288),
    ("ipoh", 4.5975, 101.0901),
    ("melaka", 2.1896, 102.2501),
    ("kota_bharu", 6.1254, 102.2381),
    ("kuala_terengganu", 5.3296, 103.1370),
    ("kuantan", 3.8077, 103.3260),
    ("kota_kinabalu", 5.9804, 116.0735),
    ("kuching", 1.5533, 110.3592),
)

WIKIMEDIA_QUERIES: tuple[str, ...] = (
    "Malaysia road traffic sign",
    "Malaysia street road",
    "Malaysian expressway traffic",
    "Malaysia road at night",
    "Malaysia road rain",
)


class HarvestObject(BaseModel):
    """Download candidate produced by a source adapter."""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    source_id: str
    batch_id: str
    group_id: str
    sequence_id: str = ""
    source_url: str
    download_url: str
    filename: str
    expected_bytes: int | None = Field(default=None, ge=0)
    expected_checksum_algorithm: Literal["", "md5", "sha256"] = ""
    expected_checksum: str = ""
    mime_type: str = "application/octet-stream"
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    source_timestamp: str = ""
    coarse_location: str = ""
    contributor_id: str = ""
    camera_id: str = ""
    licence_name: str = ""
    licence_url: str = ""
    attribution: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class HarvestManifest(BaseModel):
    """Immutable metadata snapshot emitted by a source adapter."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    plan_id: str = PLAN_ID
    adapter: str
    adapter_version: str = ADAPTER_VERSION
    source_id: str
    batch_id: str
    harvested_at: datetime
    source_url: str
    query: str = ""
    geographic_filter: str = ""
    licence_status: str = "unreviewed"
    permitted_use: str = "none_pending_review"
    redistribution_review_required: bool = True
    state: str
    state_reason: str = ""
    objects: list[HarvestObject]


def _now() -> datetime:
    return datetime.now(UTC)


def _stable_id(*values: object) -> str:
    payload = ":".join(str(value) for value in values)
    return hashlib.sha256(payload.encode()).hexdigest()


def _as_mapping(value: object) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected object, got {type(value).__name__}")
    return cast(Mapping[object, object], value)


def _mapping_or_empty(value: object) -> Mapping[object, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[object, object], value)
    return {}


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"Expected list, got {type(value).__name__}")
    return cast(list[object], value)


def _string(value: object) -> str:
    return "" if value is None else str(value)


def _integer(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    return None


def _required_int(value: object) -> int:
    parsed = _integer(value)
    if parsed is None:
        raise TypeError(f"Expected integer-compatible value, got {type(value).__name__}")
    return parsed


def _client(
    client: httpx.Client | None = None, *, timeout_seconds: float = 60.0
) -> tuple[httpx.Client, bool]:
    if client is not None:
        return client, False
    return (
        httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_seconds, connect=min(20.0, timeout_seconds)),
        ),
        True,
    )


def _accepted_source_ids(
    registry_path: str | Path = "data/manifests/dataset_sources.json",
) -> set[str]:
    path = project_path(registry_path)
    if not path.is_file():
        return set()
    raw_object: object = json.loads(path.read_text(encoding="utf-8"))
    raw = _as_mapping(raw_object)
    sources = _as_list(raw.get("sources", []))
    accepted: set[str] = set()
    for source_object in sources:
        source = _as_mapping(source_object)
        if source.get("decision") == "accepted" and source.get("source_id"):
            accepted.add(_string(source["source_id"]))
    return accepted


def harvest_zenodo(
    record_id: int,
    *,
    source_id: str,
    batch_id: str,
    client: httpx.Client | None = None,
) -> HarvestManifest:
    """Harvest public record/file metadata from Zenodo's documented REST API."""
    http, owned = _client(client)
    try:
        response = http.get(f"https://zenodo.org/api/records/{record_id}")
        response.raise_for_status()
        raw = _as_mapping(cast(object, response.json()))
    finally:
        if owned:
            http.close()
    metadata = _as_mapping(raw.get("metadata", {}))
    licence_object = metadata.get("license", {})
    licence = _mapping_or_empty(licence_object)
    licence_name = _string(licence.get("id") or licence.get("title"))
    source_url = _string(_as_mapping(raw.get("links", {})).get("self_html"))
    if not source_url:
        source_url = f"https://zenodo.org/records/{record_id}"
    objects: list[HarvestObject] = []
    for file_object in _as_list(raw.get("files", [])):
        file = _as_mapping(file_object)
        filename = _string(file.get("key"))
        links = _as_mapping(file.get("links", {}))
        checksum_value = _string(file.get("checksum"))
        algorithm, _, checksum = checksum_value.partition(":")
        algorithm = algorithm.lower() if algorithm.lower() in {"md5", "sha256"} else ""
        objects.append(
            HarvestObject(
                object_id=f"zenodo:{record_id}:{_stable_id(filename, checksum_value)[:24]}",
                source_id=source_id,
                batch_id=batch_id,
                group_id=f"zenodo-record:{record_id}",
                source_url=source_url,
                download_url=_string(links.get("self") or links.get("content")),
                filename=filename,
                expected_bytes=_integer(file.get("size")),
                expected_checksum_algorithm=cast(Literal["", "md5", "sha256"], algorithm),
                expected_checksum=checksum,
                mime_type="application/zip"
                if filename.lower().endswith(".zip")
                else "application/octet-stream",
                licence_name=licence_name,
                licence_url=_string(licence.get("url")),
                attribution=_string(metadata.get("creators")),
                metadata={
                    "record_id": record_id,
                    "record_title": _string(metadata.get("title")),
                    "publication_date": _string(metadata.get("publication_date")),
                    "doi": _string(metadata.get("doi")),
                },
            )
        )
    accepted = source_id in _accepted_source_ids()
    return HarvestManifest(
        adapter="zenodo",
        source_id=source_id,
        batch_id=batch_id,
        harvested_at=_now(),
        source_url=source_url,
        query=f"record_id={record_id}",
        licence_status="accepted_registry" if accepted else "unreviewed",
        permitted_use="registry_policy" if accepted else "none_pending_review",
        redistribution_review_required=not accepted,
        state="success" if objects else "exhausted",
        objects=objects,
    )


def _kartaview_photo(
    photo: Mapping[object, object], *, batch_id: str, query_name: str
) -> HarvestObject | None:
    photo_id = _string(photo.get("id"))
    url = _string(photo.get("fileurlProc") or photo.get("imageProcUrl"))
    if not photo_id or not url:
        return None
    sequence_object = photo.get("sequence", {})
    sequence = _mapping_or_empty(sequence_object)
    sequence_id = _string(sequence.get("id"))
    contributor = _string(sequence.get("userId"))
    camera = ":".join(
        value
        for value in (
            _string(sequence.get("platformName")),
            _string(sequence.get("deviceName")),
        )
        if value
    )
    lat, lng = _string(photo.get("lat")), _string(photo.get("lng"))
    filename = _string(photo.get("name")) or Path(urlparse(url).path).name
    return HarvestObject(
        object_id=f"kartaview:{photo_id}",
        source_id="kartaview_malaysia",
        batch_id=batch_id,
        group_id=f"kartaview-sequence:{sequence_id or photo_id}",
        sequence_id=sequence_id,
        source_url=f"https://kartaview.org/details/{sequence_id}/{photo_id}",
        download_url=url,
        filename=filename,
        mime_type="image/jpeg",
        width=_integer(photo.get("width")),
        height=_integer(photo.get("height")),
        source_timestamp=_string(photo.get("shotDate")),
        coarse_location=f"{lat},{lng}" if lat and lng else query_name,
        contributor_id=contributor,
        camera_id=camera,
        attribution="KartaView contributor; preserve photo, sequence, and contributor IDs",
        metadata={
            "query_location": query_name,
            "latitude": lat,
            "longitude": lng,
            "heading": _string(photo.get("heading")),
            "sequence_index": _string(photo.get("sequenceIndex")),
            "country_code": _string(sequence.get("countryCode")),
            "state_code": _string(sequence.get("stateCode")),
            "address": _string(sequence.get("address")),
            "sequence_photo_count": _string(sequence.get("countActivePhotos")),
            "upload_source": _string(sequence.get("uploadSource")),
            "quality_level": _string(photo.get("qualityLevel")),
            "processing_result": _string(photo.get("autoImgProcessingResult")),
        },
    )


def harvest_kartaview(
    *,
    batch_id: str,
    points: Sequence[tuple[str, float, float]] = MALAYSIA_KARTAVIEW_POINTS,
    radius: int = 100,
    limit_per_point: int = 100,
    request_delay_seconds: float = 0.25,
    client: httpx.Client | None = None,
) -> HarvestManifest:
    """Harvest Malaysian photo metadata through KartaView's public nearby-photo API."""
    http, owned = _client(client)
    objects: dict[str, HarvestObject] = {}
    failures: list[str] = []
    try:
        for index, (name, latitude, longitude) in enumerate(points):
            try:
                response = http.get(
                    "https://api.openstreetcam.org/2.0/photo/",
                    params={
                        "lat": latitude,
                        "lng": longitude,
                        "zoomLevel": 12,
                        "join": "sequence",
                        "orderBy": "id",
                        "orderDirection": "desc",
                        "radius": radius,
                        "limit": limit_per_point,
                    },
                )
                response.raise_for_status()
                raw = _as_mapping(cast(object, response.json()))
                result_object = raw.get("result")
                if not isinstance(result_object, Mapping):
                    continue
                result = cast(Mapping[object, object], result_object)
                data_object = result.get("data", [])
                if not isinstance(data_object, list):
                    continue
                for photo_object in cast(list[object], data_object):
                    if not isinstance(photo_object, Mapping):
                        continue
                    photo = cast(Mapping[object, object], photo_object)
                    candidate = _kartaview_photo(photo, batch_id=batch_id, query_name=name)
                    if candidate is not None and candidate.metadata.get("country_code") in {
                        "",
                        "MY",
                    }:
                        objects[candidate.object_id] = candidate
            except (httpx.HTTPError, ValueError) as exc:
                failures.append(f"{name}:{type(exc).__name__}:{exc}")
            if index + 1 < len(points) and request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
    finally:
        if owned:
            http.close()
    ordered = sorted(
        objects.values(),
        key=lambda item: (
            item.sequence_id,
            _integer(item.metadata.get("sequence_index")) or 0,
            item.object_id,
        ),
    )
    return HarvestManifest(
        adapter="kartaview",
        source_id="kartaview_malaysia",
        batch_id=batch_id,
        harvested_at=_now(),
        source_url="https://kartaview.org/landing/open-imagery",
        query="nearby photos around predeclared Malaysian city centres",
        geographic_filter="Malaysia; " + ",".join(name for name, _, _ in points),
        licence_status="unreviewed",
        permitted_use="none_pending_review",
        redistribution_review_required=True,
        state="success" if ordered else "blocked" if failures else "exhausted",
        state_reason=";".join(failures),
        objects=ordered,
    )


def _metadata_value(metadata: Mapping[object, object], key: str) -> str:
    raw = metadata.get(key, "")
    if isinstance(raw, Mapping):
        return _string(cast(Mapping[object, object], raw).get("value"))
    return _string(raw)


def harvest_wikimedia(
    *,
    batch_id: str,
    queries: Sequence[str] = WIKIMEDIA_QUERIES,
    limit_per_query: int = 50,
    request_delay_seconds: float = 0.25,
    client: httpx.Client | None = None,
) -> HarvestManifest:
    """Harvest Commons file metadata, including per-file author and licence tags."""
    http, owned = _client(client)
    objects: dict[str, HarvestObject] = {}
    failures: list[str] = []
    try:
        for index, query in enumerate(queries):
            try:
                response = http.get(
                    "https://commons.wikimedia.org/w/api.php",
                    params={
                        "action": "query",
                        "format": "json",
                        "generator": "search",
                        "gsrsearch": query,
                        "gsrnamespace": 6,
                        "gsrlimit": min(limit_per_query, 50),
                        "prop": "imageinfo",
                        "iiprop": "url|size|mime|timestamp|user|extmetadata",
                        "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|AttributionRequired|UsageTerms",
                    },
                )
                response.raise_for_status()
                raw = _as_mapping(cast(object, response.json()))
                query_object = raw.get("query", {})
                if not isinstance(query_object, Mapping):
                    continue
                pages_object = cast(Mapping[object, object], query_object).get("pages", {})
                if not isinstance(pages_object, Mapping):
                    continue
                for page_object in cast(Mapping[object, object], pages_object).values():
                    if not isinstance(page_object, Mapping):
                        continue
                    page = cast(Mapping[object, object], page_object)
                    info_objects = page.get("imageinfo", [])
                    if not isinstance(info_objects, list) or not info_objects:
                        continue
                    info = _as_mapping(cast(list[object], info_objects)[0])
                    metadata_object = info.get("extmetadata", {})
                    metadata = _mapping_or_empty(metadata_object)
                    url = _string(info.get("url"))
                    title = _string(page.get("title"))
                    page_id = _string(page.get("pageid"))
                    filename = title.removeprefix("File:") or Path(urlparse(url).path).name
                    if not url or not filename:
                        continue
                    candidate = HarvestObject(
                        object_id=f"wikimedia:{page_id or _stable_id(title)[:24]}",
                        source_id="wikimedia_commons_road_media",
                        batch_id=batch_id,
                        group_id=f"wikimedia-file:{page_id or _stable_id(title)[:24]}",
                        source_url=_string(info.get("descriptionurl")),
                        download_url=url,
                        filename=filename,
                        expected_bytes=_integer(info.get("size")),
                        mime_type=_string(info.get("mime")) or "application/octet-stream",
                        width=_integer(info.get("width")),
                        height=_integer(info.get("height")),
                        source_timestamp=_string(info.get("timestamp")),
                        contributor_id=_string(info.get("user")),
                        licence_name=_metadata_value(metadata, "LicenseShortName"),
                        licence_url=_metadata_value(metadata, "LicenseUrl"),
                        attribution=_metadata_value(metadata, "Artist")
                        or _metadata_value(metadata, "Credit"),
                        metadata={
                            "query": query,
                            "title": title,
                            "usage_terms": _metadata_value(metadata, "UsageTerms"),
                            "attribution_required": _metadata_value(
                                metadata, "AttributionRequired"
                            ),
                        },
                    )
                    objects[candidate.object_id] = candidate
            except (httpx.HTTPError, ValueError) as exc:
                failures.append(f"{query}:{type(exc).__name__}:{exc}")
            if index + 1 < len(queries) and request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
    finally:
        if owned:
            http.close()
    ordered = sorted(objects.values(), key=lambda item: item.object_id)
    return HarvestManifest(
        adapter="wikimedia",
        source_id="wikimedia_commons_road_media",
        batch_id=batch_id,
        harvested_at=_now(),
        source_url="https://commons.wikimedia.org/",
        query="; ".join(queries),
        geographic_filter="Malaysia keyword and road-context queries",
        licence_status="per_file_observed_unreviewed",
        permitted_use="none_pending_review",
        redistribution_review_required=True,
        state="success" if ordered else "blocked" if failures else "exhausted",
        state_reason=";".join(failures),
        objects=ordered,
    )


def harvest_direct(
    *,
    source_id: str,
    batch_id: str,
    url: str,
    filename: str = "",
    expected_bytes: int | None = None,
    checksum: str = "",
) -> HarvestManifest:
    """Create a quarantined manifest for one stable direct archive/object URL."""
    name = filename or Path(urlparse(url).path).name
    algorithm, _, value = checksum.partition(":")
    algorithm = algorithm.lower() if algorithm.lower() in {"md5", "sha256"} else ""
    return HarvestManifest(
        adapter="direct_archive",
        source_id=source_id,
        batch_id=batch_id,
        harvested_at=_now(),
        source_url=url,
        query="direct stable URL",
        state="success",
        objects=[
            HarvestObject(
                object_id=f"direct:{_stable_id(source_id, url)[:24]}",
                source_id=source_id,
                batch_id=batch_id,
                group_id=f"direct:{source_id}",
                source_url=url,
                download_url=url,
                filename=name,
                expected_bytes=expected_bytes,
                expected_checksum_algorithm=cast(Literal["", "md5", "sha256"], algorithm),
                expected_checksum=value,
            )
        ],
    )


def harvest_generic_manifest(
    manifest_path: str | Path,
    *,
    source_id: str,
    batch_id: str,
) -> HarvestManifest:
    """Import URL-bearing CSV or JSON rows without treating their labels as truth."""
    path = project_path(manifest_path)
    raw_rows: list[dict[str, object]] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            raw_rows = [
                {str(key): value or "" for key, value in row.items() if key is not None}
                for row in csv.DictReader(handle)
            ]
    else:
        raw_object: object = json.loads(path.read_text(encoding="utf-8"))
        values = _as_list(raw_object)
        raw_rows = [{str(key): value for key, value in _as_mapping(row).items()} for row in values]
    objects: dict[str, HarvestObject] = {}
    for index, row in enumerate(raw_rows, start=1):
        url = next(
            (
                _string(row.get(field_name))
                for field_name in (
                    "download_url",
                    "direct_image_url",
                    "source_direct_or_reference_url",
                    "url",
                )
                if row.get(field_name)
            ),
            "",
        )
        if not url.startswith(("https://", "http://")):
            continue
        candidate_id = next(
            (
                _string(row.get(name))
                for name in ("object_id", "candidate_id", "sample_id")
                if row.get(name)
            ),
            f"row-{index:06d}",
        )
        filename = _string(row.get("filename") or row.get("website_filename"))
        filename = filename or Path(urlparse(url).path).name or f"{candidate_id}.bin"
        group = (
            _string(row.get("source_group") or row.get("group_id")) or f"manifest:{candidate_id}"
        )
        objects[candidate_id] = HarvestObject(
            object_id=f"manifest:{source_id}:{_stable_id(candidate_id, url)[:24]}",
            source_id=source_id,
            batch_id=batch_id,
            group_id=group,
            sequence_id=_string(row.get("source_video_group") or row.get("sequence_id")),
            source_url=_string(row.get("source_page_url") or row.get("source_url") or url),
            download_url=url,
            filename=filename,
            expected_bytes=_integer(row.get("byte_size") or row.get("size")),
            licence_name=_string(row.get("stated_license") or row.get("licence_name")),
            attribution=_string(row.get("creator_publisher") or row.get("attribution")),
            metadata={"import_manifest": str(path), "import_row": index},
        )
    return HarvestManifest(
        adapter="generic_manifest",
        source_id=source_id,
        batch_id=batch_id,
        harvested_at=_now(),
        source_url=str(path),
        query=f"import:{path.name}",
        state="success" if objects else "exhausted",
        objects=sorted(objects.values(), key=lambda item: item.object_id),
    )


def _event_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    values: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                raw_object: object = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(raw_object, Mapping):
                event_id = cast(Mapping[object, object], raw_object).get("event_id")
                if event_id:
                    values.add(_string(event_id))
    return values


def _append_events(path: Path, events: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = _event_ids(path)
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            event_id = _string(event.get("event_id"))
            if not event_id or event_id in seen:
                continue
            handle.write(json.dumps(dict(event), sort_keys=True, default=str) + "\n")
            handle.flush()
            seen.add(event_id)
            written += 1
    return written


def _source_event(manifest: HarvestManifest) -> SourceLedgerEntry:
    first = manifest.objects[0] if manifest.objects else None
    return SourceLedgerEntry(
        event_id=_stable_id("source_harvest", manifest.source_id, manifest.batch_id),
        observed_at=manifest.harvested_at,
        source_id=manifest.source_id,
        source_name=manifest.source_id,
        publisher=manifest.adapter,
        owner="per-source metadata pending review",
        source_url=manifest.source_url,
        dataset_api_version="public interface observed 2026-09-04",
        licence_name=first.licence_name if first else "",
        licence_url=first.licence_url if first else "",
        licence_status=manifest.licence_status,
        access_method=manifest.adapter,
        adapter_version=manifest.adapter_version,
        query=manifest.query,
        geographic_filter=manifest.geographic_filter,
        batch_id=manifest.batch_id,
        attribution=first.attribution if first else "",
        redistribution_review_required=manifest.redistribution_review_required,
        state=manifest.state,
        state_reason=manifest.state_reason,
    )


def save_harvest_manifest(
    manifest: HarvestManifest,
    *,
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    """Persist an immutable harvest snapshot and append one source-ledger event."""
    total_known = sum(item.expected_bytes or 0 for item in manifest.objects)
    unknown = sum(item.expected_bytes is None for item in manifest.objects)
    summary: dict[str, object] = {
        "adapter": manifest.adapter,
        "source_id": manifest.source_id,
        "batch_id": manifest.batch_id,
        "state": manifest.state,
        "objects": len(manifest.objects),
        "known_bytes": total_known,
        "unknown_size_objects": unknown,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary
    root = project_path(acquisition_root)
    path = root / "manifests" / "harvest" / manifest.source_id / f"{manifest.batch_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite harvest manifest without --resume: {path}")
    if path.exists() and resume:
        existing = HarvestManifest.model_validate_json(path.read_text(encoding="utf-8"))
        summary["manifest"] = str(path)
        summary["resumed_existing"] = True
        manifest = existing
    else:
        temp = path.with_suffix(".json.tmp")
        temp.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
        summary["manifest"] = str(path)
        summary["resumed_existing"] = False
    ledger = root / "manifests" / "source_events.jsonl"
    written = _append_events(ledger, [_source_event(manifest).model_dump(mode="json")])
    summary["source_events_appended"] = written
    return summary


def import_local_inventory(
    inventory_path: str | Path,
    *,
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    dry_run: bool = False,
    max_objects: int = DEFAULT_MAX_BATCH_OBJECTS,
    offset: int = 0,
) -> dict[str, object]:
    """Import the Phase B0 local/DVC inventory into the common append-only object ledger."""
    if not 0 < max_objects <= DEFAULT_MAX_BATCH_OBJECTS:
        raise ValueError(f"max_objects must be within 1..{DEFAULT_MAX_BATCH_OBJECTS}")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    inventory = project_path(inventory_path)
    parquet: Any = pq
    table: Any = parquet.read_table(inventory)
    all_rows = cast(list[dict[str, object]], table.to_pylist())
    rows = all_rows[offset : offset + max_objects]
    if dry_run:
        return {
            "adapter": "local_dvc",
            "offset": offset,
            "objects": len(rows),
            "bytes": sum(_required_int(row["byte_size"]) for row in rows),
            "dry_run": True,
        }
    root = project_path(acquisition_root)
    object_ledger = root / "manifests" / "object_events.jsonl"
    events: list[Mapping[str, object]] = []
    for row in rows:
        sources_object: object = json.loads(_string(row.get("sources_json", "[]")))
        groups_object: object = json.loads(_string(row.get("groups_json", "[]")))
        sources = [str(value) for value in _as_list(sources_object)]
        groups = [str(value) for value in _as_list(groups_object)]
        entry = ObjectLedgerEntry(
            event_id=_stable_id("local_inventory", row["inventory_id"], row["sha256"]),
            observed_at=_now(),
            object_id=_string(row["object_id"]),
            source_id=sources[0] if len(sources) == 1 else "local_unknown_or_multiple",
            batch_id="phase_b0_local_inventory_20260904",
            group_id=groups[0] if len(groups) == 1 else "local_group_unknown_or_multiple",
            filename=Path(_string(row["absolute_path"])).name,
            sha256=_string(row["sha256"]),
            byte_size=_required_int(row["byte_size"]),
            mime_type=_string(row["mime_type"]),
            width=_integer(row.get("width")),
            height=_integer(row.get("height")),
            raw_path=_string(row["absolute_path"]),
            duplicate_cluster_id=_string(row.get("duplicate_cluster_id")),
            capture_domain=_string(row.get("context_type")) or "unknown",
            road_context=_string(row.get("context_type")) or "unknown",
            collection_state=_string(row["collection_state"]),
            licence_status=_string(row["licence_status"]),
            permitted_use=_string(row["permitted_use"]),
            review_status="existing_observation",
        )
        events.append(entry.model_dump(mode="json"))
    written = _append_events(object_ledger, events)
    local_manifest = HarvestManifest(
        adapter="local_dvc",
        source_id="local_dvc_inventory",
        batch_id="phase_b0_local_inventory_20260904",
        harvested_at=_now(),
        source_url=str(inventory),
        query="Phase B0 inventory import",
        licence_status="mixed_per_object",
        permitted_use="mixed_per_object",
        redistribution_review_required=True,
        state="success",
        objects=[],
    )
    source_written = _append_events(
        root / "manifests" / "source_events.jsonl",
        [_source_event(local_manifest).model_dump(mode="json")],
    )
    return {
        "adapter": "local_dvc",
        "offset": offset,
        "objects_considered": len(rows),
        "object_events_appended": written,
        "source_events_appended": source_written,
        "object_ledger": str(object_ledger),
        "dry_run": False,
    }


def _safe_filename(value: str, *, maximum_length: int = 80) -> str:
    name = Path(value).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    cleaned = cleaned or "object.bin"
    if len(cleaned) <= maximum_length:
        return cleaned
    suffix = Path(cleaned).suffix[:16]
    stem_limit = max(1, maximum_length - len(suffix))
    stem = Path(cleaned).stem[:stem_limit].rstrip("._-") or "object"
    return f"{stem}{suffix}"


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_size(http: httpx.Client, item: HarvestObject) -> int | None:
    if item.expected_bytes is not None:
        return item.expected_bytes
    try:
        response = http.head(item.download_url)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise RuntimeError("Source rate-limited metadata lookup; batch must stop") from exc
        return None
    except httpx.HTTPError:
        return None
    return _integer(response.headers.get("content-length"))


def _download_one(
    http: httpx.Client,
    item: HarvestObject,
    destination: Path,
    *,
    retries: int,
    request_delay_seconds: float,
) -> tuple[str, int]:
    part = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            return _verify_download(destination, item)
        except OSError as exc:
            raise RuntimeError(
                f"Existing download failed integrity checks; preserved: {exc}"
            ) from exc
    for attempt in range(retries + 1):
        existing = part.stat().st_size if part.is_file() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            if "archive_header_offset" in item.metadata:
                if item.expected_bytes is None:
                    raise OSError("ZIP member is missing its declared length")
                part.write_bytes(read_zip_member(http, item.metadata, item.expected_bytes))
            else:
                with http.stream("GET", item.download_url, headers=headers) as response:
                    response.raise_for_status()
                    mode = "ab" if existing and response.status_code == 206 else "wb"
                    with part.open(mode) as handle:
                        for chunk in response.iter_bytes(1024 * 1024):
                            handle.write(chunk)
            sha256, size = _verify_download(part, item)
            os.replace(part, destination)
            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
            return sha256, size
        except (httpx.HTTPError, OSError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                raise RuntimeError(
                    "Source rate-limited the request; no immediate retry. Retry-After: "
                    + exc.response.headers.get("Retry-After", "not supplied")
                ) from exc
            if attempt >= retries:
                raise RuntimeError(f"download failed after {retries + 1} attempts: {exc}") from exc
            time.sleep(min(8.0, 2.0**attempt))
    raise AssertionError("unreachable")


def _verify_download(path: Path, item: HarvestObject) -> tuple[str, int]:
    """Apply source integrity checks both to new files and resumable cached files."""
    size = path.stat().st_size
    if item.expected_bytes is not None and size != item.expected_bytes:
        raise OSError(f"size mismatch: expected {item.expected_bytes}, got {size}")
    if item.expected_checksum_algorithm and item.expected_checksum:
        actual = _hash(path, item.expected_checksum_algorithm)
        if actual.lower() != item.expected_checksum.lower():
            raise OSError(f"{item.expected_checksum_algorithm} mismatch")
    expected_crc = item.metadata.get("archive_member_crc32")
    if expected_crc is not None:
        crc = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                crc = zlib.crc32(chunk, crc)
        if crc != int(expected_crc):
            raise OSError(f"archive member CRC32 mismatch: expected {expected_crc}, got {crc}")
    return _hash(path), size


def download_harvest_batch(
    harvest_manifest_path: str | Path,
    *,
    download_batch_id: str,
    acquisition_root: str | Path = DEFAULT_ACQUISITION_ROOT,
    max_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    max_objects: int = DEFAULT_MAX_BATCH_OBJECTS,
    minimum_free_gib: float = 15.0,
    maximum_acquisition_gib: float = 10.0,
    request_delay_seconds: float = 0.15,
    request_timeout_seconds: float = 60.0,
    download_workers: int = 1,
    retries: int = 3,
    dry_run: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, object]:
    """Download one bounded batch; oversized/unknown objects are reported, never forced."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive; the default remains a 2-GB batch")
    if not 0 < max_objects <= DEFAULT_MAX_BATCH_OBJECTS:
        raise ValueError(f"max_objects must be within 1..{DEFAULT_MAX_BATCH_OBJECTS}")
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if not 1 <= download_workers <= 4:
        raise ValueError("download_workers must be within 1..4")
    manifest_path = project_path(harvest_manifest_path)
    manifest = HarvestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.objects[:max_objects]
    http, owned = _client(client, timeout_seconds=request_timeout_seconds)
    selected: list[tuple[HarvestObject, int]] = []
    skipped: list[dict[str, object]] = []
    planned_bytes = 0
    rate_limited = threading.Event()
    try:
        for item in candidates:
            if rate_limited.is_set():
                skipped.append({"object_id": item.object_id, "reason": "source_rate_limit"})
                continue
            try:
                size = _remote_size(http, item)
            except RuntimeError:
                rate_limited.set()
                skipped.append({"object_id": item.object_id, "reason": "source_rate_limit"})
                continue
            if size is None:
                skipped.append({"object_id": item.object_id, "reason": "size_unknown"})
                continue
            if size > max_bytes:
                skipped.append(
                    {
                        "object_id": item.object_id,
                        "reason": "object_exceeds_batch_limit",
                        "bytes": size,
                    }
                )
                continue
            if planned_bytes + size > max_bytes:
                skipped.append(
                    {"object_id": item.object_id, "reason": "batch_byte_limit", "bytes": size}
                )
                continue
            selected.append((item, size))
            planned_bytes += size
        budget = check_disk_budget(
            project_path(acquisition_root),
            acquisition_root=acquisition_root,
            minimum_free_gib=minimum_free_gib,
            maximum_acquisition_gib=maximum_acquisition_gib,
            planned_additional_bytes=planned_bytes,
        )
        plan: dict[str, object] = {
            "source_id": manifest.source_id,
            "harvest_batch_id": manifest.batch_id,
            "download_batch_id": download_batch_id,
            "candidate_objects": len(candidates),
            "selected_objects": len(selected),
            "selected_bytes": planned_bytes,
            "skipped": skipped,
            "disk_budget": {
                "allowed": budget.allowed,
                "free_bytes": budget.free_bytes,
                "minimum_free_bytes": budget.minimum_free_bytes,
                "acquisition_bytes": budget.acquisition_bytes,
                "maximum_acquisition_bytes": budget.maximum_acquisition_bytes,
                "planned_additional_bytes": budget.planned_additional_bytes,
                "reasons": list(budget.reasons),
            },
            "dry_run": dry_run,
        }
        if dry_run:
            return plan
        if not budget.allowed:
            raise RuntimeError(f"Disk budget guard blocked batch: {', '.join(budget.reasons)}")
        root = project_path(acquisition_root)
        downloads = root / "downloads" / manifest.source_id / download_batch_id
        ledger = root / "manifests" / "object_events.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        seen_event_ids = _event_ids(ledger)
        completed: list[dict[str, object]] = []
        failed: list[dict[str, object]] = []

        def fetch(item: HarvestObject, destination: Path) -> tuple[str, int]:
            if rate_limited.is_set():
                raise RuntimeError("Not requested because this source is rate-limited")
            try:
                return _download_one(
                    http,
                    item,
                    destination,
                    retries=retries,
                    request_delay_seconds=request_delay_seconds,
                )
            except RuntimeError as exc:
                if "rate-limited" in str(exc):
                    rate_limited.set()
                raise

        with (
            ThreadPoolExecutor(max_workers=download_workers) as executor,
            ledger.open("a", encoding="utf-8") as ledger_handle,
        ):
            work: list[tuple[HarvestObject, Path, Future[tuple[str, int]]]] = []
            for item, _expected_size in selected:
                destination = (
                    downloads / f"{_stable_id(item.object_id)[:12]}_{_safe_filename(item.filename)}"
                )
                future = executor.submit(fetch, item, destination)
                work.append((item, destination, future))
            for item, destination, future in work:
                try:
                    sha256, actual_size = future.result()
                    entry = ObjectLedgerEntry(
                        event_id=_stable_id("download", download_batch_id, item.object_id, sha256),
                        observed_at=_now(),
                        object_id=item.object_id,
                        source_id=item.source_id,
                        batch_id=download_batch_id,
                        group_id=item.group_id,
                        sequence_id=item.sequence_id,
                        source_url=item.source_url,
                        filename=item.filename,
                        sha256=sha256,
                        byte_size=actual_size,
                        mime_type=item.mime_type,
                        width=item.width,
                        height=item.height,
                        source_timestamp=item.source_timestamp,
                        coarse_location=item.coarse_location,
                        raw_path=str(destination),
                        collection_state="quarantined",
                        licence_status=manifest.licence_status,
                        permitted_use="none_pending_review",
                        review_status="pending",
                    )
                    appended = False
                    if entry.event_id not in seen_event_ids:
                        ledger_handle.write(
                            json.dumps(entry.model_dump(mode="json"), sort_keys=True) + "\n"
                        )
                        ledger_handle.flush()
                        seen_event_ids.add(entry.event_id)
                        appended = True
                    completed.append(
                        {
                            "object_id": item.object_id,
                            "path": str(destination),
                            "bytes": actual_size,
                            "sha256": sha256,
                            "ledger_event_appended": appended,
                        }
                    )
                except RuntimeError as exc:
                    failed.append({"object_id": item.object_id, "error": str(exc)})
        plan.update(
            {
                "completed": completed,
                "failed": failed,
                "completed_objects": len(completed),
                "completed_bytes": sum(_required_int(row["bytes"]) for row in completed),
                "object_ledger": str(ledger),
                "rate_limited": rate_limited.is_set(),
                "status": "success"
                if completed and not failed
                else "partial"
                if completed
                else "blocked"
                if skipped and not failed
                else "failed",
            }
        )
        reports = root / "manifests" / "download_reports"
        reports.mkdir(parents=True, exist_ok=True)
        report_path = reports / f"{download_batch_id}.json"
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite download report: {report_path}")
        report_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        plan["report"] = str(report_path)
        return plan
    finally:
        if owned:
            http.close()


__all__ = [
    "DEFAULT_MAX_BATCH_BYTES",
    "DEFAULT_MAX_BATCH_OBJECTS",
    "MALAYSIA_KARTAVIEW_POINTS",
    "WIKIMEDIA_QUERIES",
    "HarvestManifest",
    "HarvestObject",
    "download_harvest_batch",
    "harvest_direct",
    "harvest_generic_manifest",
    "harvest_kartaview",
    "harvest_wikimedia",
    "harvest_zenodo",
    "import_local_inventory",
    "save_harvest_manifest",
]
