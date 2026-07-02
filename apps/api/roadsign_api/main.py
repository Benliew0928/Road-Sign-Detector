from __future__ import annotations

# pyright: reportUnusedFunction=false
import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import secrets
import socket
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Literal, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import cv2
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from roadsign_api.dependencies import get_engine
from roadsign_assist import __version__
from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.catalogue.models import SignCatalogue
from roadsign_assist.catalogue.repository import load_catalogue
from roadsign_assist.diagnostics import collect_diagnostics
from roadsign_assist.inference.engine import annotate_frame, decode_image, encode_jpeg
from roadsign_assist.inference.models import FrameResultModel, InferenceMode, SignEventModel
from roadsign_assist.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FPS_WINDOW_SECONDS = 1.5
PHONE_ACCESS_TTL_SECONDS = 6 * 60 * 60
OPERATOR_ACCESS_TTL_SECONDS = 12 * 60 * 60
PHONE_STREAM_IDLE_TIMEOUT_SECONDS = 15.0
PHONE_DEVICE_ID_MAX_LENGTH = 96
DEFAULT_MAX_PHONE_STREAMS = 12
GENERATED_DEMO_SECRET = secrets.token_urlsafe(32)
GENERATED_OPERATOR_TOKEN = secrets.token_urlsafe(24)


def _empty_sign_events() -> list[SignEventModel]:
    return []


def _empty_strings() -> list[str]:
    return []


def _float_deque() -> deque[float]:
    return deque()


def _local_ipv4_addresses() -> list[str]:
    candidates: set[str] = set()
    hostname = socket.gethostname()
    try:
        for result in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            address = str(result[4][0])
            parsed = ip_address(address)
            if not parsed.is_loopback and not parsed.is_link_local:
                candidates.add(address)
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            address = probe.getsockname()[0]
            parsed = ip_address(address)
            if not parsed.is_loopback and not parsed.is_link_local:
                candidates.add(address)
    except OSError:
        pass
    return sorted(candidates)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_base_url(raw: str) -> str | None:
    value = raw.strip().rstrip("/")
    if not value:
        return None
    if "://" not in value:
        value = f"https://{value}"
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def _public_base_url() -> str | None:
    return _normalize_base_url(os.environ.get("ROADSIGN_PUBLIC_BASE_URL", ""))


def _public_tunnel_enabled() -> bool:
    return _public_base_url() is not None or _truthy_env("ROADSIGN_TUNNEL_MODE")


def _public_netloc() -> str | None:
    public_base = _public_base_url()
    if public_base is None:
        return None
    return urlparse(public_base).netloc.lower()


def _phone_max_streams() -> int:
    raw_value = os.environ.get("ROADSIGN_PHONE_MAX_STREAMS", "").strip()
    if not raw_value:
        return DEFAULT_MAX_PHONE_STREAMS
    try:
        return max(1, int(raw_value))
    except ValueError:
        LOGGER.warning("Invalid ROADSIGN_PHONE_MAX_STREAMS=%r; using default", raw_value)
        return DEFAULT_MAX_PHONE_STREAMS


def _request_host_candidates(headers: Mapping[str, str]) -> set[str]:
    candidates: set[str] = set()
    for header in ("host", "x-forwarded-host", "x-original-host", "x-forwarded-server"):
        value = headers.get(header, "").strip()
        if not value:
            continue
        first_value = value.split(",", 1)[0].strip().lower()
        candidates.add(first_value)
        parsed_host = first_value.rsplit("@", 1)[-1].split(":", 1)[0]
        if parsed_host:
            candidates.add(parsed_host)
    return candidates


def _has_tunnel_proxy_headers(headers: Mapping[str, str]) -> bool:
    if "cf-connecting-ip" in headers:
        return True
    if "ngrok-skip-browser-warning" in headers:
        return True
    cdn_loop = headers.get("cdn-loop", "").lower()
    return "cloudflare" in cdn_loop


def _is_public_headers(headers: Mapping[str, str]) -> bool:
    public_netloc = _public_netloc()
    if public_netloc is None:
        return False
    public_host = public_netloc.split(":", 1)[0]
    host_candidates = _request_host_candidates(headers)
    return (
        public_netloc in host_candidates
        or public_host in host_candidates
        or _has_tunnel_proxy_headers(headers)
    )


def _is_public_request(request: Request) -> bool:
    return _is_public_headers(request.headers)


def _is_public_websocket(websocket: WebSocket) -> bool:
    return _is_public_headers(websocket.headers)


def _url_with_params(url: str, params: Mapping[str, str | None]) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    for key, value in params.items():
        if value is None:
            continue
        query[key] = value
    return urlunparse(parsed._replace(query=urlencode(query)))


def _websocket_url_for(base_url: str, path: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    base_path = parsed.path.rstrip("/")
    return urlunparse((scheme, parsed.netloc, f"{base_path}{path}", "", "", ""))


def _demo_secret() -> bytes:
    value = os.environ.get("ROADSIGN_DEMO_SECRET", "").strip() or GENERATED_DEMO_SECRET
    return value.encode("utf-8")


def _operator_token() -> str:
    return os.environ.get("ROADSIGN_OPERATOR_TOKEN", "").strip() or GENERATED_OPERATOR_TOKEN


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}".encode("ascii"))


def _sign_access_token(kind: str, subject: str, ttl_seconds: int) -> str:
    now = int(time.time())
    payload = {
        "kind": kind,
        "sub": subject,
        "iat": now,
        "exp": now + ttl_seconds,
        "nonce": secrets.token_urlsafe(8),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded_payload = _base64url_encode(payload_bytes)
    signature = hmac.new(_demo_secret(), encoded_payload.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded_payload}.{_base64url_encode(signature)}"


def _verify_access_token(token: str | None, kind: str, subject: str | None = None) -> bool:
    if not token or "." not in token:
        return False
    encoded_payload, encoded_signature = token.split(".", 1)
    expected = hmac.new(_demo_secret(), encoded_payload.encode("ascii"), hashlib.sha256).digest()
    try:
        supplied = _base64url_decode(encoded_signature)
        payload = json.loads(_base64url_decode(encoded_payload))
    except (binascii.Error, ValueError, json.JSONDecodeError):
        return False
    if not hmac.compare_digest(supplied, expected):
        return False
    if payload.get("kind") != kind:
        return False
    if subject is not None and payload.get("sub") != subject:
        return False
    expires_at = payload.get("exp")
    return isinstance(expires_at, int) and expires_at >= int(time.time())


def _has_operator_access(token: str | None) -> bool:
    if not token:
        return False
    return hmac.compare_digest(token, _operator_token()) or _verify_access_token(
        token,
        "operator",
        "live",
    )


def _operator_token_from_request(request: Request) -> str | None:
    return request.query_params.get("operator") or request.headers.get("x-roadsign-operator-token")


def _operator_token_from_websocket(websocket: WebSocket) -> str | None:
    return websocket.query_params.get("operator") or websocket.headers.get(
        "x-roadsign-operator-token"
    )


def _validate_public_phone_access(session_id: str, token: str | None) -> bool:
    return _verify_access_token(token, "phone", session_id)


def _ensure_monitor_access(request: Request) -> None:
    if _is_public_request(request):
        if _has_operator_access(_operator_token_from_request(request)):
            return
        raise HTTPException(status_code=403, detail="Operator access token is required.")
    _ensure_host_client(request)


def _ensure_phone_page_access(request: Request) -> None:
    if not _is_public_request(request):
        return
    session_id = request.query_params.get("session") or ""
    access_token = request.query_params.get("access")
    if session_id and _validate_public_phone_access(session_id, access_token):
        return
    raise HTTPException(status_code=403, detail="Phone link is expired or not authorized.")


def _phone_base_urls(request: Request) -> tuple[str, list[str]]:
    public_base = _public_base_url()
    public_host = os.environ.get("ROADSIGN_PUBLIC_HOST", "").strip()
    scheme = request.url.scheme
    request_host = request.url.netloc
    selected_host = public_host or request_host
    if public_host and ":" not in public_host and request.url.port:
        selected_host = f"{public_host}:{request.url.port}"

    def base_for(host: str) -> str:
        return f"{scheme}://{host}"

    candidates = [public_base] if public_base else []
    candidates.append(base_for(selected_host))
    request_hostname = request.url.hostname or ""
    if request_hostname in {"127.0.0.1", "localhost", "::1"}:
        port = request.url.port
        suffix = f":{port}" if port else ""
        candidates.extend(base_for(f"{address}{suffix}") for address in _local_ipv4_addresses())
    unique_candidates = list(dict.fromkeys(candidates))
    return unique_candidates[0], unique_candidates


def _is_host_client(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        parsed = ip_address(host)
    except ValueError:
        return False
    if parsed.is_loopback:
        return True
    return str(parsed) in set(_local_ipv4_addresses())


def _ensure_host_client(request: Request) -> None:
    client_host = request.client.host if request.client else None
    if not _is_host_client(client_host):
        raise HTTPException(
            status_code=403,
            detail="Live camera wall is available only from the host laptop.",
        )


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    index = 2
    standalone_markers = {0x01, *range(0xD0, 0xD9)}
    size_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while index < len(data) - 9:
        if data[index] != 0xFF:
            index += 1
            continue
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            return None
        marker = data[index]
        index += 1
        if marker in standalone_markers:
            continue
        if index + 2 > len(data):
            return None
        segment_length = int.from_bytes(data[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > len(data):
            return None
        if marker in size_markers and segment_length >= 7:
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            if width > 0 and height > 0:
                return width, height
            return None
        index += segment_length
    return None


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if width > 0 and height > 0:
        return width, height
    return None


def _image_dimensions(data: bytes) -> tuple[int, int]:
    dimensions = _jpeg_dimensions(data) or _png_dimensions(data)
    if dimensions is not None:
        return dimensions
    image = decode_image(data)
    return image.shape[1], image.shape[0]


def _empty_frame_result(
    engine_mode: InferenceMode,
    warnings: list[str],
    frame_id: int,
    width: int,
    height: int,
    latency_ms: float,
) -> FrameResultModel:
    return FrameResultModel(
        frame_id=frame_id,
        width=width,
        height=height,
        mode=engine_mode,
        latency_ms=latency_ms,
        events=[],
        warnings=warnings,
    )


def _ack_result_for_frame(
    latest_result: FrameResultModel | None,
    engine_mode: InferenceMode,
    warnings: list[str],
    frame_id: int,
    width: int,
    height: int,
    latency_ms: float,
) -> FrameResultModel:
    if latest_result is None:
        return _empty_frame_result(engine_mode, warnings, frame_id, width, height, latency_ms)
    return latest_result.model_copy(
        update={
            "frame_id": frame_id,
            "width": width,
            "height": height,
            "latency_ms": latency_ms,
        }
    )


class ImageInferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: FrameResultModel
    annotated_jpeg_base64: str


class DiagnosticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python: str
    platform: str
    project_root: str
    opencv: str
    cuda_available: bool
    onnxruntime_available: bool
    official_image_count: int
    writable: bool
    ffmpeg_available: bool
    offline_ocr_ready: bool
    production_segmenter_present: bool
    production_classifier_present: bool
    semantic_ai_ready: bool
    experimental_model_count: int
    healthy: bool


class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: InferenceMode
    detector: str
    detector_available: bool
    detector_loaded: bool
    detector_device: str | None
    detector_profile: dict[str, object] = Field(default_factory=dict)
    classifier: str
    classifier_available: bool
    classifier_loaded: bool
    classifier_providers: list[str] = Field(default_factory=list)
    classifier_profile: dict[str, object] = Field(default_factory=dict)
    tracker: str
    ocr_available: bool
    ocr_loaded: bool
    ocr_load_error: str | None
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    version: str
    diagnostics: DiagnosticsResponse
    models: ModelStatusResponse


class BatchInferenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str | None = None
    result: FrameResultModel | None = None
    error: str | None = None


class BatchInferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    results: list[BatchInferenceItem]


class VideoInferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frames_read: int = Field(ge=0)
    sampled_frames: int = Field(ge=0)
    events: int = Field(ge=0)
    event_samples: list[SignEventModel] = Field(default_factory=_empty_sign_events)
    representative_result: FrameResultModel | None = None


class PhoneConnectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=8)
    phone_url: str
    websocket_url: str
    candidate_urls: list[str] = Field(default_factory=_empty_strings)
    https: bool
    camera_requires_https: bool = True
    mode: Literal["local", "public_tunnel"] = "local"
    public_base_url: str | None = None
    access_token: str | None = None
    operator_live_url: str | None = None


class PhoneStreamSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stream_id: str
    session_id: str
    device_id: str | None = None
    label: str
    connected_at: float
    updated_at: float
    frame_seq: int = Field(ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    jpeg_base64: str | None = None
    result: FrameResultModel | None = None
    live_fps: float = Field(default=0.0, ge=0.0)
    inference_fps: float = Field(default=0.0, ge=0.0)
    inference_pending: bool = False
    inference_frame_seq: int = Field(default=0, ge=0)


class PhoneStreamsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    streams: list[PhoneStreamSnapshot]


@dataclass
class PhoneStreamState:
    stream_id: str
    session_id: str
    device_id: str | None
    label: str
    connection_token: str
    connected_at: float
    updated_at: float
    frame_seq: int = 0
    width: int | None = None
    height: int | None = None
    jpeg_base64: str | None = None
    result: FrameResultModel | None = None
    live_fps: float = 0.0
    inference_fps: float = 0.0
    inference_pending: bool = False
    inference_frame_seq: int = 0
    frame_times: deque[float] = field(default_factory=_float_deque)
    inference_times: deque[float] = field(default_factory=_float_deque)


@dataclass(frozen=True)
class PhoneStreamHandle:
    stream_id: str
    token: str


@dataclass(frozen=True)
class PhoneFrame:
    frame_id: int
    frame_seq: int
    data: bytes
    width: int
    height: int
    received_at: float


class PhoneStreamLimitError(RuntimeError):
    pass


class PhoneStreamRegistry:
    def __init__(self) -> None:
        self._streams: dict[str, PhoneStreamState] = {}
        self._subscribers: set[asyncio.Queue[dict[str, object]]] = set()
        self._lock = asyncio.Lock()
        self._next_label = 1

    def _snapshots_unlocked(self) -> list[PhoneStreamSnapshot]:
        return [
            PhoneStreamSnapshot(
                stream_id=stream.stream_id,
                session_id=stream.session_id,
                device_id=stream.device_id,
                label=stream.label,
                connected_at=stream.connected_at,
                updated_at=stream.updated_at,
                frame_seq=stream.frame_seq,
                width=stream.width,
                height=stream.height,
                jpeg_base64=stream.jpeg_base64,
                result=stream.result,
                live_fps=stream.live_fps,
                inference_fps=stream.inference_fps,
                inference_pending=stream.inference_pending,
                inference_frame_seq=stream.inference_frame_seq,
            )
            for stream in sorted(self._streams.values(), key=lambda item: item.connected_at)
        ]

    def _snapshot_message_unlocked(self) -> dict[str, object]:
        return {
            "type": "snapshot",
            "streams": [
                snapshot.model_dump(mode="json") for snapshot in self._snapshots_unlocked()
            ],
        }

    def _snapshot_for_stream(self, stream: PhoneStreamState) -> PhoneStreamSnapshot:
        return PhoneStreamSnapshot(
            stream_id=stream.stream_id,
            session_id=stream.session_id,
            device_id=stream.device_id,
            label=stream.label,
            connected_at=stream.connected_at,
            updated_at=stream.updated_at,
            frame_seq=stream.frame_seq,
            width=stream.width,
            height=stream.height,
            jpeg_base64=stream.jpeg_base64,
            result=stream.result,
            live_fps=stream.live_fps,
            inference_fps=stream.inference_fps,
            inference_pending=stream.inference_pending,
            inference_frame_seq=stream.inference_frame_seq,
        )

    def _update_message(self, snapshot: PhoneStreamSnapshot) -> dict[str, object]:
        return {"type": "update", "stream": snapshot.model_dump(mode="json")}

    def _publish_unlocked(self, message: dict[str, object]) -> None:
        for queue in list(self._subscribers):
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(message)

    def _record_fps(self, times: deque[float], now: float) -> float:
        times.append(now)
        cutoff = now - FPS_WINDOW_SECONDS
        while times and times[0] < cutoff:
            times.popleft()
        if len(times) <= 1:
            return float(len(times))
        elapsed = max(0.001, times[-1] - times[0])
        return (len(times) - 1) / elapsed

    def _normalized_device_id(self, device_id: str | None) -> str | None:
        value = (device_id or "").strip()
        if not value:
            return None
        return value[:PHONE_DEVICE_ID_MAX_LENGTH]

    async def register(self, session_id: str, device_id: str | None = None) -> PhoneStreamHandle:
        normalized_device_id = self._normalized_device_id(device_id)
        stream_id = uuid.uuid4().hex
        token = uuid.uuid4().hex
        now = time.time()
        async with self._lock:
            stream_label: str | None = None
            if normalized_device_id is not None:
                for existing in list(self._streams.values()):
                    if (
                        existing.session_id == session_id
                        and existing.device_id == normalized_device_id
                    ):
                        stream_label = existing.label
                        del self._streams[existing.stream_id]
                        break
            if len(self._streams) >= _phone_max_streams():
                raise PhoneStreamLimitError("Maximum connected phone streams reached")
            if stream_label is None:
                stream_label = f"Device {self._next_label}"
                self._next_label += 1
            self._streams[stream_id] = PhoneStreamState(
                stream_id=stream_id,
                session_id=session_id,
                device_id=normalized_device_id,
                label=stream_label,
                connection_token=token,
                connected_at=now,
                updated_at=now,
            )
            self._publish_unlocked(self._snapshot_message_unlocked())
        return PhoneStreamHandle(stream_id=stream_id, token=token)

    async def update_live_frame(
        self,
        stream_id: str,
        token: str,
        jpeg_bytes: bytes,
        frame_seq: int,
        width: int,
        height: int,
        result: FrameResultModel,
    ) -> None:
        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        async with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None or stream.connection_token != token:
                return
            now = time.time()
            stream.updated_at = now
            stream.frame_seq = frame_seq
            stream.width = width
            stream.height = height
            stream.jpeg_base64 = encoded
            stream.result = result
            stream.live_fps = self._record_fps(stream.frame_times, now)
            snapshot = self._snapshot_for_stream(stream)
            self._publish_unlocked(self._update_message(snapshot))

    async def update_inference_result(
        self,
        stream_id: str,
        token: str,
        result: FrameResultModel,
        frame_seq: int,
    ) -> None:
        async with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None or stream.connection_token != token:
                return
            now = time.time()
            stream.updated_at = now
            stream.result = result
            stream.inference_pending = False
            stream.inference_frame_seq = frame_seq
            stream.inference_fps = self._record_fps(stream.inference_times, now)
            snapshot = self._snapshot_for_stream(stream)
            self._publish_unlocked(self._update_message(snapshot))

    async def set_inference_pending(self, stream_id: str, token: str, pending: bool) -> None:
        async with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None or stream.connection_token != token:
                return
            stream.inference_pending = pending
            snapshot = self._snapshot_for_stream(stream)
            self._publish_unlocked(self._update_message(snapshot))

    async def unregister(self, stream_id: str, token: str) -> None:
        async with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None or stream.connection_token != token:
                return
            del self._streams[stream_id]
            self._publish_unlocked(self._snapshot_message_unlocked())

    async def response(self) -> PhoneStreamsResponse:
        async with self._lock:
            return PhoneStreamsResponse(streams=self._snapshots_unlocked())

    async def subscribe(self) -> tuple[asyncio.Queue[dict[str, object]], dict[str, object]]:
        async with self._lock:
            queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=2)
            self._subscribers.add(queue)
            return queue, self._snapshot_message_unlocked()

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, object]]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)


PHONE_STREAMS = PhoneStreamRegistry()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        engine = get_engine()
        await asyncio.to_thread(engine.warmup)
        yield

    application = FastAPI(
        title="RoadSign Assist API",
        version=__version__,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @application.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        diagnostics = collect_diagnostics()
        engine = get_engine()
        return HealthResponse(
            status="ok" if diagnostics.healthy else "degraded",
            version=__version__,
            diagnostics=DiagnosticsResponse.model_validate(
                {**asdict(diagnostics), "healthy": diagnostics.healthy}
            ),
            models=ModelStatusResponse.model_validate(engine.model_status),
        )

    @application.get("/api/v1/catalogue", response_model=SignCatalogue)
    async def catalogue() -> SignCatalogue:
        return load_catalogue()

    @application.get("/api/v1/models", response_model=ModelStatusResponse)
    async def models() -> ModelStatusResponse:
        return ModelStatusResponse.model_validate(get_engine().model_status)

    @application.get("/api/v1/phone/connection", response_model=PhoneConnectionResponse)
    async def phone_connection(request: Request) -> PhoneConnectionResponse:
        if _is_public_request(request) and not _has_operator_access(
            _operator_token_from_request(request)
        ):
            raise HTTPException(status_code=403, detail="Operator access token is required.")

        session_id = uuid.uuid4().hex
        base_url, candidate_bases = _phone_base_urls(request)
        public_base = _public_base_url()
        access_token = (
            _sign_access_token("phone", session_id, PHONE_ACCESS_TTL_SECONDS)
            if public_base is not None
            else None
        )
        phone_url = _url_with_params(
            f"{base_url}/phone",
            {"session": session_id, "access": access_token},
        )
        websocket_url = _url_with_params(
            _websocket_url_for(base_url, f"/api/v1/ws/camera/{session_id}"),
            {"access": access_token},
        )
        operator_live_url = (
            _url_with_params(f"{public_base}/live", {"operator": _operator_token()})
            if public_base is not None
            else None
        )
        return PhoneConnectionResponse(
            session_id=session_id,
            phone_url=phone_url,
            websocket_url=websocket_url,
            candidate_urls=[
                _url_with_params(
                    f"{candidate}/phone",
                    {"session": session_id, "access": access_token},
                )
                for candidate in candidate_bases
            ],
            https=urlparse(base_url).scheme == "https",
            mode="public_tunnel" if public_base is not None else "local",
            public_base_url=public_base,
            access_token=access_token,
            operator_live_url=operator_live_url,
        )

    @application.get("/api/v1/phone/streams", response_model=PhoneStreamsResponse)
    async def phone_streams(request: Request) -> PhoneStreamsResponse:
        _ensure_monitor_access(request)
        return await PHONE_STREAMS.response()

    @application.post("/api/v1/infer/image", response_model=ImageInferenceResponse)
    async def infer_image(file: Annotated[UploadFile, File()]) -> ImageInferenceResponse:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 20 MB")
        try:
            image = decode_image(data)
            engine = get_engine().new_session()
            result = await asyncio.to_thread(
                engine.process_frame,
                image,
                assume_stable=True,
            )
            annotated = annotate_frame(image, result)
            encoded = base64.b64encode(encode_jpeg(annotated)).decode("ascii")
            return ImageInferenceResponse(result=result, annotated_jpeg_base64=encoded)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @application.post("/api/v1/infer/batch", response_model=BatchInferenceResponse)
    async def infer_batch(files: Annotated[list[UploadFile], File()]) -> BatchInferenceResponse:
        if not files or len(files) > 100:
            raise HTTPException(status_code=400, detail="Provide between 1 and 100 images")
        results: list[BatchInferenceItem] = []
        for file in files:
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                results.append(
                    BatchInferenceItem(filename=file.filename, error="Image exceeds 20 MB")
                )
                continue
            try:
                image = decode_image(data)
                engine = get_engine().new_session()
                result = await asyncio.to_thread(
                    engine.process_frame,
                    image,
                    assume_stable=True,
                )
                results.append(BatchInferenceItem(filename=file.filename, result=result))
            except ValueError as exc:
                results.append(BatchInferenceItem(filename=file.filename, error=str(exc)))
        return BatchInferenceResponse(count=len(results), results=results)

    @application.post("/api/v1/infer/video", response_model=VideoInferenceResponse)
    async def infer_video(file: Annotated[UploadFile, File()]) -> VideoInferenceResponse:
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        temp_root = PROJECT_ROOT / "outputs" / "uploads"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_path = temp_root / f"video-{id(file)}{suffix}"
        data = await file.read(250 * 1024 * 1024 + 1)
        if len(data) > 250 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Video exceeds 250 MB")
        temp_path.write_bytes(data)
        capture = cv2.VideoCapture(str(temp_path))
        if not capture.isOpened():
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Unable to decode video")
        engine = get_engine().new_session()
        frames = 0
        event_count = 0
        event_samples: list[SignEventModel] = []
        representative_result: FrameResultModel | None = None
        try:
            while frames < 300:
                success, frame = capture.read()
                if not success:
                    break
                if frames % 3 == 0:
                    result = await asyncio.to_thread(
                        engine.process_frame,
                        cast(UInt8Image, frame),
                    )
                    frame_events = result.events
                    event_count += len(frame_events)
                    if frame_events:
                        event_samples.extend(frame_events)
                        event_samples = event_samples[-40:]
                        representative_result = result
                frames += 1
        finally:
            capture.release()
            temp_path.unlink(missing_ok=True)
        return VideoInferenceResponse(
            frames_read=frames,
            sampled_frames=(frames + 2) // 3,
            events=event_count,
            event_samples=event_samples,
            representative_result=representative_result,
        )

    @application.websocket("/api/v1/ws/camera/{session_id}")
    async def camera_socket(websocket: WebSocket, session_id: str) -> None:
        if _is_public_websocket(websocket) and not _validate_public_phone_access(
            session_id,
            websocket.query_params.get("access"),
        ):
            await websocket.close(code=1008)
            return

        await websocket.accept()
        try:
            stream_handle = await PHONE_STREAMS.register(
                session_id,
                websocket.query_params.get("device"),
            )
        except PhoneStreamLimitError:
            await websocket.send_json(
                {"error": "Too many phones are streaming. Stop one device first."}
            )
            await websocket.close(code=1013)
            return
        engine = get_engine().new_session()
        latest_frame: PhoneFrame | None = None
        latest_result: FrameResultModel | None = None
        latest_lock = asyncio.Lock()
        inference_event = asyncio.Event()
        frame_seq = 0

        async def inference_loop() -> None:
            nonlocal latest_result
            while True:
                await inference_event.wait()
                inference_event.clear()
                async with latest_lock:
                    frame = latest_frame
                if frame is None:
                    continue
                await PHONE_STREAMS.set_inference_pending(
                    stream_handle.stream_id,
                    stream_handle.token,
                    True,
                )
                try:
                    image = await asyncio.to_thread(decode_image, frame.data)
                    result = await asyncio.to_thread(engine.process_frame, image)
                except ValueError as exc:
                    LOGGER.warning("Phone inference skipped a bad frame: %s", exc)
                    await PHONE_STREAMS.set_inference_pending(
                        stream_handle.stream_id,
                        stream_handle.token,
                        False,
                    )
                    continue
                result = result.model_copy(
                    update={
                        "frame_id": frame.frame_id,
                        "width": frame.width,
                        "height": frame.height,
                    }
                )
                async with latest_lock:
                    latest_result = result
                await PHONE_STREAMS.update_inference_result(
                    stream_handle.stream_id,
                    stream_handle.token,
                    result,
                    frame.frame_seq,
                )

        inference_task = asyncio.create_task(inference_loop())
        try:
            while True:
                started = time.perf_counter()
                data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=PHONE_STREAM_IDLE_TIMEOUT_SECONDS,
                )
                if len(data) > MAX_UPLOAD_BYTES:
                    await websocket.send_json({"error": "Frame exceeds 20 MB"})
                    continue
                try:
                    width, height = _image_dimensions(data)
                except ValueError as exc:
                    await websocket.send_json({"error": str(exc)})
                    continue

                frame_seq += 1
                frame_id = frame_seq - 1
                async with latest_lock:
                    ack_result = _ack_result_for_frame(
                        latest_result,
                        engine.mode,
                        list(engine.warnings),
                        frame_id,
                        width,
                        height,
                        (time.perf_counter() - started) * 1000,
                    )
                    latest_frame = PhoneFrame(
                        frame_id=frame_id,
                        frame_seq=frame_seq,
                        data=data,
                        width=width,
                        height=height,
                        received_at=time.time(),
                    )
                await websocket.send_json(ack_result.model_dump(mode="json"))
                await PHONE_STREAMS.update_live_frame(
                    stream_handle.stream_id,
                    stream_handle.token,
                    data,
                    frame_seq,
                    width,
                    height,
                    ack_result,
                )
                inference_event.set()
        except TimeoutError:
            LOGGER.info("Camera WebSocket closed after %.1f seconds without frames", PHONE_STREAM_IDLE_TIMEOUT_SECONDS)
            await websocket.close(code=1001)
        except WebSocketDisconnect:
            LOGGER.info("Camera WebSocket disconnected")
        finally:
            inference_task.cancel()
            with suppress(asyncio.CancelledError):
                await inference_task
            await PHONE_STREAMS.unregister(stream_handle.stream_id, stream_handle.token)

    @application.websocket("/api/v1/ws/phone/monitor")
    async def phone_monitor_socket(websocket: WebSocket) -> None:
        if _is_public_websocket(websocket):
            if not _has_operator_access(_operator_token_from_websocket(websocket)):
                await websocket.close(code=1008)
                return
        elif not _is_host_client(websocket.client.host if websocket.client else None):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        queue, initial = await PHONE_STREAMS.subscribe()
        try:
            await websocket.send_json(initial)
            while True:
                await websocket.send_json(await queue.get())
        except WebSocketDisconnect:
            LOGGER.info("Phone monitor WebSocket disconnected")
        finally:
            await PHONE_STREAMS.unsubscribe(queue)

    web_dist = PROJECT_ROOT / "apps" / "web" / "dist"
    if web_dist.exists():
        @application.get("/live", include_in_schema=False)
        @application.get("/live/", include_in_schema=False)
        async def live_wall_app(request: Request) -> FileResponse:
            _ensure_monitor_access(request)
            return FileResponse(web_dist / "index.html")

        @application.get("/phone", include_in_schema=False)
        @application.get("/phone/", include_in_schema=False)
        async def phone_app(request: Request) -> FileResponse:
            _ensure_phone_page_access(request)
            return FileResponse(web_dist / "index.html")

        application.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return application


app = create_app()
