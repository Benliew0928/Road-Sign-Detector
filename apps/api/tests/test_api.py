# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roadsign_api.main import MAX_UPLOAD_BYTES, app
from roadsign_assist.paths import OFFICIAL_ROOT, PROJECT_ROOT


def _sample_image_bytes() -> bytes:
    return next((OFFICIAL_ROOT / "assignment_images").rglob("*.png")).read_bytes()


async def test_health_endpoint() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["diagnostics"]["official_image_count"] == 84
        assert body["diagnostics"]["healthy"] is True
        assert body["diagnostics"]["project_root"]
        models = body["models"]
        assert models["mode"] in {"baseline", "deep", "auto"}
        assert isinstance(models["detector_loaded"], bool)
        assert "detector_device" in models
        assert isinstance(models["classifier_loaded"], bool)
        assert isinstance(models["classifier_providers"], list)
        assert isinstance(models["tracker"], str)
        assert isinstance(models["ocr_loaded"], bool)
        assert "ocr_load_error" in models


async def test_models_endpoint_contract() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] in {"baseline", "deep", "auto"}
        assert isinstance(body["detector"], str)
        assert isinstance(body["detector_available"], bool)
        assert isinstance(body["detector_loaded"], bool)
        assert "detector_device" in body
        assert isinstance(body["classifier"], str)
        assert isinstance(body["classifier_available"], bool)
        assert isinstance(body["classifier_loaded"], bool)
        assert isinstance(body["classifier_providers"], list)
        assert isinstance(body["tracker"], str)
        assert isinstance(body["ocr_available"], bool)
        assert isinstance(body["ocr_loaded"], bool)
        assert "ocr_load_error" in body
        assert isinstance(body["warnings"], list)


async def test_catalogue_endpoint() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/catalogue")
        assert response.status_code == 200
        assert len(response.json()["entries"]) >= 60


async def test_image_and_batch_inference_contracts() -> None:
    image = _sample_image_bytes()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        image_response = await client.post(
            "/api/v1/infer/image",
            files={"file": ("sample.png", image, "image/png")},
        )
        assert image_response.status_code == 200
        image_body = image_response.json()
        assert image_body["result"]["frame_id"] == 0
        assert image_body["annotated_jpeg_base64"]

        batch_response = await client.post(
            "/api/v1/infer/batch",
            files=[
                ("files", ("first.png", image, "image/png")),
                ("files", ("second.png", image, "image/png")),
            ],
        )
        assert batch_response.status_code == 200
        batch_body = batch_response.json()
        assert batch_body["count"] == 2
        assert [item["result"]["frame_id"] for item in batch_body["results"]] == [0, 0]


async def test_image_upload_size_limit() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/infer/image",
            files={"file": ("large.jpg", b"0" * (MAX_UPLOAD_BYTES + 1), "image/jpeg")},
        )
    assert response.status_code == 413
    assert response.json()["detail"] == "Image exceeds 20 MB"


async def test_batch_rejects_more_than_100_images() -> None:
    image = _sample_image_bytes()
    files = [("files", (f"sample-{index}.png", image, "image/png")) for index in range(101)]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/infer/batch", files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "Provide between 1 and 100 images"


def test_camera_websocket_contract() -> None:
    with (
        TestClient(app) as client,
        client.websocket_connect("/api/v1/ws/camera/test-session") as websocket,
    ):
        websocket.send_bytes(_sample_image_bytes())
        body = websocket.receive_json()
        assert body["frame_id"] == 0
        assert body["width"] > 0
        assert "events" in body


def test_camera_websocket_recovers_after_bad_frame() -> None:
    with (
        TestClient(app) as client,
        client.websocket_connect("/api/v1/ws/camera/recovery-session") as websocket,
    ):
        websocket.send_bytes(b"not an image")
        error_body = websocket.receive_json()
        assert "error" in error_body

        websocket.send_bytes(_sample_image_bytes())
        body = websocket.receive_json()
        assert body["frame_id"] == 0
        assert body["width"] > 0
        assert "events" in body


def test_phone_stream_snapshot_tracks_connected_camera() -> None:
    session_id = "live-wall-session"
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with client.websocket_connect(f"/api/v1/ws/camera/{session_id}") as websocket:
            websocket.send_bytes(_sample_image_bytes())
            camera_body = websocket.receive_json()

            response = client.get("/api/v1/phone/streams")
            assert response.status_code == 200
            streams = response.json()["streams"]
            stream = next(item for item in streams if item["session_id"] == session_id)
            assert stream["stream_id"]
            assert stream["label"].startswith("Device ")
            assert stream["frame_seq"] == 1
            assert stream["jpeg_base64"]
            assert stream["live_fps"] >= 0
            assert stream["inference_fps"] >= 0
            assert isinstance(stream["inference_pending"], bool)
            assert stream["result"]["frame_id"] == camera_body["frame_id"]
            assert stream["result"]["width"] == camera_body["width"]

        response = client.get("/api/v1/phone/streams")
        assert response.status_code == 200
        assert all(item["session_id"] != session_id for item in response.json()["streams"])


def test_phone_streams_keep_multiple_devices_from_same_qr() -> None:
    session_id = "shared-qr-session"
    with (
        TestClient(app, client=("127.0.0.1", 50000)) as client,
        client.websocket_connect(f"/api/v1/ws/camera/{session_id}") as first,
        client.websocket_connect(f"/api/v1/ws/camera/{session_id}") as second,
    ):
        first.send_bytes(_sample_image_bytes())
        first.receive_json()
        second.send_bytes(_sample_image_bytes())
        second.receive_json()

        response = client.get("/api/v1/phone/streams")
        assert response.status_code == 200
        streams = [
            item for item in response.json()["streams"] if item["session_id"] == session_id
        ]
        assert len(streams) == 2
        assert len({item["stream_id"] for item in streams}) == 2
        assert len({item["label"] for item in streams}) == 2
        assert all(item["jpeg_base64"] for item in streams)


def test_phone_stream_restart_replaces_same_device_tile() -> None:
    session_id = "restart-same-device-session"
    device_id = "phone-device-a"
    with (
        TestClient(app, client=("127.0.0.1", 50000)) as client,
        client.websocket_connect(
            f"/api/v1/ws/camera/{session_id}?device={device_id}"
        ) as first,
    ):
        first.send_bytes(_sample_image_bytes())
        first.receive_json()

        with client.websocket_connect(
            f"/api/v1/ws/camera/{session_id}?device={device_id}"
        ) as second:
            second.send_bytes(_sample_image_bytes())
            second.receive_json()

            response = client.get("/api/v1/phone/streams")
            assert response.status_code == 200
            streams = [
                item for item in response.json()["streams"] if item["session_id"] == session_id
            ]
            assert len(streams) == 1
            assert streams[0]["device_id"] == device_id
            assert streams[0]["jpeg_base64"]


def test_phone_stream_same_qr_keeps_different_device_tiles() -> None:
    session_id = "shared-qr-different-devices"
    with (
        TestClient(app, client=("127.0.0.1", 50000)) as client,
        client.websocket_connect(f"/api/v1/ws/camera/{session_id}?device=device-a") as first,
        client.websocket_connect(f"/api/v1/ws/camera/{session_id}?device=device-b") as second,
    ):
        first.send_bytes(_sample_image_bytes())
        first.receive_json()
        second.send_bytes(_sample_image_bytes())
        second.receive_json()

        response = client.get("/api/v1/phone/streams")
        assert response.status_code == 200
        streams = [
            item for item in response.json()["streams"] if item["session_id"] == session_id
        ]
        assert len(streams) == 2
        assert {item["device_id"] for item in streams} == {"device-a", "device-b"}
        assert len({item["stream_id"] for item in streams}) == 2


def test_phone_stream_snapshot_is_host_only() -> None:
    with TestClient(app, client=("192.0.2.10", 50000)) as client:
        response = client.get("/api/v1/phone/streams")
    assert response.status_code == 403
    assert response.json()["detail"] == "Live camera wall is available only from the host laptop."


def test_phone_monitor_websocket_receives_camera_updates() -> None:
    session_id = "monitor-ws-session"
    with (
        TestClient(app, client=("127.0.0.1", 50000)) as client,
        client.websocket_connect("/api/v1/ws/phone/monitor") as monitor,
    ):
        initial = monitor.receive_json()
        assert initial["type"] == "snapshot"

        with client.websocket_connect(f"/api/v1/ws/camera/{session_id}") as camera:
            connected = monitor.receive_json()
            assert connected["type"] == "snapshot"
            assert any(item["session_id"] == session_id for item in connected["streams"])

            camera.send_bytes(_sample_image_bytes())
            camera_body = camera.receive_json()
            update = monitor.receive_json()
            assert update["type"] == "update"
            assert update["stream"]["session_id"] == session_id
            assert update["stream"]["stream_id"]
            assert update["stream"]["jpeg_base64"]
            assert update["stream"]["result"]["frame_id"] == camera_body["frame_id"]


async def test_video_inference_contract(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter.fourcc(*"MJPG"),
        10.0,
        (64, 64),
    )
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((64, 64, 3), 30 + index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/infer/video",
            files={"file": ("sample.avi", video_path.read_bytes(), "video/x-msvideo")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["frames_read"] == 3
    assert body["sampled_frames"] == 1
    assert body["events"] == 0
    assert body["event_samples"] == []
    assert body["representative_result"] is None


async def test_phone_connection_contract() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="https://192.168.1.20:8443") as client:
        response = await client.get("/api/v1/phone/connection")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["phone_url"].startswith("https://192.168.1.20:8443/phone?session=")
    assert body["websocket_url"].startswith("wss://192.168.1.20:8443/api/v1/ws/camera/")
    assert body["phone_url"] in body["candidate_urls"]
    assert body["https"] is True
    assert body["camera_requires_https"] is True
    assert body["mode"] == "local"
    assert body["public_base_url"] is None
    assert body["access_token"] is None
    assert body["operator_live_url"] is None


async def test_public_tunnel_connection_uses_signed_public_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROADSIGN_PUBLIC_BASE_URL", "https://roadsign-demo.trycloudflare.com")
    monkeypatch.setenv("ROADSIGN_DEMO_SECRET", "test-demo-secret")
    monkeypatch.setenv("ROADSIGN_OPERATOR_TOKEN", "test-operator-token")

    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 50000))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8443") as client:
        response = await client.get("/api/v1/phone/connection")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "public_tunnel"
    assert body["public_base_url"] == "https://roadsign-demo.trycloudflare.com"
    assert body["https"] is True
    assert body["access_token"]
    assert body["phone_url"].startswith("https://roadsign-demo.trycloudflare.com/phone?")
    assert body["websocket_url"].startswith(
        "wss://roadsign-demo.trycloudflare.com/api/v1/ws/camera/"
    )
    phone_query = parse_qs(urlparse(body["phone_url"]).query)
    websocket_query = parse_qs(urlparse(body["websocket_url"]).query)
    assert phone_query["session"] == [body["session_id"]]
    assert phone_query["access"] == [body["access_token"]]
    assert websocket_query["access"] == [body["access_token"]]
    assert body["candidate_urls"][0] == body["phone_url"]
    assert body["operator_live_url"] == (
        "https://roadsign-demo.trycloudflare.com/live?operator=test-operator-token"
    )


async def test_public_tunnel_requires_operator_for_public_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROADSIGN_PUBLIC_BASE_URL", "https://roadsign-demo.trycloudflare.com")
    monkeypatch.setenv("ROADSIGN_DEMO_SECRET", "test-demo-secret")
    monkeypatch.setenv("ROADSIGN_OPERATOR_TOKEN", "test-operator-token")

    transport = httpx.ASGITransport(app=app, client=("203.0.113.10", 50000))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://roadsign-demo.trycloudflare.com",
    ) as client:
        denied = await client.get("/api/v1/phone/connection")
        allowed = await client.get("/api/v1/phone/connection?operator=test-operator-token")
    assert denied.status_code == 403
    assert denied.json()["detail"] == "Operator access token is required."
    assert allowed.status_code == 200
    assert allowed.json()["mode"] == "public_tunnel"


async def test_public_tunnel_protects_phone_live_and_stream_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (PROJECT_ROOT / "apps" / "web" / "dist" / "index.html").exists():
        pytest.skip("web dist is not built")

    monkeypatch.setenv("ROADSIGN_PUBLIC_BASE_URL", "https://roadsign-demo.trycloudflare.com")
    monkeypatch.setenv("ROADSIGN_DEMO_SECRET", "test-demo-secret")
    monkeypatch.setenv("ROADSIGN_OPERATOR_TOKEN", "test-operator-token")

    local_transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 50000))
    async with httpx.AsyncClient(transport=local_transport, base_url="http://127.0.0.1:8443") as client:
        connection = (await client.get("/api/v1/phone/connection")).json()

    public_transport = httpx.ASGITransport(app=app, client=("203.0.113.10", 50000))
    async with httpx.AsyncClient(
        transport=public_transport,
        base_url="https://roadsign-demo.trycloudflare.com",
    ) as client:
        phone_denied = await client.get(f"/phone?session={connection['session_id']}")
        phone_allowed = await client.get(
            f"/phone?session={connection['session_id']}&access={connection['access_token']}"
        )
        live_denied = await client.get("/live")
        live_allowed = await client.get("/live?operator=test-operator-token")
        streams_denied = await client.get("/api/v1/phone/streams")
        streams_allowed = await client.get("/api/v1/phone/streams?operator=test-operator-token")

    assert phone_denied.status_code == 403
    assert phone_allowed.status_code == 200
    assert live_denied.status_code == 403
    assert live_allowed.status_code == 200
    assert streams_denied.status_code == 403
    assert streams_allowed.status_code == 200


async def test_phone_route_serves_react_app() -> None:
    if not (PROJECT_ROOT / "apps" / "web" / "dist" / "index.html").exists():
        pytest.skip("web dist is not built")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="https://test") as client:
        response = await client.get("/phone?session=test-session")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<!doctype html>" in response.text.lower()


async def test_live_wall_route_is_host_only() -> None:
    if not (PROJECT_ROOT / "apps" / "web" / "dist" / "index.html").exists():
        pytest.skip("web dist is not built")

    host_transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 50000))
    async with httpx.AsyncClient(transport=host_transport, base_url="https://test") as client:
        response = await client.get("/live")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    phone_transport = httpx.ASGITransport(app=app, client=("192.0.2.10", 50000))
    async with httpx.AsyncClient(transport=phone_transport, base_url="https://test") as client:
        response = await client.get("/live")
    assert response.status_code == 403
    assert response.json()["detail"] == "Live camera wall is available only from the host laptop."


async def test_invalid_video_upload_is_cleaned_up() -> None:
    upload_root = PROJECT_ROOT / "outputs" / "uploads"
    before = set(upload_root.glob("*")) if upload_root.exists() else set()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/infer/video",
            files={"file": ("invalid.avi", b"not a video", "video/x-msvideo")},
        )
    after = set(upload_root.glob("*")) if upload_root.exists() else set()
    assert response.status_code == 400
    assert response.json()["detail"] == "Unable to decode video"
    assert after == before
