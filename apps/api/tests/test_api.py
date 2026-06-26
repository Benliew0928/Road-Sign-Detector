# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from pathlib import Path

import cv2
import httpx
import numpy as np
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
    assert body == {"frames_read": 3, "sampled_frames": 1, "events": 0}


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
