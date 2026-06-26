from __future__ import annotations

# pyright: reportUnusedFunction=false
import asyncio
import base64
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, cast

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from roadsign_api.dependencies import get_engine
from roadsign_assist import __version__
from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.catalogue.models import SignCatalogue
from roadsign_assist.catalogue.repository import load_catalogue
from roadsign_assist.diagnostics import collect_diagnostics
from roadsign_assist.inference.engine import annotate_frame, decode_image, encode_jpeg
from roadsign_assist.inference.models import FrameResultModel, InferenceMode
from roadsign_assist.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


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
    classifier: str
    classifier_available: bool
    classifier_loaded: bool
    classifier_providers: list[str] = Field(default_factory=list)
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
                    event_count += len(result.events)
                frames += 1
        finally:
            capture.release()
            temp_path.unlink(missing_ok=True)
        return VideoInferenceResponse(
            frames_read=frames,
            sampled_frames=(frames + 2) // 3,
            events=event_count,
        )

    @application.websocket("/api/v1/ws/camera/{session_id}")
    async def camera_socket(websocket: WebSocket, session_id: str) -> None:
        del session_id
        await websocket.accept()
        engine = get_engine().new_session()
        try:
            while True:
                data = await websocket.receive_bytes()
                if len(data) > MAX_UPLOAD_BYTES:
                    await websocket.send_json({"error": "Frame exceeds 20 MB"})
                    continue
                try:
                    image = decode_image(data)
                    result = await asyncio.to_thread(engine.process_frame, image)
                    await websocket.send_json(result.model_dump(mode="json"))
                except ValueError as exc:
                    await websocket.send_json({"error": str(exc)})
        except WebSocketDisconnect:
            LOGGER.info("Camera WebSocket disconnected")

    web_dist = PROJECT_ROOT / "apps" / "web" / "dist"
    if web_dist.exists():
        application.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return application


app = create_app()
