from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from roadsign_api.main import app


def _json_schema_ref(schema: Mapping[str, Any], path: str, method: str) -> str:
    operation = schema["paths"][path][method]
    response = operation["responses"]["200"]
    return str(response["content"]["application/json"]["schema"].get("$ref", ""))


def test_openapi_exposes_typed_p13_http_contracts() -> None:
    schema = app.openapi()
    expected_refs = {
        ("/api/v1/health", "get"): "#/components/schemas/HealthResponse",
        ("/api/v1/catalogue", "get"): "#/components/schemas/SignCatalogue",
        ("/api/v1/models", "get"): "#/components/schemas/ModelStatusResponse",
        ("/api/v1/infer/image", "post"): "#/components/schemas/ImageInferenceResponse",
        ("/api/v1/infer/batch", "post"): "#/components/schemas/BatchInferenceResponse",
        ("/api/v1/infer/video", "post"): "#/components/schemas/VideoInferenceResponse",
    }
    assert set(schema["paths"]) >= {path for path, _ in expected_refs}
    for (path, method), expected_ref in expected_refs.items():
        assert _json_schema_ref(schema, path, method) == expected_ref


def test_openapi_contains_runtime_fields_used_by_react() -> None:
    schemas = app.openapi()["components"]["schemas"]
    diagnostics = schemas["DiagnosticsResponse"]["properties"]
    model_status = schemas["ModelStatusResponse"]["properties"]
    assert "healthy" in diagnostics
    for field in (
        "detector_loaded",
        "detector_device",
        "classifier_loaded",
        "classifier_providers",
        "ocr_loaded",
        "ocr_load_error",
    ):
        assert field in model_status
