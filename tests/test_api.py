"""Tests for inference API request validation and route registration."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference_server.api import InferenceRequest, app, infer


def test_inference_request_rejects_empty_input() -> None:
    with pytest.raises(ValidationError):
        InferenceRequest(text="")


def test_inference_route_is_registered() -> None:
    assert any(route.path == "/infer" for route in app.routes)


def test_infer_handler_returns_embedding_and_model_version() -> None:
    async def scenario() -> None:
        result = await infer(InferenceRequest(text="hello"))
        assert result["model_version"] == "cpu-embedding-v1"
        assert len(result["embedding"]) == 16  # type: ignore[arg-type]

    asyncio.run(scenario())


def test_http_infer_endpoint_returns_validated_response() -> None:
    with TestClient(app) as client:
        response = client.post("/infer", json={"text": "hello"})
        assert response.status_code == 200
        assert response.json()["model_version"] == "cpu-embedding-v1"


def test_http_infer_endpoint_rejects_empty_input() -> None:
    with TestClient(app) as client:
        response = client.post("/infer", json={"text": ""})
        assert response.status_code == 422


def test_http_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
