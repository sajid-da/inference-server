"""Tests for VectorDB + RAG integration client and adapter."""

from fastapi.testclient import TestClient

from inference_server.api import app
from inference_server.integration import (
    InProcessEmbeddingAdapter,
)
from inference_server.model import CpuEmbeddingModel


def test_in_process_embedding_adapter() -> None:
    model = CpuEmbeddingModel()
    adapter = InProcessEmbeddingAdapter(model)

    embedding = adapter.embed("rag query text")
    assert len(embedding) == 16
    assert isinstance(embedding, list)


def test_remote_inference_client_with_test_client() -> None:
    with TestClient(app) as client:
        response = client.post("/infer", json={"text": "test remote client"})
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert len(data["embedding"]) == 16
        assert data["model_version"] == "cpu-embedding-v1"


def test_end_to_end_rag_integration_contract() -> None:
    """Verify that InProcessEmbeddingAdapter fulfills the RAG embedding interface."""
    model = CpuEmbeddingModel()
    adapter = InProcessEmbeddingAdapter(model)

    # Embedding provider contract: embed(text: str) returns 16-element float vector
    vec1 = adapter.embed("query 1")
    vec2 = adapter.embed("query 2")

    assert len(vec1) == 16
    assert len(vec2) == 16
    assert vec1 != vec2
