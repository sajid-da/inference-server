"""Integration adapter providing compatibility with VectorDB + RAG."""

from __future__ import annotations

from typing import Any

import httpx


class RemoteInferenceClient:
    """Client for calling the Inference Server POST /infer HTTP API."""

    def __init__(
        self, base_url: str = "http://127.0.0.1:8000", timeout_seconds: float = 10.0
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def embed(self, text: str) -> list[float]:
        """Embed one string via HTTP POST /infer."""
        url = f"{self._base_url}/infer"
        response = httpx.post(
            url,
            json={"text": text, "timeout_seconds": self._timeout_seconds},
            timeout=self._timeout_seconds + 2.0,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data["embedding"]  # type: ignore[no-any-return]


class InProcessEmbeddingAdapter:
    """In-process adapter fulfilling VectorDB + RAG EmbeddingProvider protocol."""

    def __init__(self, model_or_queue: Any) -> None:
        self._target = model_or_queue

    def embed(self, text: str) -> list[float]:
        """Return embedding list of floats."""
        return self._target.infer(text)  # type: ignore[no-any-return]
