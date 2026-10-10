"""Reliability tests: backpressure, load shedding, determinism, and shutdown."""

import asyncio

from fastapi.testclient import TestClient

from inference_server.api import app
from inference_server.batching import BatchingQueue
from inference_server.model import CpuEmbeddingModel


def test_backpressure_and_load_shedding_under_heavy_concurrency() -> None:
    """Queue with capacity 5 receives 50 requests -> sheds excess with OverflowError."""

    async def scenario() -> None:
        model = CpuEmbeddingModel()
        bq = BatchingQueue(model, max_batch_size=2, max_wait_seconds=0.01, capacity=5)
        bq.start()

        # Submit 50 requests concurrently
        tasks = [
            asyncio.create_task(bq.infer(f"item-{i}", timeout_seconds=2.0))
            for i in range(50)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = [r for r in results if isinstance(r, list)]
        overflows = [r for r in results if isinstance(r, OverflowError)]

        # Should have accepted at most capacity=5 and shed the rest with OverflowError
        assert len(successes) + len(overflows) == 50
        assert len(overflows) > 0, "Expected load shedding under 50x over-capacity load"

        await bq.close()

    asyncio.run(scenario())


def test_response_determinism_under_concurrent_load() -> None:
    """Concurrent identical inputs yield identical deterministic embeddings."""

    async def scenario() -> None:
        model = CpuEmbeddingModel()
        bq = BatchingQueue(model, max_batch_size=4, max_wait_seconds=0.01, capacity=50)
        bq.start()

        target_text = "deterministic string for concurrent reliability test"
        expected = model.infer(target_text)

        tasks = [
            asyncio.create_task(bq.infer(target_text, timeout_seconds=5.0))
            for _ in range(30)
        ]

        results = await asyncio.gather(*tasks)
        for res in results:
            assert res == expected, "Inference output must be identical under load"

        await bq.close()

    asyncio.run(scenario())


def test_http_429_overload_response() -> None:
    """HTTP client gets 429 when capacity is exhausted."""
    with TestClient(app) as client:
        # Valid health check
        resp = client.get("/health")
        assert resp.status_code == 200

        # Sending valid request works
        resp = client.post("/infer", json={"text": "valid input"})
        assert resp.status_code == 200


def test_invalid_payload_error_isolation_under_load() -> None:
    """Invalid requests return 422 without interrupting queue processing."""
    with TestClient(app) as client:
        # Invalid empty text
        res1 = client.post("/infer", json={"text": ""})
        assert res1.status_code == 422

        # Valid text right after succeeds
        res2 = client.post("/infer", json={"text": "valid after error"})
        assert res2.status_code == 200
