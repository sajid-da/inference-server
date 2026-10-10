"""Tests for dynamic batching: concurrent requests, ordering, timeouts, shutdown."""

import asyncio
from threading import Event

import pytest

from inference_server.batching import BatchingQueue
from inference_server.model import CpuEmbeddingModel


class SlowBatchModel:
    """Model that blocks until released, for timeout and capacity testing."""

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.batch_sizes: list[int] = []

    def infer_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        self.started.set()
        self.release.wait(timeout=5)
        return [[float(len(t))] for t in texts]


class FailingBatchModel:
    """Model that raises on every batch call."""

    def infer_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("intentional model failure")


class MismatchBatchModel:
    """Model that returns wrong number of results."""

    def infer_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0]]  # always returns 1 result regardless of batch size


# ── basic functionality ────────────────────────────────────────────


def test_batching_returns_correct_embeddings() -> None:
    async def scenario() -> None:
        model = CpuEmbeddingModel()
        bq = BatchingQueue(model, max_batch_size=4, max_wait_seconds=0.1)
        bq.start()

        result = await bq.infer("hello")
        assert len(result) == 16
        # Should match single-inference result
        expected = model.infer("hello")
        assert result == expected

        await bq.close()

    asyncio.run(scenario())


def test_batching_handles_multiple_concurrent_requests() -> None:
    async def scenario() -> None:
        model = CpuEmbeddingModel()
        bq = BatchingQueue(
            model, max_batch_size=4, max_wait_seconds=0.5, capacity=10
        )
        bq.start()

        texts = ["alpha", "beta", "gamma", "delta"]
        tasks = [asyncio.create_task(bq.infer(t, timeout_seconds=5.0)) for t in texts]
        results = await asyncio.gather(*tasks)

        # Each result should match individual inference
        for text, result in zip(texts, results, strict=True):
            expected = model.infer(text)
            assert result == expected, f"Mismatch for '{text}'"

        await bq.close()

    asyncio.run(scenario())


def test_batching_preserves_response_ordering() -> None:
    """Each request gets back its own result, not another request's."""

    async def scenario() -> None:
        model = CpuEmbeddingModel()
        bq = BatchingQueue(
            model, max_batch_size=8, max_wait_seconds=0.5, capacity=20
        )
        bq.start()

        texts = [f"text-{i}" for i in range(10)]
        tasks = [asyncio.create_task(bq.infer(t, timeout_seconds=5.0)) for t in texts]
        results = await asyncio.gather(*tasks)

        for text, result in zip(texts, results, strict=True):
            expected = model.infer(text)
            assert result == expected

        await bq.close()

    asyncio.run(scenario())


# ── configuration validation ───────────────────────────────────────


def test_batching_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="max_batch_size"):
        BatchingQueue(CpuEmbeddingModel(), max_batch_size=0)


def test_batching_rejects_invalid_wait_time() -> None:
    with pytest.raises(ValueError, match="max_wait_seconds"):
        BatchingQueue(CpuEmbeddingModel(), max_wait_seconds=0)


def test_batching_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        BatchingQueue(CpuEmbeddingModel(), capacity=0)


# ── capacity and overload ──────────────────────────────────────────


def test_batching_rejects_requests_before_start() -> None:
    async def scenario() -> None:
        bq = BatchingQueue(CpuEmbeddingModel())
        with pytest.raises(RuntimeError, match="not accepting"):
            await bq.infer("hello")

    asyncio.run(scenario())


def test_batching_rejects_when_full() -> None:
    async def scenario() -> None:
        slow = SlowBatchModel()
        bq = BatchingQueue(
            slow, max_batch_size=1, max_wait_seconds=0.01, capacity=1  # type: ignore[arg-type]
        )
        bq.start()

        # First request will block
        first = asyncio.create_task(bq.infer("first", timeout_seconds=5.0))
        # Wait until model starts processing
        await asyncio.to_thread(slow.started.wait, 2)

        # Second request should be rejected
        with pytest.raises(OverflowError, match="full"):
            await bq.infer("second")

        slow.release.set()
        await first
        await bq.close()

    asyncio.run(scenario())


# ── timeout behavior ──────────────────────────────────────────────


def test_batching_timeout_keeps_capacity_reserved() -> None:
    async def scenario() -> None:
        slow = SlowBatchModel()
        bq = BatchingQueue(
            slow, max_batch_size=1, max_wait_seconds=0.01, capacity=1  # type: ignore[arg-type]
        )
        bq.start()

        task = asyncio.create_task(bq.infer("slow", timeout_seconds=0.01))
        await asyncio.to_thread(slow.started.wait, 2)

        with pytest.raises(TimeoutError, match="remains admitted"):
            await task

        # Capacity is still held
        with pytest.raises(OverflowError, match="full"):
            await bq.infer("second")

        slow.release.set()
        await bq.close()

    asyncio.run(scenario())


# ── error handling ─────────────────────────────────────────────────


def test_batching_propagates_model_errors() -> None:
    async def scenario() -> None:
        bq = BatchingQueue(
            FailingBatchModel(),  # type: ignore[arg-type]
            max_batch_size=1,
            max_wait_seconds=0.01,
        )
        bq.start()

        with pytest.raises(RuntimeError, match="intentional"):
            await bq.infer("fail", timeout_seconds=2.0)

        await bq.close()

    asyncio.run(scenario())


def test_batching_handles_result_count_mismatch() -> None:
    async def scenario() -> None:
        bq = BatchingQueue(
            MismatchBatchModel(),  # type: ignore[arg-type]
            max_batch_size=4,
            max_wait_seconds=0.01,
        )
        bq.start()

        tasks = [
            asyncio.create_task(bq.infer(f"text-{i}", timeout_seconds=2.0))
            for i in range(3)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        # All should get RuntimeError about result count mismatch
        for r in results:
            assert isinstance(r, RuntimeError)
            assert "results" in str(r)

        await bq.close()

    asyncio.run(scenario())


# ── shutdown ──────────────────────────────────────────────────────


def test_batching_drains_on_close() -> None:
    async def scenario() -> None:
        model = CpuEmbeddingModel()
        bq = BatchingQueue(model, max_batch_size=4, max_wait_seconds=0.5)
        bq.start()

        # Submit work then immediately close
        tasks = [
            asyncio.create_task(bq.infer(f"drain-{i}", timeout_seconds=5.0))
            for i in range(3)
        ]

        # Brief yield to ensure items are queued
        await asyncio.sleep(0.01)
        await bq.close()

        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, list)

    asyncio.run(scenario())


def test_batching_rejects_after_close() -> None:
    async def scenario() -> None:
        bq = BatchingQueue(CpuEmbeddingModel())
        bq.start()
        await bq.close()

        with pytest.raises(RuntimeError, match="not accepting"):
            await bq.infer("too-late")

    asyncio.run(scenario())


# ── batch size enforcement ─────────────────────────────────────────


def test_batching_respects_max_batch_size() -> None:
    """Verify that no single batch exceeds max_batch_size."""

    class TrackingModel:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def infer_batch(self, texts: list[str]) -> list[list[float]]:
            self.batch_sizes.append(len(texts))
            return [[1.0] * 16 for _ in texts]

    async def scenario() -> None:
        tracker = TrackingModel()
        bq = BatchingQueue(
            tracker,  # type: ignore[arg-type]
            max_batch_size=3,
            max_wait_seconds=0.01,
            capacity=20,
        )
        bq.start()

        tasks = [
            asyncio.create_task(bq.infer(f"t-{i}", timeout_seconds=5.0))
            for i in range(7)
        ]
        await asyncio.gather(*tasks)
        await bq.close()

        # No batch should exceed max_batch_size
        for bs in tracker.batch_sizes:
            assert bs <= 3, f"Batch size {bs} exceeds max of 3"

    asyncio.run(scenario())


# ── advanced shutdown & edge cases ─────────────────────────────────


def test_batching_shutdown_timeout_and_forced_cancellation() -> None:
    """Worker hanging on model call is cancelled when close timeout expires."""

    class HangingModel:
        def __init__(self) -> None:
            self.started = Event()
            self.never_released = Event()

        def infer_batch(self, texts: list[str]) -> list[list[float]]:
            self.started.set()
            self.never_released.wait(timeout=10)
            return [[1.0] * 16 for _ in texts]

    async def scenario() -> None:
        model = HangingModel()
        bq = BatchingQueue(
            model,  # type: ignore[arg-type]
            max_batch_size=1,
            max_wait_seconds=0.01,
            capacity=5,
        )
        bq.start()

        task = asyncio.create_task(bq.infer("hang", timeout_seconds=10.0))
        await asyncio.to_thread(model.started.wait, 2)

        # close() with short timeout should force cancel worker task
        await bq.close(timeout_seconds=0.05)

        assert not bq.is_running
        with pytest.raises((RuntimeError, TimeoutError, asyncio.CancelledError)):
            await task

    asyncio.run(scenario())


def test_batching_shutdown_repeated_idempotent() -> None:
    """Repeated calls to close() are safe and idempotent."""

    async def scenario() -> None:
        model = CpuEmbeddingModel()
        bq = BatchingQueue(model, max_batch_size=2, max_wait_seconds=0.01)
        bq.start()

        await bq.close()
        await bq.close()
        await bq.close()

        assert not bq.is_running

    asyncio.run(scenario())


def test_batching_shutdown_with_worker_failure() -> None:
    """close() handles a worker that already crashed due to unhandled exception."""

    async def scenario() -> None:
        bq = BatchingQueue(
            FailingBatchModel(),  # type: ignore[arg-type]
            max_batch_size=1,
            max_wait_seconds=0.01,
        )
        bq.start()

        with pytest.raises(RuntimeError, match="intentional"):
            await bq.infer("fail", timeout_seconds=2.0)

        # Worker may be done due to failure
        await bq.close()
        assert not bq.is_running

    asyncio.run(scenario())


def test_batching_testclient_context_manager_exit() -> None:
    """FastAPI TestClient enters and exits lifespan cleanly multiple times."""
    from fastapi.testclient import TestClient

    from inference_server.api import app

    for _ in range(3):
        with TestClient(app) as client:
            res = client.get("/health")
            assert res.status_code == 200
            res = client.post("/infer", json={"text": "testclient lifespan"})
            assert res.status_code == 200
            assert "embedding" in res.json()

