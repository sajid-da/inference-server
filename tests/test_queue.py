"""Tests for bounded admission, timeouts, and graceful queue draining."""

import asyncio
from threading import Event

import pytest

from inference_server.model import CpuEmbeddingModel
from inference_server.queue import InferenceQueue
from inference_server.service import NaiveInferenceService


def test_queue_returns_one_inference_result() -> None:
    async def scenario() -> None:
        queue = InferenceQueue(NaiveInferenceService(CpuEmbeddingModel()), capacity=1)
        queue.start()
        assert len(await queue.infer("queued")) == 16
        await queue.close()

    asyncio.run(scenario())


def test_queue_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        InferenceQueue(NaiveInferenceService(CpuEmbeddingModel()), capacity=0)


def test_timeout_keeps_capacity_reserved_until_cpu_work_finishes() -> None:
    class BlockingService:
        def __init__(self) -> None:
            self.started = Event()
            self.release = Event()

        def infer(self, text: str) -> list[float]:
            self.started.set()
            self.release.wait(timeout=1)
            return [1.0]

    async def scenario() -> None:
        service = BlockingService()
        queue = InferenceQueue(service, capacity=1)  # type: ignore[arg-type]
        queue.start()
        first = asyncio.create_task(queue.infer("first", timeout_seconds=0.01))
        assert await asyncio.to_thread(service.started.wait, 1)
        with pytest.raises(TimeoutError, match="remains admitted"):
            await first
        with pytest.raises(OverflowError, match="full"):
            await queue.infer("second")
        service.release.set()
        await queue.close()

    asyncio.run(scenario())


def test_queue_rejects_requests_before_start() -> None:
    async def scenario() -> None:
        queue = InferenceQueue(NaiveInferenceService(CpuEmbeddingModel()))
        with pytest.raises(RuntimeError, match="not accepting"):
            await queue.infer("queued")

    asyncio.run(scenario())
