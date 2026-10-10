"""A bounded single-worker queue with explicit timeout and shutdown semantics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from inference_server.service import NaiveInferenceService


@dataclass
class PendingInference:
    """One admitted request and the future that receives its result."""

    text: str
    response: asyncio.Future[list[float]]


class InferenceQueue:
    """Serialize CPU work while bounding queued plus in-flight requests.

    A client timeout cancels only the client's wait. Its admitted work remains
    counted until the single worker finishes it, which prevents capacity from
    being silently exceeded by detached CPU work.
    """

    def __init__(self, service: NaiveInferenceService, capacity: int = 100) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._service = service
        self._max_capacity = capacity
        self._requests: asyncio.Queue[PendingInference | None] = asyncio.Queue()
        self._capacity = asyncio.BoundedSemaphore(capacity)
        self._worker: asyncio.Task[None] | None = None
        self._accepting = False

    def start(self) -> None:
        """Start the single worker in the current event loop."""
        if self._worker is not None and not self._worker.done():
            return
        self._requests = asyncio.Queue()
        self._capacity = asyncio.BoundedSemaphore(self._max_capacity)
        self._accepting = True
        self._worker = asyncio.create_task(self._run(), name="inference-worker")

    @property
    def is_running(self) -> bool:
        """Whether a worker exists and is currently running."""
        return self._worker is not None and not self._worker.done()

    async def infer(self, text: str, timeout_seconds: float = 1.0) -> list[float]:
        """Admit one request or reject it when capacity is exhausted."""
        if not self._accepting:
            raise RuntimeError("inference queue is not accepting requests")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self._capacity.locked():
            raise OverflowError("inference queue is full")

        await self._capacity.acquire()
        response: asyncio.Future[list[float]] = (
            asyncio.get_running_loop().create_future()
        )
        await self._requests.put(PendingInference(text=text, response=response))
        try:
            return await asyncio.wait_for(asyncio.shield(response), timeout_seconds)
        except TimeoutError as error:
            message = "inference request timed out while work remains admitted"
            raise TimeoutError(message) from error

    async def close(self) -> None:
        """Stop admission and drain work already admitted before returning."""
        if self._worker is None:
            self._accepting = False
            return
        self._accepting = False
        await self._requests.put(None)
        await self._worker
        self._worker = None

    async def _run(self) -> None:
        while True:
            pending = await self._requests.get()
            if pending is None:
                self._requests.task_done()
                return
            try:
                # The worker awaits the CPU call to completion. A client timeout
                # never releases its admission slot early, so detached work
                # cannot silently exceed configured capacity.
                result = await asyncio.to_thread(self._service.infer, pending.text)
                if not pending.response.done():
                    pending.response.set_result(result)
            except ValueError as error:
                if not pending.response.done():
                    pending.response.set_exception(error)
            finally:
                self._capacity.release()
                self._requests.task_done()
