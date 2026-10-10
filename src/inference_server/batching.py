"""Dynamic batching: accumulate requests up to a size or time limit."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


class BatchableModel(Protocol):
    """A model that can process a batch of inputs at once."""

    def infer_batch(self, texts: list[str]) -> list[list[float]]:
        """Run inference on a batch of texts. Returns one embedding per input."""


@dataclass
class _PendingItem:
    """One request waiting to join a batch."""

    text: str
    future: asyncio.Future[list[float]]


class BatchingQueue:
    """Accumulate requests into batches bounded by size and wait time.

    Requests are collected until either *max_batch_size* items have arrived or
    *max_wait_seconds* have elapsed since the first item in the current batch,
    whichever comes first. The batch is then dispatched to the model in a
    worker thread so the event loop stays responsive.

    Capacity is bounded: at most *capacity* requests may be queued or in-flight
    at any time. Excess requests receive an ``OverflowError`` immediately.
    A client-side timeout raises ``TimeoutError`` but leaves the admitted slot
    occupied until the model finishes processing it (same semantics as
    ``InferenceQueue``).
    """

    def __init__(
        self,
        model: BatchableModel,
        *,
        max_batch_size: int = 8,
        max_wait_seconds: float = 0.05,
        capacity: int = 100,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be positive")
        if capacity < 1:
            raise ValueError("capacity must be at least 1")

        self._model = model
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_seconds
        self._capacity = capacity

        self._semaphore: asyncio.BoundedSemaphore | None = None
        self._pending: list[_PendingItem] = []
        self._pending_lock: asyncio.Lock | None = None
        self._batch_event: asyncio.Event | None = None
        self._worker: asyncio.Task[None] | None = None
        self._accepting = False
        self._timer_task: asyncio.Task[None] | None = None

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        """Start the batching worker in the current event loop."""
        if self._worker is not None and not self._worker.done():
            return
        self._semaphore = asyncio.BoundedSemaphore(self._capacity)
        self._pending_lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        self._pending = []
        self._accepting = True
        self._worker = asyncio.create_task(self._run(), name="batch-worker")

    @property
    def is_running(self) -> bool:
        """Whether the batching worker is active."""
        return self._worker is not None and not self._worker.done()

    async def close(self, timeout_seconds: float = 5.0) -> None:
        """Stop accepting requests, signal worker, and drain pending work safely."""
        self._accepting = False

        if self._timer_task is not None and not self._timer_task.done():
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
            self._timer_task = None

        if self._worker is None:
            self._reject_pending(RuntimeError("batching queue is closed"))
            return

        if self._worker.done():
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
            self._reject_pending(RuntimeError("batching queue is closed"))
            self._worker = None
            return

        if self._batch_event is not None:
            self._batch_event.set()

        worker_task = self._worker
        try:
            await asyncio.wait_for(
                asyncio.shield(worker_task), timeout=timeout_seconds
            )
        except TimeoutError:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            if not worker_task.done():
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
            raise
        except Exception:
            pass
        finally:
            self._worker = None
            self._reject_pending(RuntimeError("batching queue is closed"))

    def _reject_pending(self, exc: Exception) -> None:
        """Fail all pending items and release semaphore units."""
        if self._pending:
            remaining = list(self._pending)
            self._pending.clear()
            for item in remaining:
                if not item.future.done():
                    item.future.set_exception(exc)
                if self._semaphore is not None:
                    try:
                        self._semaphore.release()
                    except ValueError:
                        pass

    # ── public API ──────────────────────────────────────────────────

    async def infer(self, text: str, timeout_seconds: float = 1.0) -> list[float]:
        """Submit one inference request and wait for the batched result."""
        if not self._accepting or self._worker is None or self._worker.done():
            raise RuntimeError("batching queue is not accepting requests")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        assert self._semaphore is not None
        assert self._pending_lock is not None
        assert self._batch_event is not None

        if self._semaphore.locked():
            raise OverflowError("batching queue is full")

        await self._semaphore.acquire()
        future: asyncio.Future[list[float]] = (
            asyncio.get_running_loop().create_future()
        )
        item = _PendingItem(text=text, future=future)

        async with self._pending_lock:
            self._pending.append(item)
            if len(self._pending) >= self._max_batch_size:
                self._batch_event.set()
            elif len(self._pending) == 1:
                self._start_timer()

        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout_seconds
            )
        except TimeoutError as error:
            msg = "inference request timed out while work remains admitted"
            raise TimeoutError(msg) from error

    # ── internals ───────────────────────────────────────────────────

    def _start_timer(self) -> None:
        """Start a timer that will trigger batch dispatch after max_wait."""
        if self._timer_task is not None and not self._timer_task.done():
            self._timer_task.cancel()

        async def _timer() -> None:
            try:
                await asyncio.sleep(self._max_wait_seconds)
                if self._batch_event is not None:
                    self._batch_event.set()
            except asyncio.CancelledError:
                pass

        self._timer_task = asyncio.create_task(_timer(), name="batch-timer")

    async def _run(self) -> None:
        """Worker loop: wait for batch signals, collect items, dispatch."""
        assert self._batch_event is not None
        assert self._pending_lock is not None
        assert self._semaphore is not None

        try:
            while True:
                await self._batch_event.wait()
                self._batch_event.clear()

                async with self._pending_lock:
                    batch = self._pending[: self._max_batch_size]
                    self._pending = self._pending[self._max_batch_size :]

                    if self._pending:
                        self._batch_event.set()

                if batch:
                    await self._dispatch_batch(batch)

                if not self._accepting:
                    async with self._pending_lock:
                        has_more = len(self._pending) > 0
                    if not has_more:
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            raise
        finally:
            self._reject_pending(RuntimeError("batching queue worker stopped"))

    async def _dispatch_batch(self, batch: list[_PendingItem]) -> None:
        """Run model on batch and satisfy or fail futures."""
        assert self._semaphore is not None
        texts = [item.text for item in batch]
        try:
            results = await asyncio.to_thread(self._model.infer_batch, texts)
            if len(results) != len(batch):
                exc = RuntimeError(
                    f"model returned {len(results)} results for {len(batch)} inputs"
                )
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
                    self._semaphore.release()
                return

            for item, result in zip(batch, results, strict=True):
                if not item.future.done():
                    item.future.set_result(result)
                self._semaphore.release()
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
                self._semaphore.release()
