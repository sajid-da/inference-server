"""HTTP entry point."""

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from inference_server.batching import BatchingQueue
from inference_server.canary import CanaryRouter
from inference_server.metrics import Metrics
from inference_server.model import CpuEmbeddingModel
from inference_server.queue import InferenceQueue
from inference_server.registry import ModelHandle, ModelRegistry
from inference_server.service import NaiveInferenceService

# Default models, registry, router, and metrics setup (I6, I7, I9)
default_handle = ModelHandle("cpu-embedding-v1", CpuEmbeddingModel())
registry = ModelRegistry(default_handle)
canary_router = CanaryRouter(registry)
metrics = Metrics()

service = NaiveInferenceService(default_handle.model)
queue = InferenceQueue(service)

batching_queue = BatchingQueue(
    default_handle.model,
    max_batch_size=8,
    max_wait_seconds=0.05,
    capacity=100,
)

USE_BATCHING = True


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Start queues on startup, drain on shutdown."""
    if USE_BATCHING:
        batching_queue.start()
    else:
        queue.start()
    yield
    if USE_BATCHING:
        await batching_queue.close()
    else:
        await queue.close()


app = FastAPI(title="Inference Server", lifespan=lifespan)


class InferenceRequest(BaseModel):
    """Validated request body for the local CPU model."""

    text: str = Field(min_length=1, max_length=10_000)
    timeout_seconds: float = Field(default=1.0, gt=0, le=30)


class CanaryControlRequest(BaseModel):
    """Payload for configuring canary deployment."""

    version: str = Field(min_length=1, max_length=100)
    weight: float = Field(ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a liveness response."""
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Expose operational counters in Prometheus text format."""
    return Response(metrics.render(), media_type="text/plain; version=0.0.4")


@app.get("/models/active")
def get_active_models() -> dict[str, object]:
    """Return currently active primary and canary model metadata."""
    primary = registry.active()
    canary = canary_router.canary_handle
    config = canary_router.config
    return {
        "primary_version": primary.version,
        "canary_version": canary.version if canary else None,
        "canary_weight": config.canary_weight,
    }


@app.post("/models/canary")
def set_canary(payload: CanaryControlRequest) -> dict[str, object]:
    """Configure a new canary model version and traffic split weight."""
    canary_handle = ModelHandle(payload.version, CpuEmbeddingModel())
    canary_router.set_canary(canary_handle, payload.weight)
    return {
        "status": "canary_configured",
        "canary_version": payload.version,
        "weight": payload.weight,
    }


@app.post("/models/rollback")
def rollback_canary() -> dict[str, str]:
    """Roll back canary deployment immediately to 100% primary traffic."""
    canary_router.rollback()
    return {"status": "rolled_back", "active_version": registry.active().version}


@app.post("/models/promote")
def promote_canary() -> dict[str, str]:
    """Promote current canary to become the active primary model."""
    try:
        canary_router.promote_canary()
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    return {"status": "promoted", "active_version": registry.active().version}


@app.post("/infer")
async def infer(payload: InferenceRequest) -> dict[str, object]:
    """Queue and execute one inference request with model version routing."""
    start_time = time.perf_counter()
    handle = canary_router.select_handle()
    try:
        if USE_BATCHING:
            if not batching_queue.is_running:
                batching_queue.start()
            embedding = await batching_queue.infer(
                payload.text, payload.timeout_seconds
            )
        else:
            if not queue.is_running:
                queue.start()
            embedding = await queue.infer(
                payload.text, payload.timeout_seconds
            )
        duration = time.perf_counter() - start_time
        metrics.record_request(200, handle.version, duration)
    except OverflowError as error:
        metrics.record_overflow()
        metrics.record_request(429, handle.version)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS
        ) from error
    except TimeoutError as error:
        metrics.record_timeout()
        metrics.record_request(503, handle.version)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        ) from error
    except ValueError as error:
        metrics.record_request(422, handle.version)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        ) from error
    return {"embedding": embedding, "model_version": handle.version}
