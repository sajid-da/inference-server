"""HTTP entry point."""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from inference_server.model import CpuEmbeddingModel
from inference_server.queue import InferenceQueue
from inference_server.service import NaiveInferenceService

app = FastAPI(title="Inference Server")
queue = InferenceQueue(NaiveInferenceService(CpuEmbeddingModel()))


class InferenceRequest(BaseModel):
    """Validated request body for the local CPU model."""

    text: str = Field(min_length=1, max_length=10_000)
    timeout_seconds: float = Field(default=1.0, gt=0, le=30)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a liveness response."""
    return {"status": "ok"}


@app.post("/infer")
async def infer(payload: InferenceRequest) -> dict[str, object]:
    """Queue and execute one inference request."""
    if not queue.is_running:
        queue.start()
    try:
        embedding = await queue.infer(payload.text, payload.timeout_seconds)
    except OverflowError as error:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS) from error
    except TimeoutError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY) from error
    return {"embedding": embedding, "model_version": "cpu-embedding-v1"}
