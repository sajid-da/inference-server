"""HTTP entry point."""

from fastapi import FastAPI

app = FastAPI(title="Inference Server")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a liveness response."""
    return {"status": "ok"}
