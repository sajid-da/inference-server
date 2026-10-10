"""Naive one-request-at-a-time inference service."""

from inference_server.model import CpuEmbeddingModel


class NaiveInferenceService:
    """Serve a model synchronously for I1 baseline measurements."""

    def __init__(self, model: CpuEmbeddingModel) -> None:
        self._model = model

    def infer(self, text: str) -> list[float]:
        """Run exactly one inference request."""
        return self._model.infer(text)
