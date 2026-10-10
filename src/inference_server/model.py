"""Small deterministic CPU model used to exercise the serving layer."""

import math


class CpuEmbeddingModel:
    """Convert text to a stable, normalized 16-dimensional feature vector."""

    dimension = 16

    def infer(self, text: str) -> list[float]:
        """Run one request synchronously, as the intentionally naive baseline."""
        if not text:
            raise ValueError("text must not be empty")
        vector = [0.0] * self.dimension
        for position, character in enumerate(text.encode("utf-8")):
            vector[position % self.dimension] += character / 255
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def infer_batch(self, texts: list[str]) -> list[list[float]]:
        """Run inference on a batch of texts. Returns one embedding per input."""
        return [self.infer(text) for text in texts]
