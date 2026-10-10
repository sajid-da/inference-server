"""Local evaluation primitives for CPU model-variant comparisons."""

import math
import time
from dataclasses import dataclass
from typing import Protocol

from inference_server.model import CpuEmbeddingModel


class EmbeddingModel(Protocol):
    """Minimal model contract shared by production and evaluation code."""

    def infer(self, text: str) -> list[float]:
        """Return one embedding."""


@dataclass(frozen=True)
class VariantResult:
    """Measured latency and agreement with a reference model."""

    name: str
    mean_cosine_similarity: float
    elapsed_seconds: float
    requests: int


class SimulatedQuantizedModel:
    """Simulated 8-bit quantized CPU embedding model variant for precision testing."""

    dimension = 16

    def __init__(self, base_model: CpuEmbeddingModel | None = None) -> None:
        self._base_model = base_model or CpuEmbeddingModel()

    def infer(self, text: str) -> list[float]:
        """Run inference and apply uniform INT8 quantization/dequantization."""
        raw_vector = self._base_model.infer(text)
        # Scale float values [-1.0, 1.0] -> INT8 [-127, 127] and dequantize
        quantized = [round(v * 127.0) for v in raw_vector]
        dequantized = [q / 127.0 for q in quantized]
        # Re-normalize
        norm = math.sqrt(sum(v * v for v in dequantized))
        return [v / norm for v in dequantized] if norm > 0 else dequantized


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for equally shaped non-zero vectors."""
    if len(left) != len(right):
        raise ValueError("vectors must have equal dimensions")
    denominator = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    if denominator == 0:
        raise ValueError("vectors must be non-zero")
    return sum(x * y for x, y in zip(left, right, strict=True)) / denominator


def evaluate_variant(
    name: str, reference: EmbeddingModel, candidate: EmbeddingModel, texts: list[str]
) -> VariantResult:
    """Measure candidate latency and embedding agreement on a fixed local set."""
    if not texts:
        raise ValueError("at least one evaluation text is required")
    start = time.perf_counter()
    similarities = [
        cosine_similarity(reference.infer(text), candidate.infer(text))
        for text in texts
    ]
    elapsed = time.perf_counter() - start
    return VariantResult(
        name=name,
        mean_cosine_similarity=sum(similarities) / len(similarities),
        elapsed_seconds=elapsed,
        requests=len(texts),
    )
