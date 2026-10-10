"""Local evaluation primitives for future CPU model-variant comparisons."""

import math
import time
from dataclasses import dataclass
from typing import Protocol


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
