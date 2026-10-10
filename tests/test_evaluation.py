"""Tests for model-variant comparison primitives."""

import pytest

from inference_server.evaluation import cosine_similarity, evaluate_variant
from inference_server.model import CpuEmbeddingModel


def test_evaluation_reports_perfect_agreement_for_identical_models() -> None:
    result = evaluate_variant(
        "baseline", CpuEmbeddingModel(), CpuEmbeddingModel(), ["one", "two"]
    )

    assert result.requests == 2
    assert result.mean_cosine_similarity == pytest.approx(1.0)
    assert result.elapsed_seconds >= 0


def test_cosine_similarity_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        cosine_similarity([1.0], [1.0, 0.0])
