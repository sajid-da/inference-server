"""Tests for deterministic baseline inference."""

import pytest

from inference_server.model import CpuEmbeddingModel
from inference_server.service import NaiveInferenceService


def test_inference_is_deterministic_and_normalized() -> None:
    result = NaiveInferenceService(CpuEmbeddingModel()).infer("hello")

    assert result == NaiveInferenceService(CpuEmbeddingModel()).infer("hello")
    assert sum(value * value for value in result) == pytest.approx(1.0)


def test_inference_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="empty"):
        CpuEmbeddingModel().infer("")
