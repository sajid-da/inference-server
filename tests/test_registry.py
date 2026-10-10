"""Tests for local model versioning and atomic replacement."""

import pytest

from inference_server.model import CpuEmbeddingModel
from inference_server.registry import ModelHandle, ModelRegistry


def test_registry_replaces_active_model_version() -> None:
    registry = ModelRegistry(ModelHandle("v1", CpuEmbeddingModel()))

    registry.replace(ModelHandle("v2", CpuEmbeddingModel()))

    assert registry.active().version == "v2"


def test_registry_rejects_empty_versions() -> None:
    registry = ModelRegistry(ModelHandle("v1", CpuEmbeddingModel()))

    with pytest.raises(ValueError, match="version"):
        registry.replace(ModelHandle("", CpuEmbeddingModel()))
