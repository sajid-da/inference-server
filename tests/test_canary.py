"""Tests for I6 & I7: Model Registry, Canary Deployment, Routing, and Rollback."""

import pytest
from fastapi.testclient import TestClient

from inference_server.api import app
from inference_server.canary import CanaryConfig, CanaryRouter
from inference_server.model import CpuEmbeddingModel
from inference_server.registry import ModelHandle, ModelRegistry


def test_canary_config_validation() -> None:
    with pytest.raises(ValueError, match="canary_weight"):
        CanaryConfig(canary_weight=-0.1)

    with pytest.raises(ValueError, match="canary_weight"):
        CanaryConfig(canary_weight=1.5)


def test_canary_routing_traffic_split() -> None:
    v1 = ModelHandle("v1", CpuEmbeddingModel())
    v2 = ModelHandle("v2", CpuEmbeddingModel())
    reg = ModelRegistry(v1)
    router = CanaryRouter(reg)

    router.set_canary(v2, weight=0.3)

    # Seed < 0.3 routes to canary (v2)
    assert router.select_handle(seed=0.1).version == "v2"
    assert router.select_handle(seed=0.29).version == "v2"

    # Seed >= 0.3 routes to primary (v1)
    assert router.select_handle(seed=0.3).version == "v1"
    assert router.select_handle(seed=0.8).version == "v1"


def test_canary_rollback_diverts_all_traffic_to_primary() -> None:
    v1 = ModelHandle("v1", CpuEmbeddingModel())
    v2 = ModelHandle("v2", CpuEmbeddingModel())
    reg = ModelRegistry(v1)
    router = CanaryRouter(reg)

    router.set_canary(v2, weight=0.5)
    assert router.canary_handle is not None

    router.rollback()

    assert router.canary_handle is None
    assert router.config.canary_weight == 0.0
    assert router.select_handle(seed=0.1).version == "v1"


def test_canary_promotion_swaps_primary_in_registry() -> None:
    v1 = ModelHandle("v1", CpuEmbeddingModel())
    v2 = ModelHandle("v2-canary", CpuEmbeddingModel())
    reg = ModelRegistry(v1)
    router = CanaryRouter(reg)

    router.set_canary(v2, weight=0.2)
    router.promote_canary()

    assert reg.active().version == "v2-canary"
    assert router.canary_handle is None
    assert router.config.canary_weight == 0.0


def test_canary_promotion_fails_when_no_canary_set() -> None:
    v1 = ModelHandle("v1", CpuEmbeddingModel())
    reg = ModelRegistry(v1)
    router = CanaryRouter(reg)

    with pytest.raises(RuntimeError, match="no active canary"):
        router.promote_canary()


def test_http_canary_endpoints() -> None:
    """Test HTTP administration endpoints for canary management."""
    with TestClient(app) as client:
        # Check active models
        res = client.get("/models/active")
        assert res.status_code == 200
        assert res.json()["primary_version"] == "cpu-embedding-v1"

        # Configure canary
        res = client.post("/models/canary", json={"version": "v2-test", "weight": 1.0})
        assert res.status_code == 200

        # Now 100% traffic goes to v2-test
        res = client.post("/infer", json={"text": "canary routing test"})
        assert res.status_code == 200
        assert res.json()["model_version"] == "v2-test"

        # Rollback
        res = client.post("/models/rollback")
        assert res.status_code == 200

        # Back to primary
        res = client.post("/infer", json={"text": "post rollback test"})
        assert res.status_code == 200
        assert res.json()["model_version"] == "cpu-embedding-v1"
