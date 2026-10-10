"""Tests for Prometheus metrics collection and rendering."""

from fastapi.testclient import TestClient

from inference_server.api import app


def test_metrics_endpoint_returns_prometheus_format() -> None:
    with TestClient(app) as client:
        # Perform inference
        res = client.post("/infer", json={"text": "metrics test"})
        assert res.status_code == 200

        # Fetch metrics
        metrics_res = client.get("/metrics")
        assert metrics_res.status_code == 200
        assert "text/plain" in metrics_res.headers["content-type"]

        body = metrics_res.text
        assert "http_requests_total" in body
        assert 'status="200"' in body
        assert "queue_overflows_total" in body
        assert "queue_timeouts_total" in body
