"""Health endpoint tests."""

from inference_server.api import health


def test_health() -> None:
    assert health() == {"status": "ok"}
