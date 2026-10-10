"""Thread-safe Prometheus metrics exporter for operational counters and latencies."""

from threading import Lock


class Metrics:
    """Collect process-level counters and render Prometheus text format metrics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests_total: dict[tuple[str, str], int] = {}
        self._queue_overflows = 0
        self._queue_timeouts = 0
        self._total_duration_seconds = 0.0

    def record_request(
        self, status_code: int, model_version: str, duration_seconds: float = 0.0
    ) -> None:
        """Record one completed HTTP request with status code and model version."""
        key = (str(status_code), model_version)
        with self._lock:
            self._requests_total[key] = self._requests_total.get(key, 0) + 1
            self._total_duration_seconds += duration_seconds

    def record_overflow(self) -> None:
        """Record a queue capacity overflow (HTTP 429)."""
        with self._lock:
            self._queue_overflows += 1

    def record_timeout(self) -> None:
        """Record a request wait timeout (HTTP 503)."""
        with self._lock:
            self._queue_timeouts += 1

    def render(self) -> str:
        """Render metrics in Prometheus text format 0.0.4."""
        lines = [
            "# HELP http_requests_total Total number of HTTP inference requests.",
            "# TYPE http_requests_total counter",
        ]
        with self._lock:
            for (code, version), count in sorted(self._requests_total.items()):
                metric_line = (
                    f'http_requests_total{{status="{code}",'
                    f'model_version="{version}"}} {count}'
                )
                lines.append(metric_line)
            duration_val = f"{self._total_duration_seconds:.6f}"
            lines.extend(
                [
                    "# HELP queue_overflows_total Queue capacity overflows.",
                    "# TYPE queue_overflows_total counter",
                    f"queue_overflows_total {self._queue_overflows}",
                    "# HELP queue_timeouts_total Request wait timeouts.",
                    "# TYPE queue_timeouts_total counter",
                    f"queue_timeouts_total {self._queue_timeouts}",
                    "# HELP http_request_duration_seconds_sum Duration sum.",
                    "# TYPE http_request_duration_seconds_sum counter",
                    f"http_request_duration_seconds_sum {duration_val}",
                ]
            )
        return "\n".join(lines) + "\n"
