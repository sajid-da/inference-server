"""Repeatable in-process baseline measurement for the deterministic CPU model."""

import json
import statistics
import time

from inference_server.model import CpuEmbeddingModel
from inference_server.service import NaiveInferenceService

REQUESTS_PER_SAMPLE = 10_000
SAMPLES = 7
WARMUP_REQUESTS = 1_000


def run_sample(service: NaiveInferenceService) -> float:
    """Return elapsed seconds for one fixed number of in-process requests."""
    start = time.perf_counter()
    for _ in range(REQUESTS_PER_SAMPLE):
        service.infer("benchmark request")
    return time.perf_counter() - start


def main() -> None:
    """Warm up, measure repeated samples, and print machine-readable results."""
    service = NaiveInferenceService(CpuEmbeddingModel())
    for _ in range(WARMUP_REQUESTS):
        service.infer("warmup")
    seconds = [run_sample(service) for _ in range(SAMPLES)]
    throughput = [REQUESTS_PER_SAMPLE / sample for sample in seconds]
    print(
        json.dumps(
            {
                "measurement": "in-process synchronous model calls; excludes HTTP",
                "requests_per_sample": REQUESTS_PER_SAMPLE,
                "samples": SAMPLES,
                "warmup_requests": WARMUP_REQUESTS,
                "median_seconds": statistics.median(seconds),
                "median_requests_per_second": statistics.median(throughput),
                "min_requests_per_second": min(throughput),
                "max_requests_per_second": max(throughput),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
