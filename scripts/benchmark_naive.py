"""Measure the deterministic I1 baseline without external services."""

import json
import time

from inference_server.model import CpuEmbeddingModel
from inference_server.service import NaiveInferenceService

service = NaiveInferenceService(CpuEmbeddingModel())
requests = 10_000
start = time.perf_counter()
for _ in range(requests):
    service.infer("benchmark request")
seconds = time.perf_counter() - start
print(json.dumps({"requests": requests, "seconds": seconds, "requests_per_second": requests / seconds}))
