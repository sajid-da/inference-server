"""Multi-run HTTP load test script comparing single-queue vs dynamic batching."""

import asyncio
import json
import statistics
import time
from typing import Any

import httpx

from inference_server.api import app


async def run_load_sample(
    client: httpx.AsyncClient,
    num_requests: int = 500,
    concurrency: int = 50,
) -> dict[str, Any]:
    """Execute concurrent POST /infer HTTP requests and measure latencies."""
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    success_count = 0
    error_count = 0

    async def single_request(text: str) -> None:
        nonlocal success_count, error_count
        async with semaphore:
            start = time.perf_counter()
            try:
                payload = {"text": text, "timeout_seconds": 10.0}
                resp = await client.post("/infer", json=payload)
                elapsed = time.perf_counter() - start
                if resp.status_code == 200:
                    success_count += 1
                    latencies.append(elapsed * 1000.0)
                else:
                    error_count += 1
            except Exception:
                error_count += 1

    texts = [f"benchmark request input string {i}" for i in range(num_requests)]
    test_start = time.perf_counter()
    tasks = [asyncio.create_task(single_request(t)) for t in texts]
    await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - test_start

    sorted_latencies = sorted(latencies)
    p50 = statistics.median(sorted_latencies) if sorted_latencies else 0.0
    p95_idx = int(len(sorted_latencies) * 0.95)
    p99_idx = int(len(sorted_latencies) * 0.99)
    p95 = sorted_latencies[p95_idx] if sorted_latencies else 0.0
    p99 = sorted_latencies[p99_idx] if sorted_latencies else 0.0
    tps = success_count / total_elapsed if total_elapsed > 0 else 0.0

    return {
        "num_requests": num_requests,
        "concurrency": concurrency,
        "success_count": success_count,
        "error_count": error_count,
        "total_elapsed_seconds": total_elapsed,
        "throughput_req_per_sec": tps,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }


async def run_multi_sample_benchmark(
    client: httpx.AsyncClient,
    use_batching: bool,
    runs: int = 5,
    num_requests: int = 500,
    concurrency: int = 50,
) -> dict[str, Any]:
    """Execute multiple benchmark runs to obtain stable statistical distributions."""
    import inference_server.api as api_module

    api_module.USE_BATCHING = use_batching
    target_queue = api_module.batching_queue if use_batching else api_module.queue
    target_queue.start()

    # Warmup
    await run_load_sample(client, num_requests=50, concurrency=10)

    samples: list[dict[str, Any]] = []
    for _ in range(runs):
        sample = await run_load_sample(
            client, num_requests=num_requests, concurrency=concurrency
        )
        samples.append(sample)

    await target_queue.close()

    tps_list = [s["throughput_req_per_sec"] for s in samples]
    p50_list = [s["p50_ms"] for s in samples]
    p95_list = [s["p95_ms"] for s in samples]
    p99_list = [s["p99_ms"] for s in samples]

    return {
        "strategy": (
            "Dynamic Batching (I3)" if use_batching else "Single Queue Baseline (I2)"
        ),
        "runs": runs,
        "requests_per_run": num_requests,
        "concurrency": concurrency,
        "throughput_req_sec": {
            "mean": statistics.mean(tps_list),
            "median": statistics.median(tps_list),
            "min": min(tps_list),
            "max": max(tps_list),
            "stdev": statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0,
        },
        "latency_ms": {
            "p50_median": statistics.median(p50_list),
            "p95_median": statistics.median(p95_list),
            "p99_median": statistics.median(p99_list),
        },
    }


async def main() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Benchmark 50 concurrency
        batching_50 = await run_multi_sample_benchmark(
            client, use_batching=True, runs=5, num_requests=500, concurrency=50
        )
        baseline_50 = await run_multi_sample_benchmark(
            client, use_batching=False, runs=5, num_requests=500, concurrency=50
        )

    report = {
        "benchmark": "Multi-Run HTTP Load Test Comparison (5 Runs per Strategy)",
        "concurrency_50_workers": {
            "dynamic_batching": batching_50,
            "single_queue_baseline": baseline_50,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
