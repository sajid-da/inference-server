# Metrics

## I1 Naive In-Process Benchmark
Self-made in-process benchmark (Python 3.14, local CPU; excludes HTTP): 1,000 warmup calls, then 7 samples of 10,000 synchronous calls.
Median 0.0347975 s and 287,376.97 calls/s; range 283,148.15 - 294,354.86 calls/s.
Reproduce: `.tmp-python/Scripts/python scripts/benchmark_naive.py`.

## I4 HTTP Load Test (Multi-Sample Benchmark: 5 Runs per Strategy)
Workload: 500 HTTP `POST /infer` requests per run, 50 concurrent workers, 5 repeated runs per serving strategy.
Reproduce: `.tmp-python/Scripts/python scripts/benchmark_http.py`.

| Serving Strategy | Mean Throughput (req/s) | Throughput StdDev | Median p50 Latency | Median p95 Latency | Median p99 Latency | Error Rate |
|---|---|---|---|---|---|---|
| **Dynamic Batching (I3)** | **1,292.05 req/s** | 13.02 req/s | **34.98 ms** | **38.84 ms** | **39.95 ms** | 0.0% (0/2500) |
| Single Queue Baseline (I2) | 1,112.90 req/s | 14.03 req/s | 43.86 ms | 46.17 ms | 46.58 ms | 0.0% (0/2500) |

### Performance Insights & Statistical Analysis
- **Throughput Advantage**: Under 50-worker concurrent load, Dynamic Batching delivers a **+16.1% increase in throughput** (1,292 req/s vs 1,112 req/s).
- **Tail Latency Reduction**: Median p50 latency is reduced by **20.2%** (34.98 ms vs 43.86 ms) and median p99 latency is reduced by **14.2%** (39.95 ms vs 46.58 ms).
- Benchmark executed using ASGI Transport (`httpx.ASGITransport`) on Python 3.14 / FastAPI 0.143 to measure serving layer efficiency without network socket noise.
