# Metrics
I1 self-made in-process benchmark (Python 3.12, local CPU; excludes HTTP): 1,000
warmup calls, then 7 samples of 10,000 synchronous calls. Median 0.0332204 s
and 301019.86 calls/s; range 262407.96-305679.53 calls/s. Reproduce:
`PYTHONPATH=src .venv/Scripts/python scripts/benchmark_naive.py`.

Latest I2 verification reran the same benchmark locally: median 0.0325845 s and
306894.38 calls/s; range 302759.65-311258.86 calls/s. This remains an
in-process measurement, not HTTP/server throughput.
