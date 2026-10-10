# Status
I0 is implemented: FastAPI health endpoint, test, lint/type-check settings, and CI.
Run `python -m pip install -e '.[dev]'` then `make test`.

I1 is implemented: a deterministic CPU embedding model is served synchronously
as the one-request-at-a-time baseline. The measured result is in METRICS.md.

Exact next step: I2, add a bounded queue, timeout behavior, and load shedding.
