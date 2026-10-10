# Status
I0 is implemented: FastAPI health endpoint, test, lint/type-check settings, and CI.
Run `python -m pip install -e '.[dev]'` then `make test`.

I1 is implemented: a deterministic CPU embedding model is served synchronously
as the one-request-at-a-time baseline. The measured result is in METRICS.md.

I2 is in progress: a bounded worker queue and validated `POST /infer` endpoint
have been implemented, but async tests cannot complete because independent
`asyncio.run()` commands hang in the current execution sandbox. I2 is not done.

Exact next step: run the queue/API tests in a working asyncio runtime, repair
any runtime defect, then complete the milestone with overload, timeout, and
shutdown evidence.
