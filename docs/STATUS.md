# Status
I0 is implemented: FastAPI health endpoint, test, lint/type-check settings, and CI.
Run `python -m pip install -e '.[dev]'` then `make test`.

I1 is implemented: a deterministic CPU embedding model is served synchronously
as the one-request-at-a-time baseline. The measured result is in METRICS.md.

I2 is implemented: `POST /infer` validates input, queues bounded CPU work, and
returns HTTP 429 when capacity is exhausted. A client timeout returns HTTP 503
while admitted CPU work retains capacity until the worker completes it.

I6/I8 preparation is available as verified standalone code: a thread-safe
versioned model registry and a local candidate-versus-reference evaluation
interface. Neither is yet integrated into the serving path.

Exact next step: I3, add configurable dynamic batching with maximum batch size
and maximum wait time, then test ordering and no lost/duplicated responses.
