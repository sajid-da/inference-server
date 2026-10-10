# Learn
## I0
- The API module creates the web application.
- `/health` confirms the process responds.
- The test calls the route handler without a network port.
- CI repeats lint, type checking, and tests on Python 3.12.

## I1
- The model converts text into a fixed-size normalized feature vector.
- This intentionally synchronous path is the performance baseline for batching.
- The benchmark uses no network or external model, so its result is reproducible.

## I6 preparation
- A model handle combines the model object with the version returned to callers.
- An atomic replacement prevents readers from seeing a half-updated handle.

## I8 preparation
- Variant evaluation compares a candidate embedding with a reference embedding.
- Cosine similarity is a simple agreement measure for equal-sized vectors.

## I2
- A bounded queue limits both waiting work and CPU work that is already running.
- A caller can stop waiting, but the worker keeps the admitted slot until CPU work finishes.
- HTTP 429 tells a caller to retry later because the server is full.
- HTTP 503 signals that an accepted request timed out waiting for its result.
