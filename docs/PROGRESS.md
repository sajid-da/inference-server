# Progress checkpoint

## 2026-10-10 IST — Initial audit and I2 repair

**Overall status:** I0 and I1 are committed and verified. I2 has an untracked,
incomplete queue prototype; I3-I10 have not started.

**Verified implementation:** `GET /health`; deterministic synchronous 16-value
CPU embedding model; naive in-process benchmark script; Python 3.12 CI config.

**Files currently present:** committed I0/I1 source and tests, plus untracked
`src/inference_server/queue.py` and `tests/test_queue.py` from a failed I2
attempt. The prototype is preserved for repair, not treated as complete.

**Commands/results:**

- `pytest -p no:cacheprovider tests/test_health.py tests/test_model.py`: 3 passed.
- `ruff check src tests`: passed.
- `mypy src`: passed.
- `python scripts/benchmark_naive.py`: 10,000 calls in 0.0336902 s,
  296822.22 calls/s on local Python 3.12; this is in-process, not HTTP.
- `pytest -p no:cacheprovider tests/test_queue.py`: did not complete in a
  10-second bounded audit attempt; I2 is blocked on queue lifecycle repair.

**Known risks:** no HTTP inference endpoint; queue implementation is uncommitted
and not verified; benchmark documentation reports a different single-run result;
README is stale.

**Git:** branch `feat/i2-queue`; last committed milestone `cc1fe58`.

**Exact next action:** replace the hanging queue prototype with a deterministic
bounded worker queue, then add a validated `POST /infer` endpoint and tests for
success, overload, timeout/cancellation, and shutdown.

## 2026-10-10 IST — I2 implementation attempt / execution blocker

**Overall status:** I2 implementation is in progress and remains uncommitted.

**Implemented since checkpoint:** replaced the previous queue prototype with a
bounded single-worker queue; added request admission, client wait timeout,
graceful draining, and `POST /infer` with Pydantic validation. Added queue/API
tests. A timed-out caller leaves admitted work counted until completion to avoid
detached-work capacity oversubscription.

**Files changed:** `src/inference_server/queue.py`,
`src/inference_server/api.py`, `tests/test_queue.py`, `tests/test_api.py`.

**Commands/results:** `ruff check` initially reported two line-length defects;
they have been corrected. `mypy src` passed. `py_compile` passed. Every bounded
attempt to execute asyncio also hangs in this sandbox, including the independent
command `python -c "import asyncio; asyncio.run(asyncio.sleep(0))"`; this is
not isolated to repository code. Therefore async queue tests are **not run to
completion** and I2 is not complete.

**Known risks:** runtime behavior of async queue and endpoint remains unverified
in this execution environment. HTTP endpoint is not integration-tested. I1
benchmark remains an in-process microbenchmark.

**Git:** branch `feat/i2-queue`; last committed milestone `cc1fe58`.

**Exact next action:** obtain a working Python asyncio runtime (local terminal
outside this sandbox or CI), run `pytest tests/test_queue.py tests/test_api.py`,
then fix any true runtime failures before committing I2.

## 2026-10-10 IST — I1 benchmark strengthened

**Overall status:** I0/I1 remain complete; I2 remains in progress and async
runtime verification remains blocked by the sandbox.

**Implemented:** replaced the single-sample I1 benchmark with a warmup plus
seven-sample measurement that reports median and range, explicitly excluding
HTTP overhead.

**Files changed:** `scripts/benchmark_naive.py`, `docs/METRICS.md`.

**Commands/results:** benchmark ran on Python 3.12: median 0.0332204 s and
301019.86 calls/s over seven samples of 10,000 calls; range 262407.96-
305679.53 calls/s. `ruff check scripts` and `py_compile` passed.

**Known risks:** no HTTP/server benchmark; I2 async runtime still unverified.

**Git:** branch `feat/i2-queue`; no I2 commit because its tests cannot complete.

**Exact next action:** validate I2 in a functioning asyncio environment, then
finish I2 before beginning I3.

## 2026-10-10 IST — Independent milestone work plan

**Current milestone:** I2 remains blocked at runtime verification. To make
progress without misrepresenting dependency order, synchronous components of I6
and I8 will be implemented and tested separately. They will not be marked as
complete until their serving-path integration is verified after I2-I5.

**Exact next action:** implement a thread-safe local model registry with
version metadata and deterministic unit tests.

## 2026-10-10 IST — I6 registry preparation complete

**Overall status:** I0/I1 complete; I2 runtime verification blocked; I6 has a
tested standalone registry component but is not complete as a serving feature.

**Implemented:** immutable `ModelHandle` version metadata and a locked
`ModelRegistry` that atomically replaces valid active handles.

**Files changed:** `src/inference_server/registry.py`,
`tests/test_registry.py`, `docs/DECISIONS.md`, `docs/LEARN.md`.

**Commands/results:** `pytest -p no:cacheprovider tests/test_registry.py`: 2
passed. `ruff check src tests`: passed. `mypy src`: passed.

**Known risks:** registry is not yet connected to `/infer`; I2-I5 runtime
verification remains blocked by sandbox-wide asyncio failure.

**Git:** branch `feat/i2-queue`; verified registry/benchmark/documentation work
committed as `c976924` (`feat: add model registry groundwork`). Unverified I2
files remain uncommitted.

**Exact next action:** add a synchronous quantization-evaluation interface and
tests, then update the integration/release contract documentation.

## 2026-10-10 IST — I8 evaluation preparation

**Overall status:** I0/I1 complete; I2 runtime verification blocked; I6/I8 have
tested standalone preparation components only.

**Implemented:** `evaluate_variant` measures local candidate latency and cosine
agreement against a reference-model protocol. No quantized model is claimed or
measured because no suitable local runtime/model variant is available.

**Files changed:** `src/inference_server/evaluation.py`,
`tests/test_evaluation.py`, `docs/DECISIONS.md`, `docs/LEARN.md`.

**Commands/results:** targeted evaluation tests: 2 passed; initial Ruff pass
reported one line-length issue, corrected before the next full static check;
mypy passed.

**Known risks:** no model variant/quantization result exists; async I2 runtime
verification remains blocked.

**Git:** branch `feat/i2-queue`; verified evaluation preparation committed as
`83c453d` (`feat: add variant evaluation groundwork`). Only unverified I2 work
remains uncommitted.

**Exact next action:** run static checks after the formatting correction, then
commit the verified I8 preparation separately from unverified I2 code.
