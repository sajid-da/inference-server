# Decisions
- FastAPI/Uvicorn for I0; specified stack; rejected a custom HTTP server.
- No model dependency in I0; a health endpoint is the milestone; rejected premature model integration.
- A deterministic local CPU model is the I1 baseline; free and repeatable; rejected network model calls in performance tests.
- I2 retains admitted work after a client wait timeout; prevents detached CPU work from bypassing bounded capacity; rejected cancelling unknown CPU work mid-execution.
- Model handles are immutable and swapped under an `RLock`; allows version reads to observe either old or new complete state; rejected mutating a live model object.
- I8 uses a local agreement/latency evaluation interface; no supported quantized runtime is installed; rejected inventing quantization measurements.
