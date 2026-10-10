# Decisions
- FastAPI/Uvicorn for I0; specified stack; rejected a custom HTTP server.
- No model dependency in I0; a health endpoint is the milestone; rejected premature model integration.
- A deterministic local CPU model is the I1 baseline; free and repeatable; rejected network model calls in performance tests.
