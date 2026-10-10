# Inference Server

A CPU-first serving layer for batched ML inference, built incrementally.

```sh
python -m pip install -e '.[dev]'
make test
make run
```

## Limitations

I0 includes only a health endpoint. No model execution or public deployment yet.
