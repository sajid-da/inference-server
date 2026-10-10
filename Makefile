.PHONY: test lint typecheck run
test:
	python -m pytest
lint:
	python -m ruff check src tests
typecheck:
	python -m mypy src
run:
	python -m uvicorn inference_server.api:app --reload
