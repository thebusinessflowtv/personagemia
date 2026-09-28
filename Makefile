.PHONY: install test lint api

install:
	python -m pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check .

api:
	uvicorn app.api:app --reload
