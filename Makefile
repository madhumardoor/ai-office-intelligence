.PHONY: install up down logs test test-unit lint fmt

install:
	uv sync

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

test-unit:
	uv run pytest tests/unit -v

test:
	uv run pytest -v

lint:
	uv run ruff check . && uv run mypy app

fmt:
	uv run ruff check --fix . && uv run ruff format .