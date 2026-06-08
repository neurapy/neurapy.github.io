.DEFAULT_GOAL := help

UV ?= uv
TAPLO ?= RUST_LOG=error $(UV) run taplo

.PHONY: help install webdemo format lint typecheck test check clean

help:
	@printf '%s\n' \
		'Available targets:' \
		'  install    Install dependencies and git hooks' \
		'  webdemo    Serve the static webdemo on http://127.0.0.1:8080/' \
		'  format     Format Python and TOML files' \
		'  lint       Lint Python files with Ruff' \
		'  typecheck  Run Pyright' \
		'  test       Run pytest' \
		'  check      Run all verification commands' \
		'  clean      Remove local build and tool artifacts'

install: 
	$(UV) sync --all-groups
	$(UV) run pre-commit install

webdemo:
	$(UV) run python src/start_webdemo_server.py

format:
	$(UV) run ruff format .
	$(UV) run taplo fmt pyproject.toml

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run --all-groups pyright

test:
	$(UV) run pytest

check:
	$(UV) run ruff format --check .
	$(UV) run taplo fmt --check pyproject.toml
	$(UV) run ruff check .
	$(UV) run --all-groups pyright
	$(UV) run pytest

clean:
	rm -rf .coverage .coverage.* .mypy_cache .pytest_cache .pyright .ruff_cache build coverage.xml dist htmlcov wheels
	find . \
		-path ./.git -prune -o \
		-path ./.venv -prune -o \
		-type d -name __pycache__ -prune -exec rm -rf {} +
