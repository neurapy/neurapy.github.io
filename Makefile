.DEFAULT_GOAL := help

UV ?= uv
NODE ?= node
TAPLO ?= RUST_LOG=error $(UV) run taplo

.PHONY: help install webdemo format lint typecheck test check clean

help:
	@printf '%s\n' \
		'Available targets:' \
		'  install    Install dependencies and git hooks' \
		'  webdemo    Start the Vite webdemo dev server' \
		'  format     Format Python and TOML files' \
		'  lint       Lint Python files with Ruff' \
		'  typecheck  Run Pyright' \
		'  test       Run pytest' \
		'  check      Run all verification commands' \
		'  clean      Remove local build and tool artifacts'

install: 
	$(UV) sync --all-groups
	npm --prefix webdemo install
	$(UV) run pre-commit install

webdemo:
	npm --prefix webdemo run dev

format:
	$(UV) run ruff format .
	$(UV) run taplo fmt pyproject.toml

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run --all-groups pyright
	npm --prefix webdemo run typecheck

test:
	$(UV) run pytest
	npm --prefix webdemo run test

check:
	$(UV) run ruff format --check .
	$(UV) run taplo fmt --check pyproject.toml
	$(UV) run ruff check .
	$(UV) run --all-groups pyright
	$(UV) run pytest
	npm --prefix webdemo run typecheck
	npm --prefix webdemo run test
	npm --prefix webdemo run test:e2e
	npm --prefix webdemo run build
	$(UV) run python src/verify_static_demo_data.py --samples 5

clean:
	rm -rf .coverage .coverage.* .mypy_cache .pytest_cache .pyright .ruff_cache build coverage.xml htmlcov wheels
	rm -rf node_modules test-results playwright-report dist
	rm -rf webdemo/node_modules webdemo/test-results webdemo/playwright-report webdemo/dist
	find . \
		-path ./.git -prune -o \
		-path ./.venv -prune -o \
		-type d -name __pycache__ -prune -exec rm -rf {} +
