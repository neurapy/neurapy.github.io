.DEFAULT_GOAL := help

UV ?= uv
NPM ?= npm
GH ?= gh
TAPLO ?= RUST_LOG=error $(UV) run taplo
PAGES_REPO ?= neurapy/neurapy.github.io
PAGES_REF ?= main
PAGES_WORKFLOW ?= pages.yml
VERIFY_SAMPLES ?= 5

.PHONY: help install dev build preview webdemo-test webdemo-e2e webdemo-check verify-data format lint typecheck test check deploy undeploy deploy-status deploy-watch clean

help:
	@printf '%s\n' \
		'Available targets:' \
		'  install          Install Python/frontend dependencies and git hooks' \
		'  dev              Start the Vite webdemo dev server' \
		'  build            Build webdemo/dist for static hosting' \
		'  preview          Preview the built webdemo locally' \
		'  webdemo-test     Run frontend unit tests' \
		'  webdemo-e2e      Run frontend Playwright tests with fixtures' \
		'  webdemo-check    Run all frontend verification commands' \
		'  verify-data      Verify generated webdemo/public/data assets' \
		'  format           Format Python and TOML files' \
		'  lint             Lint Python files with Ruff' \
		'  typecheck        Run Pyright and TypeScript checks' \
		'  test             Run Python and frontend unit tests' \
		'  check            Run all local verification commands' \
		'  deploy           Manually trigger the GitHub Pages workflow' \
		'  undeploy         Print the command to disable GitHub Pages' \
		'  deploy-status    Show recent GitHub Pages workflow runs' \
		'  deploy-watch     Watch the latest GitHub Pages workflow run' \
		'  clean            Remove build, test, and tool artifacts'

# Installation & Deployment
install:
	$(UV) sync --all-groups
	$(NPM) --prefix webdemo install
	$(UV) run pre-commit install

dev:
	$(NPM) --prefix webdemo run dev

build:
	$(NPM) --prefix webdemo run build

preview:
	$(NPM) --prefix webdemo run preview

verify-data:
	$(UV) run python src/verify_static_demo_data.py --samples $(VERIFY_SAMPLES)

clean:
	rm -rf .coverage .coverage.* .mypy_cache .pytest_cache .pyright .ruff_cache build coverage.xml htmlcov wheels
	rm -rf dist test-results playwright-report
	rm -rf webdemo/dist webdemo/test-results webdemo/playwright-report webdemo/public/fixtures
	find . \
		-path ./.git -prune -o \
		-path ./.venv -prune -o \
		-type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf node_modules webdemo/node_modules

# Development
format:
	$(UV) run ruff format .
	$(UV) run taplo fmt pyproject.toml

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run --all-groups pyright
	$(NPM) --prefix webdemo run typecheck

test:
	$(UV) run pytest
	$(NPM) --prefix webdemo run test
	$(NPM) --prefix webdemo run test:e2e

check:
	$(UV) run ruff format --check .
	$(UV) run taplo fmt --check pyproject.toml
	$(UV) run ruff check .
	$(UV) run --all-groups pyright
	$(UV) run pytest
	$(NPM) --prefix webdemo run check
	$(MAKE) verify-data

# Deploying to Github.io
deploy:
	$(GH) workflow run $(PAGES_WORKFLOW) --repo $(PAGES_REPO) --ref $(PAGES_REF)

undeploy:
	@printf '%s\n' \
		'This will disable GitHub Pages for $(PAGES_REPO).' \
		'Run manually:' \
		'  $(GH) api --method DELETE repos/$(PAGES_REPO)/pages'

deploy-status:
	$(GH) run list --repo $(PAGES_REPO) --workflow $(PAGES_WORKFLOW) --limit 5

deploy-watch:
	$(GH) run watch $$($(GH) run list --repo $(PAGES_REPO) --workflow $(PAGES_WORKFLOW) --limit 1 --json databaseId --jq '.[0].databaseId') --repo $(PAGES_REPO) --exit-status
