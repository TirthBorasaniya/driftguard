# DriftGuard Makefile. All caches stay inside the repository (.cache, .tmp, .kube).

SHELL := /bin/bash
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.cache/uv-python
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
export PYTHONPATH := $(CURDIR)/src
UV := uv
PY := $(UV) run python

.PHONY: setup lint format test results results-check import-v1-results clean

setup:
	$(UV) python install 3.11
	$(UV) sync --frozen --python 3.11

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:
	$(UV) run pytest

results:
	$(PY) scripts/render_results.py

results-check:
	$(PY) scripts/render_results.py --check

import-v1-results:
	$(PY) scripts/import_v1_results.py
	$(PY) scripts/render_results.py

clean:
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache .ruff_cache .tmp/pytest
