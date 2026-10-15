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
	$(MAKE) env

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

.PHONY: download-cicids schemas schemas-check eval-e1 producer

download-cicids:
	$(PY) scripts/download_cicids2017_improved.py

schemas:
	$(PY) scripts/generate_avro_schemas.py

schemas-check:
	$(PY) scripts/generate_avro_schemas.py --check

eval-e1:
	$(PY) scripts/eval_e1.py
	$(PY) scripts/render_results.py

producer:
	$(PY) -m driftguard.producer.replay

.PHONY: up down logs bootstrap bootstrap-synthetic fit-gx-bounds test-integration replay

up: env
	./scripts/check_stack_free.sh compose
	@test -f data/raw/cicids2017_improved/monday.csv || $(MAKE) download-cicids
	@test -f configs/data/cicids2017_gx_bounds.json || $(MAKE) fit-gx-bounds
	docker compose up -d --build --wait

down:
	docker compose --profile replay --profile tools down

logs:
	docker compose logs -f --tail=200

bootstrap:
	docker compose run --rm bootstrap python scripts/bootstrap.py --dataset $(or $(DATASET),cicids2017)

bootstrap-synthetic:
	docker compose run --rm bootstrap python scripts/bootstrap.py --source synthetic

fit-gx-bounds:
	$(PY) scripts/fit_gx_bounds.py --dataset $(or $(DATASET),cicids2017)

test-integration:
	$(UV) run pytest -m integration tests/integration

replay:
	docker compose run --rm producer python -m driftguard.producer.replay $(ARGS)

.PHONY: benchmark-drift

benchmark-drift:
	$(PY) scripts/benchmark_drift.py
	$(PY) scripts/render_results.py

.PHONY: eval-e5 rollback test-e2e

eval-e5:
	$(PY) scripts/eval_e5.py
	$(PY) scripts/render_results.py

rollback:
	@test -n "$(DATASET)" || (echo "usage: make rollback DATASET=cicids2017" && exit 1)
	$(PY) scripts/rollback.py --dataset $(DATASET)

test-e2e:
	$(UV) run pytest -m e2e tests/e2e

.PHONY: loadtest benchmark-onnx dashboards dashboards-check benchmark eval

loadtest:
	$(PY) scripts/loadtest.py
	$(PY) scripts/render_results.py

benchmark-onnx:
	$(PY) scripts/benchmark_onnx.py
	$(PY) scripts/render_results.py

dashboards:
	$(PY) scripts/build_dashboards.py

dashboards-check:
	$(PY) scripts/build_dashboards.py --check

benchmark: benchmark-drift benchmark-onnx

eval: eval-e1 eval-e5

.PHONY: download-kyoto

download-kyoto:
	$(PY) scripts/download_kyoto_anoshift.py

.PHONY: eval-e4

eval-e4:
	./scripts/check_stack_free.sh compose
	$(PY) scripts/eval_e4.py
	$(PY) scripts/render_results.py

.PHONY: k8s-up k8s-down loadtest-k8s k8s-validate chart-files

HELM ?= $(if $(wildcard .cache/bin/helm),.cache/bin/helm,helm)
KUBECONFORM ?= $(if $(wildcard .cache/bin/kubeconform),.cache/bin/kubeconform,kubeconform)

k8s-up:
	./scripts/k8s_up.sh

k8s-down:
	./scripts/k8s_down.sh

loadtest-k8s:
	$(PY) scripts/loadtest_k8s.py
	$(PY) scripts/render_results.py

chart-files:
	$(PY) scripts/sync_chart_files.py

k8s-validate:
	$(PY) scripts/sync_chart_files.py --check
	$(HELM) lint deploy/k8s/charts/driftguard --set secrets.adminToken=x --set secrets.ipHashSalt=y
	$(HELM) template driftguard deploy/k8s/charts/driftguard -f deploy/k8s/values-kind.yaml \
		--set secrets.adminToken=x --set secrets.ipHashSalt=y > .tmp/rendered.yaml
	$(KUBECONFORM) -strict -summary -kubernetes-version 1.31.0 \
		-schema-location default \
		-schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json' \
		.tmp/rendered.yaml deploy/k8s/strimzi/*.yaml deploy/k8s/keda/*.yaml

.PHONY: demo

demo: bootstrap
	$(PY) scripts/demo.py $(ARGS)

.PHONY: env

env:
	$(PY) scripts/init_env.py
