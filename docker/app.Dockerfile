# One image for every DriftGuard Python service; the command selects the service.
FROM python:3.11.9-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl procps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
COPY flows ./flows
COPY scripts ./scripts
COPY prefect.yaml ./
RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 driftguard \
    && mkdir -p /app/data /app/results \
    && chown -R driftguard:driftguard /app
USER driftguard

CMD ["python", "-m", "driftguard.consumer.main"]
