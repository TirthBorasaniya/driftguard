# One image for every DriftGuard Python service; the command selects the service.
FROM python:3.11.9-slim-bookworm AS base

# the service user takes the host uid and gid (docker-compose.yml passes HOST_UID and HOST_GID), so
# a bind-mounted results/ owned by the host user is writable without widening any permission;
# 10001 is the default and the uid the Kubernetes chart runs as
ARG APP_UID=10001
ARG APP_GID=10001

# GitPython (imported by MLflow) warns at startup when no git binary exists in the image
ENV GIT_PYTHON_REFRESH=quiet \
    PYTHONDONTWRITEBYTECODE=1 \
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
    && { [ "${APP_UID}" -ne 0 ] && [ "${APP_GID}" -ne 0 ] \
         || { echo 'APP_UID and APP_GID must not be 0; do not run compose as root' >&2; exit 1; }; } \
    && { getent group "${APP_GID}" > /dev/null || groupadd --gid "${APP_GID}" driftguard; } \
    && useradd --create-home --uid "${APP_UID}" --gid "${APP_GID}" driftguard \
    && mkdir -p /app/data /app/results \
    && chown -R driftguard:driftguard /app
USER driftguard

CMD ["python", "-m", "driftguard.consumer.main"]
