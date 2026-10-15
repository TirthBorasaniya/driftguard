"""FastAPI serving router: `/score`, `/score/shadow`, health, metrics, and admin API (SPEC 4.2)."""

import hmac
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, field_validator

from driftguard.common.config import load_config
from driftguard.common.logging import configure_logging, get_logger
from driftguard.common.secrets import secret_problem
from driftguard.common.settings import Settings, get_settings
from driftguard.serving import admin
from driftguard.serving.model_store import AliasWatcher, ModelStore
from driftguard.serving.router import MODES, NoChampionError, Router

# ============= Constants =============

logger = get_logger("serving")
MAX_EVENTS = 256
MAX_FEATURES = 512


# ============= Schemas =============


class EventIn(BaseModel):
    """One event: its id and the feature vector in request order. No IPs or payloads."""

    event_id: str = Field(min_length=1, max_length=128)
    features: list[float]


class ScoreRequest(BaseModel):
    """A scoring batch of at most 256 events."""

    dataset: str
    feature_names: list[str] = Field(min_length=1, max_length=MAX_FEATURES)
    events: list[EventIn] = Field(min_length=1, max_length=MAX_EVENTS)

    @field_validator("events")
    @classmethod
    def _widths_match(cls, events: list[EventIn], info) -> list[EventIn]:
        width = len(info.data.get("feature_names", []))
        if any(len(event.features) != width for event in events):
            raise ValueError("every feature vector must match feature_names")
        return events


class RolloutRequest(BaseModel):
    """Admin request to change the rollout mode."""

    dataset: str
    mode: str
    canary_fraction: float | None = None


class RollbackRequest(BaseModel):
    """Admin request to restore the previous champion."""

    dataset: str
    reason: str = "operator rollback"


# ============= Application =============


def create_app(
    settings: Settings | None = None, registry=None, redis_client=None, o_start_watcher: bool = True
) -> FastAPI:
    """
    Build the serving application.

    Parameters
    ----------
    settings : Settings or None
        Runtime settings.
    registry : Registry or None
        Model registry; defaults to MLflow.
    redis_client : redis.Redis or None
        Rollout state store.
    o_start_watcher : bool
        Start the alias watcher thread (tests refresh manually).
    """
    settings = settings or get_settings()
    rollout_cfg = load_config(settings.config_path("rollout.yaml")).data

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging("serving", settings.log_level)
        nonlocal registry, redis_client
        if registry is None:
            from driftguard.common.mlflow_client import MlflowRegistry

            registry = MlflowRegistry(settings.mlflow_tracking_uri)
        if redis_client is None:
            from redis import Redis

            redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        store = ModelStore(registry, settings.dataset, int(rollout_cfg["warmup_rows"]))
        app.state.store = store
        app.state.registry = registry
        app.state.redis = redis_client
        app.state.router = Router(
            store,
            redis_client,
            settings.dataset,
            float(rollout_cfg["canary_fraction"]),
            int(rollout_cfg["recent_buffer"]),
        )
        app.state.watcher = AliasWatcher(store, float(rollout_cfg["watcher_poll_s"]))
        if o_start_watcher:
            app.state.watcher.start()
        else:
            store.refresh()
        yield
        app.state.watcher.stop()

    app = FastAPI(title="DriftGuard serving", version="2.0.0", lifespan=lifespan)

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = settings.admin_token.get_secret_value()
        problem = secret_problem(expected)
        if problem:
            raise HTTPException(503, f"admin API disabled: ADMIN_TOKEN is {problem}")
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(401, "invalid admin token")

    def check_dataset(dataset: str) -> None:
        if dataset != settings.dataset:
            raise HTTPException(400, f"this router serves {settings.dataset}")

    def matrix_of(request: ScoreRequest) -> np.ndarray:
        matrix = np.asarray([event.features for event in request.events], dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise HTTPException(422, "features must be finite")
        return matrix

    @app.post("/score")
    def score(body: ScoreRequest, request: Request) -> dict:
        check_dataset(body.dataset)
        try:
            return request.app.state.router.score(
                [e.event_id for e in body.events], body.feature_names, matrix_of(body)
            )
        except NoChampionError as exc:
            raise HTTPException(503, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/score/shadow")
    def score_shadow(body: ScoreRequest, request: Request) -> dict:
        check_dataset(body.dataset)
        try:
            result = request.app.state.router.score_shadow(
                [e.event_id for e in body.events], body.feature_names, matrix_of(body)
            )
        except KeyError as exc:
            raise HTTPException(422, str(exc)) from exc
        if result is None:
            raise HTTPException(409, "shadow mode is not active")
        return result

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request) -> Response:
        ready = request.app.state.store.ready()
        return Response(
            '{"ready": %s}' % ("true" if ready else "false"),
            status_code=200 if ready else 503,
            media_type="application/json",
        )

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/predictions/recent")
    def recent(request: Request, limit: int = Query(50, ge=1, le=1000)) -> dict:
        items = list(request.app.state.router.recent)[-limit:]
        return {"dataset": settings.dataset, "predictions": items}

    @app.post("/admin/rollout", dependencies=[Depends(require_token)])
    def rollout(body: RolloutRequest, request: Request) -> dict:
        check_dataset(body.dataset)
        if body.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {MODES}")
        try:
            admin.set_mode(request.app.state.redis, body.dataset, body.mode, body.canary_fraction)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        request.app.state.router.invalidate()
        request.app.state.watcher.poke()
        logger.info("rollout_mode_set", mode=body.mode, canary_fraction=body.canary_fraction)
        return {"dataset": body.dataset, "mode": body.mode}

    @app.post("/admin/rollback", dependencies=[Depends(require_token)])
    def rollback(body: RollbackRequest, request: Request) -> dict:
        check_dataset(body.dataset)
        try:
            entry = admin.rollback(
                request.app.state.registry,
                request.app.state.redis,
                body.dataset,
                settings.promotions_log_path,
                body.reason,
            )
        except admin.RolloutConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        request.app.state.router.invalidate()
        request.app.state.watcher.poke()
        logger.info("rollback_done", champion=entry["champion"])
        return entry

    return app


def __getattr__(name: str):
    """Build the module-level `app` lazily so importing this module never needs settings."""
    if name == "app":
        return create_app()
    raise AttributeError(name)
