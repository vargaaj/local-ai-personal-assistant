from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from time import perf_counter
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from .config import Settings

logger = logging.getLogger(__name__)

try:
    import mlflow
except Exception:  # pragma: no cover - only exercised when dependency is absent
    mlflow = None  # type: ignore[assignment]

_trace_enabled = False


def trace(func: Callable[..., Any] | None = None, **kwargs: Any) -> Callable[..., Any]:
    """Small no-op wrapper around mlflow.trace until MLflow is configured."""

    def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
        traced_inner: Callable[..., Any] | None = None

        @wraps(inner)
        def wrapper(*args: Any, **call_kwargs: Any) -> Any:
            nonlocal traced_inner
            if not _trace_enabled or mlflow is None:
                return inner(*args, **call_kwargs)

            if traced_inner is None:
                traced_inner = (
                    mlflow.trace(**kwargs)(inner)
                    if kwargs
                    else mlflow.trace(inner)
                )
            return traced_inner(*args, **call_kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def update_current_trace(
    *,
    tags: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    user: str | None = None,
    request_preview: str | None = None,
    response_preview: str | None = None,
) -> None:
    if not _trace_enabled or mlflow is None:
        return

    try:
        mlflow.update_current_trace(
            tags=_string_values(tags),
            metadata=_string_values(metadata),
            session_id=session_id,
            user=user,
            request_preview=request_preview,
            response_preview=response_preview,
        )
    except Exception as exc:
        logger.debug("Failed to update current MLflow trace: %s", exc)


@contextmanager
def timed(operation: str, **fields: Any):
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        details = " ".join(
            f"{key}={value}" for key, value in fields.items() if value is not None
        )
        suffix = f" {details}" if details else ""
        logger.info("%s completed in %.3fs%s", operation, elapsed, suffix)


def _string_values(values: dict[str, Any] | None) -> dict[str, str] | None:
    if not values:
        return None
    return {str(key): str(value) for key, value in values.items() if value is not None}


def configure_mlflow(settings: Settings) -> dict[str, Any]:
    """Configure MLflow tracing without preventing the app from starting."""

    global _trace_enabled
    _trace_enabled = False
    tracking_uri = str(settings.mlflow_tracking_uri)

    with timed("mlflow.health_check", tracking_uri=tracking_uri):
        reachable, reachability_error = _tracking_server_reachable(tracking_uri)
    if not reachable:
        return {
            "configured": False,
            "tracking_uri": tracking_uri,
            "experiment": settings.mlflow_experiment,
            "error": reachability_error,
        }

    if mlflow is None:
        return {
            "configured": False,
            "tracking_uri": tracking_uri,
            "experiment": settings.mlflow_experiment,
            "error": "mlflow package is not available",
        }

    try:
        with timed(
            "mlflow.configure",
            tracking_uri=tracking_uri,
            experiment=settings.mlflow_experiment,
        ):
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment)
            mlflow.openai.autolog()
            _trace_enabled = True
        return {
            "configured": True,
            "tracking_uri": tracking_uri,
            "experiment": settings.mlflow_experiment,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - depends on external MLflow server
        logger.warning("MLflow tracing setup failed: %s", exc)
        return {
            "configured": False,
            "tracking_uri": tracking_uri,
            "experiment": settings.mlflow_experiment,
            "error": str(exc),
        }


def _tracking_server_reachable(
    tracking_uri: str,
    timeout_seconds: float = 3.0,
) -> tuple[bool, str | None]:
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in {"http", "https"}:
        return True, None

    health_url = tracking_uri.rstrip("/") + "/health"
    try:
        with urlopen(health_url, timeout=timeout_seconds) as response:
            if 200 <= response.status < 500:
                return True, None
            return False, f"MLflow health check returned HTTP {response.status}"
    except Exception as exc:
        return False, f"MLflow tracking server is not reachable at {health_url}: {exc}"
