"""Structured JSON logging for HTTP requests and agent traces.

Kept deliberately small: stdlib `logging` with a JSON formatter and a
request-id, rather than pulling in an APM SDK. Every log line is one JSON
object per line (easy to pipe into `jq`, Loki, CloudWatch, etc.).
"""
from __future__ import annotations
import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id_ctx.get(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _configure_root() -> None:
    root = logging.getLogger("policybot")
    if root.handlers:
        return  # already configured (avoid dupes on reload)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False


_configure_root()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"policybot.{name}")


def log_event(logger: logging.Logger, message: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, message, extra={"fields": fields})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs one structured line per HTTP request with status + latency."""

    async def dispatch(self, request: Request, call_next):
        req_id = str(uuid.uuid4())[:8]
        token = _request_id_ctx.set(req_id)
        log = get_logger("http")
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log_event(log, "request_failed", level=logging.ERROR,
                       method=request.method, path=request.url.path,
                       latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            raise
        finally:
            _request_id_ctx.reset(token)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        response.headers["X-Request-ID"] = req_id
        log_event(log, "request", method=request.method, path=request.url.path,
                   status_code=response.status_code, latency_ms=latency_ms)
        return response


def current_request_id() -> str:
    return _request_id_ctx.get()
