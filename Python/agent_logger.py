"""
Structured logging helper for agents and LLM-style workflows.

Features
--------
- JSON logging with timestamped records and contextual fields (trace_id, agent).
- Async and sync decorators for wrapping agent functions.
- Cost estimation from common LLM token usage objects.
- Safe output preview to avoid huge log lines.

This module adapts the MultiAgent logger pattern and keeps defaults minimal so it
can drop into small utilities or larger services.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast

F = TypeVar("F", bound=Callable[..., Any])

# Example token costs (USD per token). Adjust to match the models you use.
TOKEN_COST_INPUT = 0.50 / 1_000_000  # $0.50 per million input tokens
TOKEN_COST_OUTPUT = 1.50 / 1_000_000  # $1.50 per million output tokens

LOG_LEVEL = logging.INFO

# --- Logger setup with JSON formatter fallback ---
try:
    from pythonjsonlogger import jsonlogger
except ImportError:

    class _FallbackJsonFormatter(logging.Formatter):
        """Minimal JSON formatter used when python-json-logger is absent."""

        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "message": record.getMessage(),
                "name": record.name,
                "lineno": record.lineno,
                "funcName": record.funcName,
                "module": record.module,
                **getattr(record, "extra", {}),
            }
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    jsonlogger = type("DummyModule", (object,), {"JsonFormatter": _FallbackJsonFormatter})()


def setup_agent_logger(name: str) -> logging.Logger:
    """
    Configure a structured logger for a given agent name.

    Handlers are added only once, so repeated calls are cheap.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s",
            timestamp=True,
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
    return logger


class TraceLoggerAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that injects trace_id and agent_name into each log record.
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def _safe_serialize(obj: Any, max_len: int = 500) -> str:
    """
    Convert an object to a string for logging, truncating if necessary.
    """

    try:
        if hasattr(obj, "text") and isinstance(getattr(obj, "text"), str):
            text = getattr(obj, "text")
        elif isinstance(obj, (dict, list)):
            text = json.dumps(obj)
        else:
            text = str(obj)
        return text if len(text) <= max_len else f"{text[:max_len]}... (truncated, len={len(text)})"
    except Exception:
        return f"<non-serializable {type(obj).__name__}>"


def calculate_cost(usage: Any) -> float:
    """
    Estimate USD cost from a token usage object (commonly from LLM SDKs).
    """

    if not usage:
        return 0.0
    input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "completion_tokens", 0)
    cost = (input_tokens * TOKEN_COST_INPUT) + (output_tokens * TOKEN_COST_OUTPUT)
    return round(cost, 6)


def get_error_tags(exc: Exception) -> Tuple[str, Optional[int]]:
    """
    Derive error metadata for logging dashboards.
    """

    status_code: Optional[int] = None
    error_type = type(exc).__name__

    if error_type == "RateLimitError":
        status_code = 429
    elif error_type.endswith("APIError"):
        status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)

    return error_type, status_code


@asynccontextmanager
async def agent_log_context(
    agent_name: str,
    func_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    trace_id: Optional[str] = None,
    logger_name_prefix: str = "agent.",
):
    """
    Async context manager that logs start, errors, and duration for an agent.
    """

    trace = trace_id or kwargs.get("trace_id") or str(uuid.uuid4())[:8]
    base_logger = setup_agent_logger(f"{logger_name_prefix}{agent_name}")
    log = TraceLoggerAdapter(base_logger, {"trace_id": trace, "agent_name": agent_name})

    start = time.time()
    log.info("▶ START | func=%s | trace_id=%s", func_name, trace, extra={"func_name": func_name})
    exception: Optional[Exception] = None

    try:
        yield log
    except Exception as exc:  # noqa: PERF203 (explicit logging path)
        exception = exc
        raise
    finally:
        duration_ms = int((time.time() - start) * 1000)
        if exception:
            error_type, status_code = get_error_tags(exception)
            log.exception(
                "❌ ERROR | duration=%dms | type=%s | status=%s",
                duration_ms,
                error_type,
                status_code,
                extra={
                    "duration_ms": duration_ms,
                    "error_type": error_type,
                    "http_status": status_code,
                    "func_name": func_name,
                },
            )


@contextmanager
def agent_log_context_sync(
    agent_name: str,
    func_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    trace_id: Optional[str] = None,
    logger_name_prefix: str = "agent.",
):
    """
    Sync counterpart to `agent_log_context` for non-async call sites.
    """

    trace = trace_id or kwargs.get("trace_id") or str(uuid.uuid4())[:8]
    base_logger = setup_agent_logger(f"{logger_name_prefix}{agent_name}")
    log = TraceLoggerAdapter(base_logger, {"trace_id": trace, "agent_name": agent_name})

    start = time.time()
    log.info("▶ START | func=%s | trace_id=%s", func_name, trace, extra={"func_name": func_name})
    exception: Optional[Exception] = None

    try:
        yield log
    except Exception as exc:  # noqa: PERF203
        exception = exc
        raise
    finally:
        duration_ms = int((time.time() - start) * 1000)
        if exception:
            error_type, status_code = get_error_tags(exception)
            log.exception(
                "❌ ERROR | duration=%dms | type=%s | status=%s",
                duration_ms,
                error_type,
                status_code,
                extra={
                    "duration_ms": duration_ms,
                    "error_type": error_type,
                    "http_status": status_code,
                    "func_name": func_name,
                },
            )


def log_agent(
    agent_name: str,
    *,
    logger_name_prefix: str = "agent.",
    enable_sync: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to wrap agent functions with structured logging and metrics.
    """

    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            trace = kwargs.get("trace_id") or str(uuid.uuid4())[:8]
            kwargs.setdefault("trace_id", trace)

            async with agent_log_context(
                agent_name, func.__name__, args, kwargs, trace_id=trace, logger_name_prefix=logger_name_prefix
            ) as log:
                result = await func(*args, **kwargs)

            usage = getattr(result, "usage", None)
            cost = calculate_cost(usage)

            final_log = TraceLoggerAdapter(
                setup_agent_logger(f"{logger_name_prefix}{agent_name}"),
                {"trace_id": trace, "agent_name": agent_name},
            )
            final_log.info(
                "✅ SUCCESS | cost=$%.6f | output_preview=%s",
                cost,
                _safe_serialize(result, max_len=200),
                extra={
                    "token_usage": getattr(usage, "total_tokens", 0) if usage else 0,
                    "estimated_cost": cost,
                    "func_name": func.__name__,
                },
            )
            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            if not enable_sync:
                raise RuntimeError("Sync usage is disabled for this logger.")

            trace = kwargs.get("trace_id") or str(uuid.uuid4())[:8]
            kwargs.setdefault("trace_id", trace)

            with agent_log_context_sync(
                agent_name, func.__name__, args, kwargs, trace_id=trace, logger_name_prefix=logger_name_prefix
            ) as log:
                result = func(*args, **kwargs)

            usage = getattr(result, "usage", None)
            cost = calculate_cost(usage)

            final_log = TraceLoggerAdapter(
                setup_agent_logger(f"{logger_name_prefix}{agent_name}"),
                {"trace_id": trace, "agent_name": agent_name},
            )
            final_log.info(
                "✅ SUCCESS | cost=$%.6f | output_preview=%s",
                cost,
                _safe_serialize(result, max_len=200),
                extra={
                    "token_usage": getattr(usage, "total_tokens", 0) if usage else 0,
                    "estimated_cost": cost,
                    "func_name": func.__name__,
                },
            )
            return result

        return cast(F, async_wrapper if is_async else sync_wrapper)

    return decorator
