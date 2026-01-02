"""
Reusable decorator utilities for logging, timing, retries, caching, rate limiting,
deprecation warnings, and cleanup hooks.

What you get
------------
- logged: log calls, arguments, results, and runtime with optional truncation.
- timed: log elapsed time for any callable with a custom label.
- retry: retry with exponential backoff, jitter, and per-attempt callback.
- rate_limited: simple token-bucket limiter to cap calls per time window.
- cached: in-memory cache with TTL and LRU-like eviction.
- deprecated: emit DeprecationWarning with optional replacement guidance.
- on_exit: always-run cleanup hook for teardown logic.

Quick start
-----------
```python
from decorators import logged, retry, cached

@logged(log_args=True, log_result=True)
def add(a, b):
    return a + b

@retry(attempts=5, base_delay_s=0.1, jitter=0.2)
def flaky_call():
    ...

@cached(ttl_s=60)
def get_config():
    ...
```
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union, cast

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger("decorators")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -------------------------------------------------------------------
# 1) logged
# -------------------------------------------------------------------
def logged(
    *,
    log_args: bool = False,
    log_result: bool = False,
    max_repr: int = 200,
    level: int = logging.INFO,
) -> Callable[[F], F]:
    def safe_repr(x: Any) -> str:
        r = repr(x)
        return r if len(r) <= max_repr else r[: max_repr - 3] + "..."

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            start = time.perf_counter()
            try:
                if log_args:
                    logger.log(level, "call | %s | args=%s kwargs=%s", func.__qualname__, safe_repr(args), safe_repr(kwargs))
                else:
                    logger.log(level, "call | %s", func.__qualname__)
                result = func(*args, **kwargs)
                if log_result:
                    logger.log(level, "return | %s | %s", func.__qualname__, safe_repr(result))
                return result
            except Exception:
                logger.exception("error | %s", func.__qualname__)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.log(level, "done | %s | %.2f ms", func.__qualname__, elapsed_ms)

        return cast(F, wrapper)

    return decorator


# -------------------------------------------------------------------
# 2) timed
# -------------------------------------------------------------------
def timed(*, label: Optional[str] = None, log_level: int = logging.INFO) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        name = label or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.log(log_level, "timed | %s | %.2f ms", name, elapsed_ms)

        return cast(F, wrapper)

    return decorator


# -------------------------------------------------------------------
# 3) retry
# -------------------------------------------------------------------
def retry(
    *,
    attempts: int = 3,
    exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = Exception,
    base_delay_s: float = 0.25,
    max_delay_s: float = 2.0,
    backoff: float = 2.0,
    jitter: float = 0.1,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Callable[[F], F]:
    excs = exceptions if isinstance(exceptions, tuple) else (exceptions,)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except excs as e:
                    if attempt == attempts:
                        raise
                    delay = min(max_delay_s, base_delay_s * (backoff ** (attempt - 1)))
                    if jitter:
                        delay += random.uniform(-delay * jitter, delay * jitter)
                    if on_retry:
                        on_retry(attempt, e, delay)
                    time.sleep(max(0, delay))

        return cast(F, wrapper)

    return decorator


# -------------------------------------------------------------------
# 4) rate_limited
# -------------------------------------------------------------------
def rate_limited(*, calls: int, per_seconds: float) -> Callable[[F], F]:
    lock = threading.RLock()
    capacity = float(calls)
    tokens = capacity
    refill_rate = capacity / per_seconds
    last = time.perf_counter()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            nonlocal tokens, last
            while True:
                with lock:
                    now = time.perf_counter()
                    tokens = min(capacity, tokens + (now - last) * refill_rate)
                    last = now
                    if tokens >= 1:
                        tokens -= 1
                        break
                time.sleep(1 / refill_rate)
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


# -------------------------------------------------------------------
# 5) cached
# -------------------------------------------------------------------
@dataclass(frozen=True)
class _CacheKey:
    args: Tuple[Any, ...]
    kwargs: Tuple[Tuple[str, Any], ...]


def cached(*, ttl_s: Optional[float] = None, maxsize: int = 256) -> Callable[[F], F]:
    store: OrderedDict[Any, Tuple[float, Any]] = OrderedDict()
    lock = threading.RLock()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            key = _CacheKey(args, tuple(sorted(kwargs.items())))
            now = time.time()

            with lock:
                if key in store:
                    ts, val = store[key]
                    if ttl_s is None or now - ts <= ttl_s:
                        store.move_to_end(key)
                        return val
                    del store[key]

            val = func(*args, **kwargs)

            with lock:
                store[key] = (now, val)
                store.move_to_end(key)
                if len(store) > maxsize:
                    store.popitem(last=False)
            return val

        return cast(F, wrapper)

    return decorator


# -------------------------------------------------------------------
# 6) deprecated
# -------------------------------------------------------------------
def deprecated(*, replacement: str = "", message: str = "") -> Callable[[F], F]:
    def decorator(func: F) -> F:
        text = f"{func.__qualname__} is deprecated."
        if replacement:
            text += f" Use {replacement} instead."
        if message:
            text += f" {message}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            warnings.warn(text, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


# -------------------------------------------------------------------
# 7) on_exit
# -------------------------------------------------------------------
def on_exit(cleanup: Callable[[], None]) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            try:
                return func(*args, **kwargs)
            finally:
                cleanup()

        return cast(F, wrapper)

    return decorator
