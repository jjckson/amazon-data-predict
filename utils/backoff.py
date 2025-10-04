"""Exponential backoff helpers."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterable, TypeVar

from utils.logging import get_logger

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int
    base_delay_ms: int
    jitter_ms: int
    max_delay_ms: int

    def compute_delay(self, attempt: int) -> float:
        delay_ms = min(self.max_delay_ms, self.base_delay_ms * (2 ** attempt))
        jitter = random.uniform(-self.jitter_ms, self.jitter_ms)
        return max(0.0, (delay_ms + jitter) / 1000)


def retryable(
    *,
    retry_policy: RetryPolicy,
    on_status: Iterable[int] = (429, 500, 502, 503, 504),
    on_exceptions: Iterable[type[Exception]] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry decorator applying exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retry_policy.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    status = getattr(exc, "status", None)
                    should_retry = (
                        isinstance(exc, tuple(on_exceptions))
                        and (status in on_status if status is not None else True)
                        and attempt < retry_policy.max_attempts - 1
                    )
                    if not should_retry:
                        raise

                    delay = retry_policy.compute_delay(attempt)
                    logger.warning(
                        "Retrying %s after %s seconds due to %s (attempt %s)",
                        func.__name__,
                        delay,
                        exc,
                        attempt + 1,
                    )
                    time.sleep(delay)
            raise RuntimeError("Retry attempts exhausted")

        return wrapper

    return decorator


def sleep_with_backoff(attempt: int, retry_policy: RetryPolicy) -> None:
    """Utility for manual backoff."""
    delay = retry_policy.compute_delay(attempt)
    time.sleep(delay)
