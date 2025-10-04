"""Token bucket rate limiter supporting multiple tenants."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

from utils.logging import get_logger

logger = get_logger(__name__)


class RateLimitTimeout(Exception):
    """Raised when acquiring tokens exceeds timeout."""


@dataclass
class TokenBucket:
    capacity: int
    refill_rate_per_sec: float
    tokens: float = field(init=False)
    last_refill: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)

    def acquire(self, tokens: int = 1, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        with self.lock:
            while True:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                if time.monotonic() >= deadline:
                    raise RateLimitTimeout(
                        f"Timed out acquiring {tokens} tokens"
                    )
                wait_time = min(
                    (tokens - self.tokens) / self.refill_rate_per_sec,
                    deadline - time.monotonic(),
                )
                wait_time = max(wait_time, 0.01)
                logger.debug(
                    "Waiting %.2fs for rate limiter tokens", wait_time
                )
                time.sleep(wait_time)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed <= 0:
            return
        refill_tokens = elapsed * self.refill_rate_per_sec
        self.tokens = min(self.capacity, self.tokens + refill_tokens)
        self.last_refill = now


class RateLimiter:
    """Manage token buckets per tenant key."""

    def __init__(self, capacity: int, qpm: int) -> None:
        self.capacity = capacity
        self.refill_rate = qpm / 60.0
        self.buckets: Dict[str, TokenBucket] = defaultdict(self._create_bucket)

    def _create_bucket(self) -> TokenBucket:
        return TokenBucket(self.capacity, self.refill_rate)

    def acquire(self, tenant: str = "default", tokens: int = 1, timeout: float = 10.0) -> None:
        bucket = self.buckets[tenant]
        bucket.acquire(tokens=tokens, timeout=timeout)


_default_limiters: Dict[str, RateLimiter] = {}

def get_rate_limiter(name: str, capacity: int, qpm: int) -> RateLimiter:
    if name not in _default_limiters:
        _default_limiters[name] = RateLimiter(capacity=capacity, qpm=qpm)
    return _default_limiters[name]
