"""Connector base classes and retry/limit orchestration."""
from __future__ import annotations

import abc
import importlib.util
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional

if importlib.util.find_spec("requests") is None:  # pragma: no cover - fallback for test envs
    class _FallbackSession:
        def request(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401 - simple fallback
            raise RuntimeError("The 'requests' package is required for HTTP operations")

    class requests:  # type: ignore[override]
        Session = _FallbackSession
else:  # pragma: no cover - runtime path when requests is installed
    import requests

from utils.backoff import RetryPolicy, sleep_with_backoff
from utils.config import load_settings
from utils.logging import get_logger
from utils.rate_limiter import RateLimitTimeout, get_rate_limiter


class ConnectorError(Exception):
    """Generic upstream communication error."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RateLimitError(ConnectorError):
    """Raised when the upstream or local limiter refuses the request."""


class UpstreamError(ConnectorError):
    """Raised for retryable upstream failures (HTTP 5xx)."""


class BadRequestError(ConnectorError):
    """Raised for non-retryable client issues (HTTP 4xx)."""


@dataclass
class MetricSnapshot:
    requests: int = 0
    success: int = 0
    failure: int = 0
    latency_ms: float = 0.0
    tokens_used: int = 0


def retry_and_rate_limit(source: str, qpm_key: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator applying token bucket acquisition and exponential backoff."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self: BaseConnector, *args: Any, **kwargs: Any) -> Any:
            limiter_key = qpm_key or source
            tenant = kwargs.get("tenant", "default")
            attempt = 0
            while True:
                limiter = self._get_limiter(limiter_key)
                timeout = self._get_timeout(limiter_key)
                start = time.perf_counter()
                try:
                    limiter.acquire(tenant=tenant, timeout=timeout)
                except RateLimitTimeout as exc:  # local limiter timeout
                    self._record_failure(source, 0.0)
                    raise RateLimitError(str(exc)) from exc

                try:
                    result = func(self, *args, **kwargs)
                except RateLimitError as exc:
                    self._record_failure(source, (time.perf_counter() - start) * 1000)
                    if attempt >= self.retry_policy.max_attempts - 1:
                        raise
                    sleep_with_backoff(attempt, self.retry_policy)
                    attempt += 1
                    continue
                except UpstreamError as exc:
                    self._record_failure(source, (time.perf_counter() - start) * 1000)
                    if attempt >= self.retry_policy.max_attempts - 1:
                        raise
                    sleep_with_backoff(attempt, self.retry_policy)
                    attempt += 1
                    continue
                except BadRequestError:
                    self._record_failure(source, (time.perf_counter() - start) * 1000)
                    raise
                except ConnectorError:
                    self._record_failure(source, (time.perf_counter() - start) * 1000)
                    raise
                else:
                    latency_ms = (time.perf_counter() - start) * 1000
                    self._record_success(source, latency_ms)
                    return result

        return wrapper

    return decorator


class BaseConnector(abc.ABC):
    """Base connector encapsulating shared behaviour across adapters."""

    def __init__(
        self,
        *,
        service_name: str,
        base_url: str,
        settings: Optional[Dict[str, Any]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.settings = settings or load_settings()
        self.logger = get_logger(service_name)
        retry_cfg = self.settings.get("retry_policy", {})
        self.retry_policy = RetryPolicy(
            max_attempts=retry_cfg.get("max_attempts", 5),
            base_delay_ms=retry_cfg.get("base_delay_ms", 500),
            jitter_ms=retry_cfg.get("jitter_ms", 300),
            max_delay_ms=retry_cfg.get("max_delay_ms", 60000),
        )
        self.session = session or requests.Session()
        self._metrics: dict[str, MetricSnapshot] = defaultdict(MetricSnapshot)
        self._rate_limiters: dict[str, Any] = {}
        self._timeout_cache: dict[str, float] = {}
        self.dead_letter: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Rate limiting helpers
    # ------------------------------------------------------------------
    def _get_limiter(self, key: str):
        if key not in self._rate_limiters:
            cfg = self.settings.get("rate_limits", {}).get(key, {})
            qpm = cfg.get("qpm", 200)
            burst = cfg.get("burst", qpm)
            self._rate_limiters[key] = get_rate_limiter(key, capacity=burst, qpm=qpm)
        return self._rate_limiters[key]

    def _get_timeout(self, key: str) -> float:
        if key not in self._timeout_cache:
            cfg = self.settings.get("rate_limits", {}).get(key, {})
            self._timeout_cache[key] = float(cfg.get("timeout_sec", 10))
        return self._timeout_cache[key]

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def _record_success(self, source: str, latency_ms: float, tokens: int = 0) -> None:
        metrics = self._metrics[source]
        metrics.requests += 1
        metrics.success += 1
        metrics.latency_ms += latency_ms
        metrics.tokens_used += tokens
        self.logger.debug(
            "connector=%s source=%s latency_ms=%.2f tokens=%s",  # pragma: no cover - debug
            self.service_name,
            source,
            latency_ms,
            tokens,
        )

    def _record_failure(self, source: str, latency_ms: float) -> None:
        metrics = self._metrics[source]
        metrics.requests += 1
        metrics.failure += 1
        metrics.latency_ms += latency_ms
        self.logger.debug(  # pragma: no cover - debug
            "connector=%s source=%s failure latency_ms=%.2f",
            self.service_name,
            source,
            latency_ms,
        )

    def _record_tokens(self, source: str, tokens: int) -> None:
        metrics = self._metrics[source]
        metrics.tokens_used += tokens

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------
    def _http_request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.request(
            method,
            url,
            params=params,
            json=json_payload,
            headers=headers,
            timeout=self._get_timeout(self.service_name),
        )
        status = getattr(response, "status_code", 500)
        if status == 429:
            raise RateLimitError(f"{self.service_name} rate limited", status=status)
        if 500 <= status:
            raise UpstreamError(f"{self.service_name} upstream failure", status=status)
        if 400 <= status:
            raise BadRequestError(f"{self.service_name} bad request", status=status)
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - upstream contract
            raise UpstreamError("Invalid JSON response", status=status) from exc

    @abc.abstractmethod
    def healthcheck(self) -> bool:
        """Validate connectivity to the upstream service."""
