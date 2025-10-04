"""Connector for the unified ASIN data API."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from connectors.base import BaseConnector


class UnifiedAPIClient(BaseConnector):
    """Client to interact with the unified product intelligence API."""

    def __init__(self, base_url: str, token: str, **kwargs: Any) -> None:
        super().__init__(service_name="unified_api", base_url=base_url, **kwargs)
        self.token = token

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def healthcheck(self) -> bool:  # noqa: D401
        try:
            self._request("GET", "health", headers=self._headers())
            return True
        except Exception:  # noqa: BLE001
            return False

    def get_product_core(self, asin: str, site: str) -> Dict[str, Any]:
        return self._request(
            "GET",
            f"products/{site}/{asin}",
            headers=self._headers(),
        )

    def get_product_timeseries(
        self,
        asin: str,
        site: str,
        start: str,
        end: str,
    ) -> List[Dict[str, Any]]:
        params = {"start": start, "end": end}
        return self._request(
            "GET",
            f"products/{site}/{asin}/timeseries",
            params=params,
            headers=self._headers(),
        )

    def get_reviews_batch(self, asin: str, site: str, since: str) -> List[Dict[str, Any]]:
        params = {"since": since}
        return self._request(
            "GET",
            f"products/{site}/{asin}/reviews",
            params=params,
            headers=self._headers(),
        )

    def get_keywords(self, asin: str, site: str) -> Iterable[Dict[str, Any]]:
        return self._request(
            "GET",
            f"products/{site}/{asin}/keywords",
            headers=self._headers(),
        )
