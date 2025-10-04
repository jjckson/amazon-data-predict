"""Helium10 keyword metrics adapter."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from connectors.base import BaseConnector, ConnectorError, retry_and_rate_limit


class Helium10Client(BaseConnector):
    """Adapter for Helium10 keyword metrics endpoints."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.helium10.com",
        settings: Dict[str, Any] | None = None,
        session: Any | None = None,
    ) -> None:
        super().__init__(service_name="helium10", base_url=base_url, settings=settings, session=session)
        self.api_key = api_key

    @retry_and_rate_limit(source="helium10")
    def _request_h10(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"X-API-KEY": self.api_key}
        return self._http_request("GET", endpoint, params=params, headers=headers)

    def get_keyword_metrics(self, keywords: Iterable[str], marketplace: str) -> List[Dict[str, Any]]:
        response = self._request_h10(
            "/keywords",
            {"keywords": ",".join(keywords), "marketplace": marketplace},
        )
        results: List[Dict[str, Any]] = []
        for row in response.get("data", []):
            results.append(
                {
                    "keyword": row.get("keyword"),
                    "site": marketplace,
                    "est_search_volume": row.get("search_volume"),
                    "difficulty": row.get("difficulty"),
                    "cpc": row.get("cpc"),
                    "_source": "helium10",
                }
            )
        return results

    def healthcheck(self) -> bool:
        try:
            self._request_h10("/status", {})
            return True
        except ConnectorError:
            return False
