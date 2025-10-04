"""JungleScout keyword metrics adapter."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from connectors.base import BaseConnector, ConnectorError, retry_and_rate_limit


class JungleScoutClient(BaseConnector):
    """Adapter for JungleScout keyword and product intelligence endpoints."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.junglescout.com",
        settings: Dict[str, Any] | None = None,
        session: Any | None = None,
    ) -> None:
        super().__init__(service_name="junglescout", base_url=base_url, settings=settings, session=session)
        self.api_key = api_key

    @retry_and_rate_limit(source="junglescout")
    def _request_js(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"X-API-KEY": self.api_key}
        return self._http_request("GET", endpoint, params=params, headers=headers)

    def get_keyword_metrics(self, keywords: Iterable[str], marketplace: str) -> List[Dict[str, Any]]:
        response = self._request_js(
            "/keywords",
            {"keywords": ",".join(keywords), "marketplace": marketplace},
        )
        metrics: List[Dict[str, Any]] = []
        for row in response.get("data", []):
            metrics.append(
                {
                    "keyword": row.get("keyword"),
                    "site": marketplace,
                    "est_search_volume": row.get("search_volume"),
                    "difficulty": row.get("difficulty"),
                    "cpc": row.get("cpc"),
                    "_source": "junglescout",
                }
            )
        return metrics

    def healthcheck(self) -> bool:
        try:
            self._request_js("/status", {})
            return True
        except ConnectorError:
            return False
