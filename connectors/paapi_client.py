"""Amazon Product Advertising API (PA-API) adapter."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from connectors.base import BaseConnector, ConnectorError, retry_and_rate_limit


class PAAPIClient(BaseConnector):
    """Client for retrieving public catalogue information via PA-API."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        partner_tag: str,
        *,
        base_url: str = "https://webservices.amazon.com/paapi5",
        settings: Dict[str, Any] | None = None,
        session: Any | None = None,
    ) -> None:
        super().__init__(service_name="paapi", base_url=base_url, settings=settings, session=session)
        self.access_key = access_key
        self.secret_key = secret_key
        self.partner_tag = partner_tag

    @retry_and_rate_limit(source="paapi")
    def _request_paapi(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload.setdefault("PartnerTag", self.partner_tag)
        payload.setdefault("PartnerType", "Associates")
        payload.setdefault("AccessKey", self.access_key)
        payload.setdefault("SecretKey", self.secret_key)
        return self._http_request("POST", endpoint, json_payload=payload)

    def get_items(self, asins: Iterable[str], marketplace: str, resources: Iterable[str]) -> List[Dict[str, Any]]:
        payload = {
            "ItemIds": list(asins),
            "Marketplace": marketplace,
            "Resources": list(resources),
        }
        response = self._request_paapi("/getitems", payload)
        results: List[Dict[str, Any]] = []
        for item in response.get("ItemsResult", {}).get("Items", []):
            results.append(
                {
                    "dim_asin": {
                        "asin": item.get("ASIN"),
                        "site": marketplace,
                        "title": item.get("ItemInfo", {}).get("Title", {}).get("DisplayValue"),
                        "brand": item.get("ItemInfo", {}).get("ByLineInfo", {}).get("Brand", {}).get("DisplayValue"),
                        "category_path": [node.get("DisplayValue") for node in item.get("BrowseNodeInfo", {}).get("BrowseNodes", []) if node.get("DisplayValue")],
                        "images": [img.get("URL") for img in item.get("Images", {}).get("Primary", {}).values() if isinstance(img, dict) and img.get("URL")],
                    },
                    "raw": item,
                }
            )
        return results

    def search_items(self, keywords: str, marketplace: str, resources: Iterable[str], page: int = 1) -> Dict[str, Any]:
        payload = {
            "Keywords": keywords,
            "Marketplace": marketplace,
            "Resources": list(resources),
            "ItemPage": page,
        }
        response = self._request_paapi("/searchitems", payload)
        items = self.get_items(
            [item.get("ASIN") for item in response.get("SearchResult", {}).get("Items", [])],
            marketplace,
            resources,
        )
        return {"items": items, "raw": response}

    def healthcheck(self) -> bool:
        try:
            self._request_paapi("/ping", {"Marketplace": "www.amazon.com"})
            return True
        except ConnectorError:
            return False
