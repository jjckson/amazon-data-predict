"""Amazon Selling Partner API adapter exposing unified payloads."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from connectors.base import (
    BadRequestError,
    BaseConnector,
    ConnectorError,
    RateLimitError,
    UpstreamError,
    retry_and_rate_limit,
)


class SPAPIClient(BaseConnector):
    """SP-API client supporting orders, inventory, reports, and ads metrics."""

    def __init__(
        self,
        refresh_token: str,
        *,
        base_url: str = "https://sellingpartnerapi.amazon.com",
        settings: Dict[str, Any] | None = None,
        session: Any | None = None,
    ) -> None:
        super().__init__(service_name="spapi", base_url=base_url, settings=settings, session=session)
        self.refresh_token = refresh_token

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    @retry_and_rate_limit(source="spapi")
    def _request_spapi(self, method: str, endpoint: str, params: Dict[str, Any] | None = None, json_body: Dict[str, Any] | None = None) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.refresh_token}"}
        return self._http_request(method, endpoint, params=params, json_payload=json_body, headers=headers)

    # ------------------------------------------------------------------
    # Orders & inventory
    # ------------------------------------------------------------------
    def get_orders(self, start: str, end: str, marketplace_ids: Iterable[str]) -> List[Dict[str, Any]]:
        payload = self._request_spapi(
            "GET",
            "orders/v0/orders",
            params={
                "CreatedAfter": start,
                "CreatedBefore": end,
                "MarketplaceIds": ",".join(marketplace_ids),
            },
        )
        results: List[Dict[str, Any]] = []
        for order in payload.get("Orders", []):
            site = order.get("MarketplaceId")
            for item in order.get("OrderItems", []):
                results.append(
                    {
                        "asin": item.get("ASIN"),
                        "site": site,
                        "order_id": order.get("AmazonOrderId"),
                        "ordered_units": item.get("QuantityOrdered"),
                        "shipped_units": item.get("QuantityShipped"),
                        "currency": order.get("OrderTotal", {}).get("CurrencyCode"),
                        "gross_sales": order.get("OrderTotal", {}).get("Amount"),
                        "purchase_date": order.get("PurchaseDate"),
                    }
                )
        return results

    def get_inventory(self, marketplace_id: str) -> List[Dict[str, Any]]:
        payload = self._request_spapi(
            "GET",
            "fba/inventory/v1/summaries",
            params={"MarketplaceId": marketplace_id},
        )
        results: List[Dict[str, Any]] = []
        for record in payload.get("inventorySummaries", []):
            results.append(
                {
                    "asin": record.get("asin"),
                    "site": marketplace_id,
                    "fulfillable_quantity": record.get("inventoryDetails", {}).get("fulfillableQuantity"),
                    "inbound_working": record.get("inventoryDetails", {}).get("inboundWorkingQuantity"),
                    "last_updated": record.get("lastUpdatedTime"),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def create_report(self, report_type: str, params: Dict[str, Any]) -> str:
        payload = self._request_spapi(
            "POST",
            "reports/2021-06-30/reports",
            json_body={"reportType": report_type, "reportOptions": params},
        )
        report_id = payload.get("reportId")
        if not report_id:
            raise UpstreamError("Failed to create report")
        return report_id

    def get_report_result(self, report_id: str) -> bytes:
        payload = self._request_spapi(
            "GET",
            f"reports/2021-06-30/reports/{report_id}",
        )
        document_id = payload.get("reportDocumentId")
        if not document_id:
            raise UpstreamError("Report not ready")
        document = self._request_spapi(
            "GET",
            f"reports/2021-06-30/documents/{document_id}",
        )
        data = document.get("data")
        if isinstance(data, str):
            return data.encode("utf-8")
        if isinstance(data, bytes):
            return data
        raise UpstreamError("Invalid report document payload")

    # ------------------------------------------------------------------
    # Advertising metrics
    # ------------------------------------------------------------------
    def get_ads_metrics(self, profile_id: str, start: str, end: str, level: str) -> List[Dict[str, Any]]:
        payload = self._request_spapi(
            "GET",
            f"/ads/2023-01-01/profiles/{profile_id}/metrics",
            params={"startDate": start, "endDate": end, "granularity": level},
        )
        metrics: List[Dict[str, Any]] = []
        for row in payload.get("metrics", []):
            metrics.append(
                {
                    "profile_id": profile_id,
                    "site": row.get("marketplaceId"),
                    "date": row.get("date"),
                    "impressions": row.get("impressions"),
                    "clicks": row.get("clicks"),
                    "spend": row.get("spend"),
                    "sales": row.get("sales"),
                    "acos": row.get("acos"),
                    "cvr": row.get("cvr"),
                }
            )
        return metrics

    # ------------------------------------------------------------------
    # Healthcheck
    # ------------------------------------------------------------------
    def healthcheck(self) -> bool:
        try:
            payload = self._request_spapi("GET", "tokens/2021-03-01/status")
            return payload.get("status") == "OK"
        except (ConnectorError, BadRequestError, RateLimitError, UpstreamError):
            return False
