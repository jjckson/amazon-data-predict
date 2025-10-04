"""Keepa connector translating responses into canonical records."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from itertools import islice
from typing import Any, Dict, Iterable, List

from connectors.base import (
    BadRequestError,
    BaseConnector,
    ConnectorError,
    RateLimitError,
    UpstreamError,
    retry_and_rate_limit,
)


SITE_TO_DOMAIN = {
    "US": 1,
    "UK": 2,
    "DE": 3,
    "FR": 4,
    "JP": 5,
    "CA": 6,
    "IT": 8,
    "ES": 9,
    "IN": 10,
    "MX": 11,
    "AU": 12,
}
DOMAIN_TO_SITE = {v: k for k, v in SITE_TO_DOMAIN.items()}


class KeepaClient(BaseConnector):
    """Wrapper around Keepa's Product API with unified field mapping."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.keepa.com",
        settings: Dict[str, Any] | None = None,
        session: Any | None = None,
    ) -> None:
        super().__init__(service_name="keepa", base_url=base_url, settings=settings, session=session)
        self.api_key = api_key
        self.batch_size = 100

    # ------------------------------------------------------------------
    # HTTP wrappers
    # ------------------------------------------------------------------
    @retry_and_rate_limit(source="keepa")
    def _request_keepa(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(params)
        payload["key"] = self.api_key
        response = self._http_request("GET", endpoint, params=payload)
        if "error" in response:
            code = response["error"].get("code")
            if code == "RATE_LIMIT" or code == "NOT_ENOUGH_TOKENS":
                raise RateLimitError("Keepa token budget exceeded")
            if response["error"].get("status") == 400:
                raise BadRequestError(response["error"].get("message", "Bad request"))
            raise UpstreamError(response["error"].get("message", "Keepa error"))
        tokens = response.get("tokensConsumed", 0)
        if tokens:
            self._record_tokens("keepa", tokens)
        return response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_product(self, asin: str, site: str) -> Dict[str, Any]:
        domain = self._resolve_domain(site)
        response = self._request_keepa(
            "product",
            {
                "asin": asin,
                "domain": domain,
                "history": 1,
                "stats": 1,
                "offers": 1,
            },
        )
        products = response.get("products", [])
        if not products:
            raise UpstreamError(f"ASIN {asin} not found on {site}")
        return self._normalise_product(products[0])

    def get_products(self, asins: List[str], site: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        domain = self._resolve_domain(site)
        for chunk in self._chunk(asins, self.batch_size):
            response = self._request_keepa(
                "product",
                {
                    "asin": ",".join(chunk),
                    "domain": domain,
                    "history": 1,
                    "stats": 1,
                    "offers": 1,
                },
            )
            products = response.get("products", [])
            for product in products:
                results.append(self._normalise_product(product))
            self._handle_failures(response, chunk, site)
        return results

    def get_variations(self, asin: str, site: str) -> Dict[str, Any]:
        domain = self._resolve_domain(site)
        response = self._request_keepa(
            "product",
            {
                "asin": asin,
                "domain": domain,
                "variations": 1,
            },
        )
        return {
            "asin": asin,
            "site": site,
            "variations": response.get("variations", []),
        }

    def get_seller(self, seller_id: str, site: str) -> Dict[str, Any]:
        domain = self._resolve_domain(site)
        response = self._request_keepa(
            "seller",
            {
                "seller": seller_id,
                "domain": domain,
            },
        )
        return {
            "seller_id": seller_id,
            "site": site,
            "details": response.get("seller", {}),
        }

    def get_deals(self, site: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        domain = self._resolve_domain(site)
        payload = {"domain": domain}
        if params:
            payload.update(params)
        response = self._request_keepa("deals", payload)
        deals = response.get("deals", [])
        for deal in deals:
            deal["site"] = site
        return deals

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _normalise_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        asin = product.get("asin")
        site = DOMAIN_TO_SITE.get(product.get("domainId"), "US")
        dim_record = self._map_dim_asin(product, site)
        timeseries = self._map_timeseries(product, site)
        reviews = self._map_reviews(product, site)
        keywords = self._map_keywords(product, site)
        return {
            "dim_asin": dim_record,
            "timeseries": timeseries,
            "reviews": reviews,
            "keywords": keywords,
            "raw": product,
        }

    def _map_dim_asin(self, product: Dict[str, Any], site: str) -> Dict[str, Any]:
        category_path = [node.get("name") for node in product.get("categoryTree", []) if node.get("name")]
        images = product.get("imagesCSV", "")
        image_list = [img for img in images.split(",") if img]
        now = datetime.now(timezone.utc)
        return {
            "asin": product.get("asin"),
            "site": site,
            "title": product.get("title"),
            "brand": product.get("brand"),
            "category_path": category_path,
            "images": image_list,
            "first_seen": now.isoformat(),
            "last_seen": now.isoformat(),
        }

    def _map_timeseries(self, product: Dict[str, Any], site: str) -> List[Dict[str, Any]]:
        history = product.get("history", {}) or {}
        sales_rank_series = self._extract_sales_rank(history)
        price_series = self._select_price_series(history)
        rating_series = self._extract_series(history.get("rating"), scale=100)
        review_series = self._extract_series(history.get("reviewCount"))
        stock_series = self._extract_series(history.get("stock"))
        buybox_series = self._extract_string_series(history.get("buyBoxSeller"))

        combined: dict[datetime.date, Dict[str, Any]] = defaultdict(dict)
        for dt_key, value in sales_rank_series.items():
            combined[dt_key]["bsr"] = value
        for dt_key, value in price_series.items():
            combined[dt_key]["price"] = value
        for dt_key, value in rating_series.items():
            combined[dt_key]["rating"] = value
        for dt_key, value in review_series.items():
            combined[dt_key]["reviews_count"] = value
        for dt_key, value in stock_series.items():
            combined[dt_key]["stock_est"] = value
        for dt_key, value in buybox_series.items():
            combined[dt_key]["buybox_seller"] = value

        records: List[Dict[str, Any]] = []
        for dt_key, values in sorted(combined.items()):
            record = {
                "asin": product.get("asin"),
                "site": site,
                "dt": dt_key.isoformat(),
                "price": values.get("price"),
                "bsr": values.get("bsr"),
                "rating": values.get("rating"),
                "reviews_count": values.get("reviews_count"),
                "stock_est": values.get("stock_est"),
                "buybox_seller": values.get("buybox_seller"),
                "_source": "keepa",
                "_ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            records.append(record)
        return records

    def _map_reviews(self, product: Dict[str, Any], site: str) -> List[Dict[str, Any]]:
        reviews = product.get("reviews", []) or []
        normalised = []
        for review in reviews:
            normalised.append(
                {
                    "asin": product.get("asin"),
                    "site": site,
                    "review_id": review.get("reviewId") or review.get("id"),
                    "dt": self._to_iso_date(review.get("timestamp")),
                    "rating": review.get("rating"),
                    "title": review.get("title"),
                    "text": review.get("text"),
                    "verified": review.get("verified"),
                    "_source": "keepa",
                }
            )
        return normalised

    def _map_keywords(self, product: Dict[str, Any], site: str) -> List[Dict[str, Any]]:
        keywords = product.get("keywords", []) or []
        normalised = []
        for kw in keywords:
            normalised.append(
                {
                    "asin": product.get("asin"),
                    "site": site,
                    "keyword": kw.get("keyword") or kw.get("term"),
                    "est_search_volume": kw.get("searchVolume"),
                    "cpc": kw.get("cpc"),
                    "difficulty": kw.get("difficulty"),
                    "_source": "keepa",
                }
            )
        return normalised

    # ------------------------------------------------------------------
    # Series extraction utilities
    # ------------------------------------------------------------------
    def _extract_sales_rank(self, history: Dict[str, Any]) -> Dict[datetime.date, int | None]:
        ranks = history.get("salesRanks") or history.get("salesRank")
        if isinstance(ranks, dict):
            # take the first category available
            for series in ranks.values():
                return self._extract_series(series)
        return {}

    def _select_price_series(self, history: Dict[str, Any]) -> Dict[datetime.date, float | None]:
        for key in ("buyBoxPrice", "newPrice", "new", "buyBoxSalePrice"):
            if key in history:
                return self._extract_series(history[key], scale=100)
        return {}

    def _extract_series(self, series: Any, *, scale: int | None = None) -> Dict[datetime.date, Any]:
        points = {}
        for timestamp, value in self._iter_points(series):
            if value is None:
                continue
            if scale:
                value = value / scale
            dt_key = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
            points[dt_key] = value
        return points

    def _extract_string_series(self, series: Any) -> Dict[datetime.date, str | None]:
        points = {}
        for timestamp, value in self._iter_points(series):
            if value is None:
                continue
            dt_key = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
            points[dt_key] = value
        return points

    def _iter_points(self, series: Any) -> Iterable[tuple[int, Any]]:
        if not series:
            return ()
        if isinstance(series, list):
            if series and isinstance(series[0], dict):
                for item in series:
                    timestamp = self._resolve_timestamp(item)
                    if timestamp is not None:
                        yield timestamp, item.get("value")
                return
            if all(isinstance(value, (int, float)) for value in series):
                yield from self._decode_compressed(series)
                return
        return ()

    def _decode_compressed(self, series: List[int | float]) -> Iterable[tuple[int, Any]]:
        iterator = iter(series)
        absolute_minute = None
        for minute_token, value in zip(iterator, iterator):
            if absolute_minute is None:
                absolute_minute = int(minute_token)
            else:
                absolute_minute += int(minute_token)
            timestamp = absolute_minute * 60
            yield timestamp, value

    def _resolve_timestamp(self, item: Dict[str, Any]) -> int | None:
        if "timestamp" in item:
            return int(item["timestamp"])
        if "time" in item:
            return int(item["time"])
        if "dt" in item:
            return int(item["dt"])
        return None

    def _to_iso_date(self, timestamp: Any) -> str | None:
        if timestamp is None:
            return None
        ts = int(timestamp)
        return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()

    def _resolve_domain(self, site: str) -> int:
        try:
            return SITE_TO_DOMAIN[site.upper()]
        except KeyError as exc:  # pragma: no cover - guard rail
            raise BadRequestError(f"Unsupported site {site}") from exc

    def _handle_failures(self, response: Dict[str, Any], batch: List[str], site: str) -> None:
        failed_asins = set(response.get("asinNotFound", []))
        error_codes = response.get("asinError", {})
        for asin in batch:
            if asin in failed_asins or asin in error_codes:
                self.dead_letter.append({"asin": asin, "site": site, "reason": "keepa_fetch_failed"})

    def _chunk(self, iterable: Iterable[str], size: int) -> Iterable[List[str]]:
        iterator = iter(iterable)
        while True:
            chunk = list(islice(iterator, size))
            if not chunk:
                break
            yield chunk

    # ------------------------------------------------------------------
    # Healthcheck
    # ------------------------------------------------------------------
    def healthcheck(self) -> bool:
        try:
            payload = self._request_keepa("token", {"domain": 1})
            return "tokensLeft" in payload
        except ConnectorError:
            return False
