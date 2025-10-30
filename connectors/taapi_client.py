"""TAAPI connector for retrieving technical analysis metrics."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from connectors.base import BaseConnector, ConnectorError, retry_and_rate_limit


class TaapiClient(BaseConnector):
    """Client wrapper for TAAPI bulk indicator endpoints."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.taapi.io",
        settings: Dict[str, Any] | None = None,
        session: Any | None = None,
    ) -> None:
        super().__init__(service_name="taapi", base_url=base_url, settings=settings, session=session)
        self.api_key = api_key

    @retry_and_rate_limit(source="taapi")
    def _request_bulk(self, construct: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "secret": self.api_key,
            "construct": list(construct),
        }
        headers = {"Content-Type": "application/json"}
        return self._http_request("POST", "/bulk", json_payload=payload, headers=headers)

    def get_market_state(
        self,
        symbol: str,
        exchange: str,
        *,
        short_interval: str = "5m",
        long_interval: str = "4h",
        short_backtrack: int = 20,
        long_backtrack: int = 20,
    ) -> Dict[str, Any]:
        """Fetch a consolidated market snapshot for ``symbol`` from TAAPI."""

        construct = [
            self._indicator("price", symbol, exchange, "1m", "current_price"),
            self._indicator(
                "ema",
                symbol,
                exchange,
                short_interval,
                "current_ema20",
                optInTimePeriod=20,
            ),
            self._indicator(
                "macd",
                symbol,
                exchange,
                short_interval,
                "current_macd",
            ),
            self._indicator(
                "rsi",
                symbol,
                exchange,
                short_interval,
                "current_rsi_7",
                optInTimePeriod=7,
            ),
            self._indicator(
                "rsi",
                symbol,
                exchange,
                short_interval,
                "current_rsi_14",
                optInTimePeriod=14,
            ),
            self._indicator(
                "midprice",
                symbol,
                exchange,
                short_interval,
                "short_mid_prices",
                backtrack=short_backtrack,
            ),
            self._indicator(
                "ema",
                symbol,
                exchange,
                short_interval,
                "short_ema20_series",
                optInTimePeriod=20,
                backtrack=short_backtrack,
            ),
            self._indicator(
                "macd",
                symbol,
                exchange,
                short_interval,
                "short_macd_series",
                backtrack=short_backtrack,
            ),
            self._indicator(
                "rsi",
                symbol,
                exchange,
                short_interval,
                "short_rsi7_series",
                optInTimePeriod=7,
                backtrack=short_backtrack,
            ),
            self._indicator(
                "rsi",
                symbol,
                exchange,
                short_interval,
                "short_rsi14_series",
                optInTimePeriod=14,
                backtrack=short_backtrack,
            ),
            self._indicator(
                "ema",
                symbol,
                exchange,
                long_interval,
                "long_ema20",
                optInTimePeriod=20,
            ),
            self._indicator(
                "ema",
                symbol,
                exchange,
                long_interval,
                "long_ema50",
                optInTimePeriod=50,
            ),
            self._indicator(
                "atr",
                symbol,
                exchange,
                long_interval,
                "long_atr3",
                optInTimePeriod=3,
            ),
            self._indicator(
                "atr",
                symbol,
                exchange,
                long_interval,
                "long_atr14",
                optInTimePeriod=14,
            ),
            self._indicator(
                "volume",
                symbol,
                exchange,
                long_interval,
                "long_volume_current",
            ),
            self._indicator(
                "sma",
                symbol,
                exchange,
                long_interval,
                "long_volume_avg",
                optInTimePeriod=long_backtrack,
            ),
            self._indicator(
                "macd",
                symbol,
                exchange,
                long_interval,
                "long_macd_series",
                backtrack=long_backtrack,
            ),
            self._indicator(
                "rsi",
                symbol,
                exchange,
                long_interval,
                "long_rsi14_series",
                optInTimePeriod=14,
                backtrack=long_backtrack,
            ),
            self._indicator(
                "oi",
                symbol,
                exchange,
                short_interval,
                "open_interest_latest",
            ),
            self._indicator(
                "oi",
                symbol,
                exchange,
                short_interval,
                "open_interest_avg",
                backtrack=short_backtrack,
            ),
            self._indicator(
                "fundingrate",
                symbol,
                exchange,
                short_interval,
                "funding_rate",
            ),
        ]

        response = self._request_bulk(construct)
        return self._normalise_market_state(
            response,
            symbol=symbol,
            short_interval=short_interval,
            long_interval=long_interval,
        )

    def _indicator(
        self,
        indicator: str,
        symbol: str,
        exchange: str,
        interval: str,
        identifier: str,
        **params: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "indicator": indicator,
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "id": identifier,
        }
        payload.update(params)
        return payload

    def _normalise_market_state(
        self,
        payload: Dict[str, Any],
        *,
        symbol: str,
        short_interval: str,
        long_interval: str,
    ) -> Dict[str, Any]:
        entries = payload.get("data", [])
        by_id: Dict[str, Dict[str, Any]] = {
            entry.get("id", ""): entry.get("result", {}) for entry in entries if entry.get("id")
        }

        def value(key: str) -> Any:
            return self._extract_single(by_id.get(key))

        def series(key: str) -> List[float]:
            return self._extract_series(by_id.get(key))

        market_state = {
            "symbol": symbol,
            "current_price": value("current_price"),
            "current_ema20": value("current_ema20"),
            "current_macd": value("current_macd"),
            "current_rsi_7": value("current_rsi_7"),
            "current_rsi_14": value("current_rsi_14"),
            "open_interest_latest": value("open_interest_latest"),
            "open_interest_avg": value("open_interest_avg"),
            "funding_rate": value("funding_rate"),
            "short_term": {
                "interval": short_interval,
                "mid_prices": series("short_mid_prices"),
                "ema_series_20": series("short_ema20_series"),
                "macd_series": series("short_macd_series"),
                "rsi_series_7": series("short_rsi7_series"),
                "rsi_series_14": series("short_rsi14_series"),
            },
            "long_term": {
                "interval": long_interval,
                "ema_20_4h": value("long_ema20"),
                "ema_50_4h": value("long_ema50"),
                "atr_3_4h": value("long_atr3"),
                "atr_14_4h": value("long_atr14"),
                "current_volume": value("long_volume_current"),
                "avg_volume": value("long_volume_avg"),
                "macd_series_4h": series("long_macd_series"),
                "rsi_series_14_4h": series("long_rsi14_series"),
            },
        }
        return market_state

    @staticmethod
    def _extract_single(result: Dict[str, Any] | None) -> Any:
        if not isinstance(result, dict):
            return None
        for key in ("value", "valueMACD", "valueMacd", "value1", "close"):
            if key in result and isinstance(result[key], (int, float)):
                return result[key]
        for val in result.values():
            if isinstance(val, (int, float)):
                return val
        return None

    @staticmethod
    def _extract_series(result: Dict[str, Any] | None) -> List[float]:
        if not isinstance(result, dict):
            return []
        values: Iterable[Any] = result.get("values", [])
        extracted: List[float] = []
        for item in values:
            numeric = TaapiClient._extract_single(item if isinstance(item, dict) else {"value": item})
            if isinstance(numeric, (int, float)):
                extracted.append(float(numeric))
        if not extracted:
            single = TaapiClient._extract_single(result)
            if isinstance(single, (int, float)):
                extracted.append(float(single))
        return extracted

    def healthcheck(self) -> bool:  # pragma: no cover - simple passthrough
        try:
            self._request_bulk([{"indicator": "price", "symbol": "BTC/USDT", "exchange": "binance", "interval": "1m", "id": "ping"}])
            return True
        except ConnectorError:
            return False
