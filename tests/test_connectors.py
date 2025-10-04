from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pytest

from connectors.base import (
    BadRequestError,
    BaseConnector,
    UpstreamError,
    retry_and_rate_limit,
)
from connectors.keepa_client import KeepaClient
from tests.mock_keepa_server import sample_product_payload

TEST_SETTINGS = {
    "retry_policy": {
        "max_attempts": 3,
        "base_delay_ms": 1,
        "jitter_ms": 0,
        "max_delay_ms": 10,
    },
    "rate_limits": {
        "keepa": {"qpm": 1000, "burst": 1000, "timeout_sec": 1},
        "spapi": {"qpm": 1000, "burst": 1000, "timeout_sec": 1},
        "paapi": {"qpm": 1000, "burst": 1000, "timeout_sec": 1},
        "helium10": {"qpm": 1000, "burst": 1000, "timeout_sec": 1},
        "junglescout": {"qpm": 1000, "burst": 1000, "timeout_sec": 1},
    },
}


class DummyConnector(BaseConnector):
    def __init__(self) -> None:
        super().__init__(service_name="keepa", base_url="https://example.com", settings=TEST_SETTINGS)
        self.attempts = 0

    @retry_and_rate_limit(source="keepa")
    def flaky_call(self) -> Dict[str, Any]:
        self.attempts += 1
        if self.attempts < 2:
            raise UpstreamError("temporary failure")
        return {"ok": True}

    @retry_and_rate_limit(source="keepa")
    def bad_call(self) -> Dict[str, Any]:
        raise BadRequestError("bad input", status=400)

    def healthcheck(self) -> bool:  # pragma: no cover - not used in tests
        return True


def test_retry_and_rate_limit_retries(monkeypatch):
    monkeypatch.setattr("connectors.base.sleep_with_backoff", lambda attempt, policy: None)
    connector = DummyConnector()
    result = connector.flaky_call()
    assert result == {"ok": True}
    metrics = connector._metrics["keepa"]
    assert metrics.requests == 2
    assert metrics.success == 1
    assert metrics.failure == 1


def test_bad_request_does_not_retry(monkeypatch):
    monkeypatch.setattr("connectors.base.sleep_with_backoff", lambda attempt, policy: None)
    connector = DummyConnector()
    with pytest.raises(BadRequestError):
        connector.bad_call()
    metrics = connector._metrics["keepa"]
    assert metrics.requests == 1
    assert metrics.failure == 1
    assert metrics.success == 0


class FakeKeepaClient(KeepaClient):
    def __init__(self) -> None:
        super().__init__("token", settings=TEST_SETTINGS)
        self._responses: List[Dict[str, Any]] = []

    def _request_keepa(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        payload = self._responses.pop(0)
        return payload


def test_keepa_normalisation(monkeypatch):
    client = KeepaClient("token", settings=TEST_SETTINGS)
    product = sample_product_payload()
    normalised = client._normalise_product(product)  # type: ignore[attr-defined]

    dim = normalised["dim_asin"]
    assert dim["asin"] == product["asin"]
    assert dim["site"] == "US"
    assert dim["title"] == product["title"]
    assert dim["brand"] == product["brand"]
    assert dim["category_path"] == ["Root", "Sub"]

    timeseries = normalised["timeseries"]
    assert len(timeseries) == 2
    first_entry = timeseries[0]
    assert first_entry["price"] == pytest.approx(19.99, rel=1e-3)
    assert first_entry["reviews_count"] == 200
    assert first_entry["bsr"] == 1500
    assert first_entry["_source"] == "keepa"

    reviews = normalised["reviews"]
    assert reviews[0]["review_id"] == "R1"
    assert reviews[0]["_source"] == "keepa"

    keywords = normalised["keywords"]
    assert keywords[0]["keyword"] == "widget"
    assert keywords[0]["_source"] == "keepa"


def test_get_products_handles_dead_letter(monkeypatch):
    fake = FakeKeepaClient()
    fake._responses.append(  # first batch
        {
            "products": [sample_product_payload()],
            "asinNotFound": ["B0002"],
        }
    )
    products = fake.get_products(["B0001", "B0002"], "US")
    assert len(products) == 1
    assert fake.dead_letter == [{"asin": "B0002", "site": "US", "reason": "keepa_fetch_failed"}]
