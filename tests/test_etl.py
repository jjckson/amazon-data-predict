from __future__ import annotations

from datetime import datetime

import pytest

pd = pytest.importorskip("pandas")

from pipelines.etl_standardize import run


def test_standardize_handles_duplicates_and_flags():
    raw = pd.DataFrame(
        [
            {
                "asin": "A",
                "site": "US",
                "dt": "2024-01-01",
                "price": -1,
                "bsr": 10,
                "rating": 4.5,
                "reviews_count": 10,
                "stock_est": 5,
                "buybox_seller": "Seller1",
                "_ingested_at": datetime(2024, 1, 1, 1),
            },
            {
                "asin": "A",
                "site": "US",
                "dt": "2024-01-01",
                "price": 20,
                "bsr": -5,
                "rating": 4.6,
                "reviews_count": 11,
                "stock_est": 6,
                "buybox_seller": "Seller1",
                "_ingested_at": datetime(2024, 1, 1, 2),
            },
        ]
    )

    result = run([raw])
    assert len(result) == 1
    row = result.iloc[0]
    assert row["price_valid"] is True
    assert row["bsr_valid"] is False
    assert row["bsr"] is None
