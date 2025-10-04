from __future__ import annotations

from datetime import datetime

import pytest

pd = pytest.importorskip("pandas")

from pipelines.etl_standardize import run
from training.build_labels import build_labels


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
    assert pd.api.types.is_float_dtype(result["price"])
    assert pd.api.types.is_float_dtype(result["bsr"])
    assert pd.api.types.is_float_dtype(result["rating"])
    assert pd.api.types.is_bool_dtype(result["price_valid"])
    assert pd.api.types.is_bool_dtype(result["bsr_valid"])
    row = result.iloc[0]
    assert bool(row["price_valid"])
    assert not row["bsr_valid"]
    assert pd.isna(row["bsr"])


def test_standardize_and_build_labels_handle_invalid_bsr():
    raw = pd.DataFrame(
        [
            {
                "asin": "A",
                "site": "US",
                "dt": "2024-01-01",
                "price": 10,
                "bsr": "not-a-number",
                "rating": 4.5,
                "reviews_count": 5,
                "stock_est": 1,
                "buybox_seller": "Seller1",
                "_ingested_at": datetime(2024, 1, 1, 1),
            },
            {
                "asin": "A",
                "site": "US",
                "dt": "2024-01-02",
                "price": 12,
                "bsr": 100,
                "rating": 4.6,
                "reviews_count": 6,
                "stock_est": 2,
                "buybox_seller": "Seller1",
                "_ingested_at": datetime(2024, 1, 2, 1),
            },
        ]
    )

    standardized = run([raw])
    assert list(standardized["bsr_valid"]) == [False, True]
    labels = build_labels(standardized)

    first_row = labels.iloc[0]
    assert pd.isna(first_row["bsr"])
    assert not first_row["bsr_valid"]
    assert first_row["future_bsr"] == pytest.approx(100.0)
    assert not first_row["bsr_improved"]
