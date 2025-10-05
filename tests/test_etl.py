from __future__ import annotations

from datetime import datetime

import pytest

pd = pytest.importorskip("pandas")

from pipelines.etl_standardize import run
from training.build_labels import LabelWindows, Thresholds, build_labels


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
                "sales": 0,
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
                "sales": 6,
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
    standardized["bsr_rank"] = standardized["bsr"]
    facts_df = standardized[["asin", "site"]].drop_duplicates().assign(category_id="generic")
    result = build_labels(
        standardized,
        facts_df=facts_df,
        windows=LabelWindows(observation_days=1, forecast_days=1),
        thresholds=Thresholds(r=0.1, p=0.05, delta=10),
    )

    samples = result.samples.sort_values("t_ref").reset_index(drop=True)
    assert len(samples) == 1
    first_row = samples.iloc[0]
    assert pd.isna(first_row["past_bsr_rank"])
    assert first_row["future_bsr_rank"] == pytest.approx(100.0)
    assert pd.isna(first_row["bsr_rank_improvement"])
    assert first_row["y_bin"] == 0
