from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from pipelines.score_baseline import score
from utils.config import load_settings


SETTINGS = load_settings()


def test_score_generates_reason_and_rank():
    data = pd.DataFrame(
        {
            "asin": ["A", "B"],
            "site": ["US", "US"],
            "dt": ["2024-01-01", "2024-01-01"],
            "category": ["Home", "Home"],
            "bsr_trend_30": [0.1, 0.2],
            "est_sales_30": [100, 120],
            "review_vel_14": [5, 10],
            "price_vol_30": [0.05, 0.07],
            "listing_quality": [0.8, 0.9],
        }
    )

    result = score(data, SETTINGS, rank_min=20)
    assert set(result.columns) == {"asin", "site", "dt", "category", "explosive_score", "reason", "rank_in_cat"}
    assert result["rank_in_cat"].min() == 1
    assert result["reason"].iloc[0].startswith("{")
