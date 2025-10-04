from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from pipelines.features_build import build_features
from utils.config import load_settings


SETTINGS = load_settings()


def test_feature_builder_generates_expected_columns():
    data = pd.DataFrame(
        {
            "asin": ["A"] * 30,
            "site": ["US"] * 30,
            "dt": pd.date_range("2024-01-01", periods=30, freq="D"),
            "bsr": range(1, 31),
            "price": [20.0 + i * 0.1 for i in range(30)],
            "rating": [4.0] * 30,
            "reviews_count": range(30),
            "est_sales": [100] * 30,
            "title_length": [90] * 30,
            "bullet_count": [6] * 30,
            "image_count": [7] * 30,
            "has_material_attr": [True] * 30,
            "has_size_attr": [True] * 30,
            "has_use_case_attr": [True] * 30,
            "has_a_plus": [True] * 30,
        }
    )

    features = build_features(data, SETTINGS)
    expected_cols = {
        "bsr_trend_7",
        "bsr_trend_30",
        "price_vol_30",
        "review_vel_14",
        "rating_mean_30",
        "listing_quality",
    }
    assert expected_cols.issubset(features.columns)
    assert features["listing_quality"].max() <= 1.0
