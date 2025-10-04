import json

import numpy as np
import pandas as pd

from training.build_labels import LabelBuilder


def _sample_settings():
    return {
        "training": {
            "labels": {
                "lookback_days": 5,
                "horizon_days": 5,
                "ratio_threshold": 1.1,
                "percentile_drop": 0.0,
                "bsr_delta": 0.1,
                "min_coverage": 0.6,
                "min_past_sales": 0.5,
                "price_band_quantiles": [0.0, 0.5, 1.0],
            },
            "splits": {"train": 0.6, "valid": 0.2, "test": 0.2},
        }
    }


def _build_mart_dataframe():
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    rows = []
    for idx, dt in enumerate(dates):
        rows.append(
            {
                "asin": "A1",
                "site": "US",
                "dt": dt,
                "price": 20.0,
                "bsr": max(1, 100 - idx),
                "sales": 10 + idx,
                "category": "cat1",
            }
        )
        rows.append(
            {
                "asin": "B2",
                "site": "US",
                "dt": dt,
                "price": 18.0,
                "bsr": 400 + idx,
                "sales": 5.0,
                "category": "cat1",
            }
        )
    return pd.DataFrame(rows)


def _build_features_dataframe(mart_df: pd.DataFrame) -> pd.DataFrame:
    features = mart_df[["asin", "site", "dt"]].copy()
    features["bsr_trend_7"] = np.linspace(0.1, 0.5, len(features))
    features["price_vol_30"] = 0.05
    features["listing_quality"] = 0.8
    return features


def test_label_builder_generates_expected_outputs():
    mart_df = _build_mart_dataframe()
    features_df = _build_features_dataframe(mart_df)

    builder = LabelBuilder(settings=_sample_settings())
    outputs = builder.build(mart_df, features_df)

    assert set(outputs.keys()) == {"bin", "rank", "reg"}
    for name, df in outputs.items():
        assert not df.empty, f"Output {name} should not be empty"
        assert {"asin", "site", "t_ref", "split"}.issubset(df.columns)
        assert df["split"].isin({"train", "valid", "test"}).all()
        json.loads(df.iloc[0]["feature_vector"])  # should be valid JSON
        meta = df.iloc[0]["meta"]
        assert "cat" in json.dumps(meta) or "category" in json.dumps(meta)

    bin_df = outputs["bin"]
    assert bin_df["y"].isin({0, 1}).all()
    assert bin_df.groupby("asin")["y"].sum()["A1"] > 0  # A1 should have positive labels

    rank_df = outputs["rank"]
    assert rank_df["y"].between(0, 1).all()
    assert rank_df["group_id"].nunique() == 1

    reg_df = outputs["reg"]
    assert reg_df["y"].ge(0).all()
