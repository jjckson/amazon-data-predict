from __future__ import annotations

import json

import pandas as pd
import pytest

from training.build_labels import (
    LabelWindows,
    Thresholds,
    build_labels,
    _normalise_for_json,
)


def test_empty_windows_returns_empty_tables() -> None:
    mart_df = pd.DataFrame(
        {
            "asin": ["A"],
            "site": ["US"],
            "dt": ["2021-01-01"],
            "sales": [2],
        }
    )

    result = build_labels(mart_df, windows=LabelWindows(observation_days=3, forecast_days=2))

    assert result.samples.empty
    assert result.train_samples_bin.empty
    assert result.train_samples_rank.empty
    assert result.train_samples_reg.empty
    assert set(result.reports.keys()) == {
        "balance",
        "coverage",
        "missingness",
        "mean_encoders",
        "global_stats",
    }


def test_sparse_sales_coverage_and_targets() -> None:
    mart_df = pd.DataFrame(
        {
            "asin": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
            ],
            "site": ["US"] * 8,
            "dt": [
                "2021-01-01",
                "2021-01-03",
                "2021-01-04",
                "2021-01-06",
                "2021-01-01",
                "2021-01-03",
                "2021-01-04",
                "2021-01-06",
            ],
            "sales": [5, 7, 4, 8, 0, 0, 0, 0],
        }
    )

    result = build_labels(
        mart_df,
        windows=LabelWindows(observation_days=3, forecast_days=2),
        thresholds=Thresholds(r=6, p=0.95, delta=3),
    )

    samples = result.samples.sort_values(["asin", "t_ref"]).reset_index(drop=True)
    samples_a = samples[samples["asin"] == "A"].reset_index(drop=True)
    assert len(samples_a) == 2
    first, second = samples_a.iloc[0], samples_a.iloc[1]

    assert pytest.approx(first.past_coverage, rel=1e-6) == 2 / 3
    assert pytest.approx(first.future_coverage, rel=1e-6) == 0.5
    assert first.future_sales == 4
    assert first.y_bin == 0

    assert pytest.approx(second.past_coverage, rel=1e-6) == 2 / 3
    assert pytest.approx(second.future_coverage, rel=1e-6) == 0.5
    assert second.future_sales == 8
    assert second.y_bin == 1

    assert set(result.train_samples_reg["asin"]) == set(samples["asin"])
    assert all(result.train_samples_reg["split"] == "train")
    assert all(samples["site_mean_y_reg"] == samples["split_mean_y_reg"])


def test_percentile_ties_share_rank() -> None:
    mart_df = pd.DataFrame(
        {
            "asin": ["A", "A", "B", "B"],
            "site": ["US"] * 4,
            "dt": [
                "2021-01-01",
                "2021-01-02",
                "2021-01-01",
                "2021-01-02",
            ],
            "sales": [1, 5, 2, 5],
        }
    )

    result = build_labels(
        mart_df,
        windows=LabelWindows(observation_days=1, forecast_days=1),
        thresholds=Thresholds(r=10, p=0.5, delta=1),
    )

    samples = result.samples.sort_values(["t_ref", "asin"]).reset_index(drop=True)
    assert len(samples) == 2
    assert pytest.approx(samples.loc[0, "y_rank"]) == pytest.approx(samples.loc[1, "y_rank"])
    assert samples.loc[0, "y_rank"] == pytest.approx(0.75)


def test_feature_leakage_guard() -> None:
    mart_df = pd.DataFrame(
        {
            "asin": ["A"] * 3,
            "site": ["US"] * 3,
            "dt": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "sales": [1, 1, 1],
        }
    )

    features_df = pd.DataFrame(
        {
            "asin": ["A", "A"],
            "site": ["US", "US"],
            "dt": ["2021-01-01", "2021-01-03"],
            "feat": [10, 99],
        }
    )

    result = build_labels(
        mart_df,
        features_df=features_df,
        windows=LabelWindows(observation_days=1, forecast_days=1),
    )

    samples = result.samples.sort_values("t_ref").reset_index(drop=True)
    assert len(samples) == 2
    second_features = samples.loc[1, "feature_vector"]
    assert second_features == {"feat": 10}


def test_meta_serialised_as_string_round_trip() -> None:
    meta = pd.Series(
        [
            {
                "features": ["wireless", "bluetooth"],
                "dimensions": {"width": 2.3, "height": [1, 2, None]},
            },
            {"features": []},
        ]
    )

    serialised = meta.apply(lambda value: json.dumps(_normalise_for_json(value)))
    round_tripped = serialised.apply(json.loads)
    expected = meta.apply(_normalise_for_json)

    assert list(round_tripped) == list(expected)
