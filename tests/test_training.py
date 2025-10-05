from __future__ import annotations

import json

import numpy as np
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
        "bsr_availability",
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
            "bsr_percentile": [0.7, 0.6, 0.55, 0.4, 0.95, 0.96, 0.97, 0.98],
        }
    )

    facts_df = pd.DataFrame(
        {
            "asin": ["A", "B"],
            "site": ["US", "US"],
            "category_id": ["c1", "c1"],
            "price_band": ["premium", "value"],
        }
    )

    result = build_labels(
        mart_df,
        facts_df=facts_df,
        windows=LabelWindows(observation_days=3, forecast_days=2),
        thresholds=Thresholds(r=0.5, p=0.1, delta=1),
    )

    samples = result.samples.sort_values(["asin", "t_ref"]).reset_index(drop=True)
    samples_a = samples[samples["asin"] == "A"].reset_index(drop=True)
    assert len(samples_a) == 2
    first, second = samples_a.iloc[0], samples_a.iloc[1]

    assert pytest.approx(first.past_coverage, rel=1e-6) == 2 / 3
    assert pytest.approx(first.future_coverage, rel=1e-6) == 0.5
    assert first.future_sales == 4
    assert first.y_bin == 0
    assert first.sales_ratio == pytest.approx(4 / 12)
    assert first.group_id == "US|c1|premium"

    assert pytest.approx(second.past_coverage, rel=1e-6) == 2 / 3
    assert pytest.approx(second.future_coverage, rel=1e-6) == 0.5
    assert second.future_sales == 8
    assert second.y_bin == 1
    assert second.sales_ratio == pytest.approx(8 / 11)
    assert second.bsr_percentile_drop == pytest.approx(0.175, rel=1e-6)

    assert set(result.train_samples_reg["asin"]) == set(samples["asin"])
    assert all(result.train_samples_reg["split"] == "train")
    assert all(samples["site_mean_y_reg"] == samples["split_mean_y_reg"])


def test_group_id_includes_site_for_cross_site_separation() -> None:
    mart_df = pd.DataFrame(
        {
            "asin": [
                "US_A",
                "US_A",
                "US_B",
                "US_B",
                "UK_A",
                "UK_A",
                "UK_B",
                "UK_B",
            ],
            "site": [
                "US",
                "US",
                "US",
                "US",
                "UK",
                "UK",
                "UK",
                "UK",
            ],
            "dt": [
                "2021-01-01",
                "2021-01-02",
                "2021-01-01",
                "2021-01-02",
                "2021-01-01",
                "2021-01-02",
                "2021-01-01",
                "2021-01-02",
            ],
            "sales": [10, 20, 15, 5, 7, 3, 9, 12],
        }
    )

    facts_df = pd.DataFrame(
        {
            "asin": ["US_A", "US_B", "UK_A", "UK_B"],
            "site": ["US", "US", "UK", "UK"],
            "category_id": ["c1", "c1", "c1", "c1"],
            "price_band": ["mid", "mid", "mid", "mid"],
        }
    )

    result = build_labels(
        mart_df,
        facts_df=facts_df,
        windows=LabelWindows(observation_days=1, forecast_days=1),
        thresholds=Thresholds(r=0.1, p=0.1, delta=1),
    )

    samples = result.samples.sort_values(["site", "asin"]).reset_index(drop=True)
    assert set(samples["group_id"]) == {"US|c1|mid", "UK|c1|mid"}

    counts = samples.groupby(["group_id", "t_ref"])["asin"].count().tolist()
    assert counts == [2, 2]

    sites_in_group = samples["group_id"].str.split("|", n=1, expand=True)
    assert set(sites_in_group[0]) == {"US", "UK"}


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
            "bsr_rank": [200, 150, 220, 150],
        }
    )

    facts_df = pd.DataFrame(
        {
            "asin": ["A", "B"],
            "site": ["US", "US"],
            "category_id": ["c1", "c1"],
        }
    )

    result = build_labels(
        mart_df,
        facts_df=facts_df,
        windows=LabelWindows(observation_days=1, forecast_days=1),
        thresholds=Thresholds(r=0.1, p=0.1, delta=5),
    )

    samples = result.samples.sort_values(["t_ref", "asin"]).reset_index(drop=True)
    assert len(samples) == 2
    assert pytest.approx(samples.loc[0, "y_rank"]) == pytest.approx(samples.loc[1, "y_rank"])
    assert samples.loc[0, "y_rank"] == pytest.approx(0.75)


def test_rank_based_positive_with_missing_percentile() -> None:
    mart_df = pd.DataFrame(
        {
            "asin": ["A", "A", "A"],
            "site": ["US", "US", "US"],
            "dt": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "sales": [0, 2, 6],
            "bsr_rank": [500, 400, 200],
        }
    )

    facts_df = pd.DataFrame(
        {
            "asin": ["A"],
            "site": ["US"],
            "category_id": ["c2"],
        }
    )

    result = build_labels(
        mart_df,
        facts_df=facts_df,
        windows=LabelWindows(observation_days=2, forecast_days=1),
        thresholds=Thresholds(r=0.5, p=0.2, delta=150),
    )

    samples = result.samples.sort_values("t_ref").reset_index(drop=True)
    assert len(samples) == 1
    row = samples.iloc[0]
    assert row.sales_ratio == pytest.approx(6 / 2)
    assert pd.isna(row.bsr_percentile_drop)
    assert row.bsr_rank_improvement == pytest.approx(250)
    assert row.y_bin == 1


def test_default_threshold_requires_bsr_signal() -> None:
    mart_df = pd.DataFrame(
        {
            "asin": ["A", "A", "A", "A"],
            "site": ["US", "US", "US", "US"],
            "dt": [
                "2021-01-01",
                "2021-01-02",
                "2021-01-03",
                "2021-01-04",
            ],
            "sales": [10, 10, 40, 40],
            "bsr_percentile": [0.4, 0.4, 0.4, 0.4],
            "bsr_rank": [100, 100, 100, 100],
        }
    )

    result = build_labels(mart_df, windows=LabelWindows(observation_days=2, forecast_days=1))

    samples = result.samples.sort_values("t_ref").reset_index(drop=True)
    assert len(samples) == 2
    row = samples.iloc[0]
    assert row.sales_ratio == pytest.approx(40 / 20)
    assert row.bsr_percentile_drop == pytest.approx(0.0)
    assert row.bsr_rank_improvement == pytest.approx(0.0)
    assert row.y_bin == 0


def test_rank_signal_allows_neutral_ratio() -> None:
    mart_df = pd.DataFrame(
        {
            "asin": ["A", "A", "A"],
            "site": ["US", "US", "US"],
            "dt": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "sales": [0, 0, 0],
            "bsr_rank": [300, 200, 100],
        }
    )

    result = build_labels(mart_df, windows=LabelWindows(observation_days=2, forecast_days=1))

    samples = result.samples.sort_values("t_ref").reset_index(drop=True)
    assert len(samples) == 1
    row = samples.iloc[0]
    assert np.isnan(row.sales_ratio)
    assert row.bsr_rank_improvement == pytest.approx(150.0)
    assert row.y_bin == 1


def test_zero_past_sales_results_in_neutral_ratio() -> None:
    mart_df = pd.DataFrame(
        {
            "asin": ["A", "A"],
            "site": ["US", "US"],
            "dt": ["2021-01-01", "2021-01-02"],
            "sales": [0, 0],
            "bsr_percentile": [0.9, 0.8],
        }
    )

    facts_df = pd.DataFrame(
        {
            "asin": ["A"],
            "site": ["US"],
            "category_id": ["c3"],
        }
    )

    result = build_labels(
        mart_df,
        facts_df=facts_df,
        windows=LabelWindows(observation_days=1, forecast_days=1),
        thresholds=Thresholds(r=0.1, p=0.05, delta=10),
    )

    samples = result.samples
    assert len(samples) == 1
    row = samples.iloc[0]
    assert pd.isna(row.sales_ratio)
    assert row.y_bin == 1


def test_missing_bsr_fields_prevent_positive_label() -> None:
    mart_df = pd.DataFrame(
        {
            "asin": ["A", "A", "A"],
            "site": ["US", "US", "US"],
            "dt": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "sales": [1, 1, 10],
        }
    )

    facts_df = pd.DataFrame(
        {
            "asin": ["A"],
            "site": ["US"],
            "category_id": ["c4"],
        }
    )

    result = build_labels(
        mart_df,
        facts_df=facts_df,
        windows=LabelWindows(observation_days=2, forecast_days=1),
        thresholds=Thresholds(r=1.5, p=0.1, delta=20),
    )

    samples = result.samples
    assert len(samples) == 1
    row = samples.iloc[0]
    assert row.sales_ratio > 1.5
    assert pd.isna(row.bsr_percentile_drop)
    assert pd.isna(row.bsr_rank_improvement)
    assert row.y_bin == 0


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
