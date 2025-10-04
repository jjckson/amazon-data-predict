"""Baseline scoring pipeline."""
from __future__ import annotations

import argparse
import json
from datetime import date
from typing import Dict

import numpy as np
import pandas as pd

from utils.config import load_settings
from utils.logging import get_logger

logger = get_logger(__name__)


def _robust_z(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        mad = 1e-9
    return (series - median) / mad


def score(features: pd.DataFrame, settings: Dict, rank_min: int = 20) -> pd.DataFrame:
    if features.empty:
        return features

    weights = settings["scoring"]["weights"]
    required = [
        "asin",
        "site",
        "dt",
        "category",
        "bsr_trend_30",
        "est_sales_30",
        "review_vel_14",
        "price_vol_30",
        "listing_quality",
    ]
    for col in required:
        if col not in features.columns:
            features[col] = np.nan

    grouped = []
    for (site, category), group in features.groupby(["site", "category"], dropna=False):
        z_scores = {}
        for key in weights.keys():
            metric = key
            if metric == "price_vol_30":
                z_series = -_robust_z(group[metric])
            else:
                z_series = _robust_z(group[metric])
            z_scores[metric] = z_series

        total = sum(weights[m] * z_scores[m] for m in weights)
        df = group[["asin", "site", "dt", "category"]].copy()
        df["explosive_score"] = total
        df["reason"] = df.apply(
            lambda row: json.dumps(
                {
                    "w": weights,
                    "z": {metric: float(z_scores[metric].loc[row.name]) for metric in weights},
                }
            ),
            axis=1,
        )
        df["rank_in_cat"] = (
            df["explosive_score"].rank(ascending=False, method="min")
        )
        df.loc[df["rank_in_cat"] > rank_min, "rank_in_cat"] = np.nan
        grouped.append(df)
    return pd.concat(grouped, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline scoring")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    settings = load_settings()
    logger.info("Scoring for %s", args.date)
    _ = settings


if __name__ == "__main__":
    main()
