"""Feature engineering for ASIN daily metrics."""
from __future__ import annotations

import argparse
import math
from datetime import date
from typing import Dict

import numpy as np
import pandas as pd

from utils.config import load_settings
from utils.logging import get_logger

logger = get_logger(__name__)


def _rolling_slope(values: pd.Series) -> float:
    if values.isnull().any():
        return np.nan
    y = values.to_numpy(dtype=float)
    x = np.arange(len(y))
    x_mean = x.mean()
    y_mean = y.mean()
    denominator = ((x - x_mean) ** 2).sum()
    if denominator == 0:
        return 0.0
    slope = ((x - x_mean) * (y - y_mean)).sum() / denominator
    return float(slope)


def _compute_listing_quality(row: pd.Series) -> float:
    title_score = min(row.get("title_length", 0) / 80, 1.0)
    bullets_score = min(row.get("bullet_count", 0) / 6, 1.0)
    media_score = 1.0 if row.get("image_count", 0) >= 7 else row.get("image_count", 0) / 7
    attrs_flags = [
        row.get("has_material_attr", False),
        row.get("has_size_attr", False),
        row.get("has_use_case_attr", False),
    ]
    attrs_score = sum(1 for flag in attrs_flags if flag) / len(attrs_flags)
    has_a_plus = 1.0 if row.get("has_a_plus", False) else 0.0
    return 0.25 * title_score + 0.25 * bullets_score + 0.25 * media_score + 0.25 * ((attrs_score + has_a_plus) / 2)


def build_features(mart_df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    if mart_df.empty:
        return mart_df

    mart_df = mart_df.sort_values(["asin", "site", "dt"])  # type: ignore[assignment]
    mart_df["dt"] = pd.to_datetime(mart_df["dt"])

    features = []
    rolling_windows = settings["features"]["rolling"]

    for (asin, site), group in mart_df.groupby(["asin", "site"], sort=False):
        group = group.sort_values("dt")
        group_features = group[["asin", "site", "dt"]].copy()

        for window in rolling_windows:
            min_periods = math.ceil(window * 0.7)
            inv_bsr = group["bsr"].replace(0, np.nan).dropna()
            slope_series = (
                group["bsr"].apply(lambda v: np.nan if not v else 1 / v)
                .rolling(window=window, min_periods=min_periods)
                .apply(_rolling_slope, raw=False)
            )
            group_features[f"bsr_trend_{window}"] = slope_series

            price_vol = group["price"].rolling(window=window, min_periods=min_periods)
            group_features[f"price_vol_{window}"] = price_vol.std() / price_vol.mean()

            rating_mean = group["rating"].rolling(window=window, min_periods=min_periods).mean()
            group_features[f"rating_mean_{window}"] = rating_mean

        group_features["review_vel_14"] = (
            group["reviews_count"].diff(14)
        )
        group_features["est_sales_30"] = (
            group["est_sales"].rolling(window=30, min_periods=math.ceil(30 * 0.7)).mean()
            if "est_sales" in group.columns
            else np.nan
        )

        meta_cols = [
            "title_length",
            "bullet_count",
            "image_count",
            "has_material_attr",
            "has_size_attr",
            "has_use_case_attr",
            "has_a_plus",
        ]
        for col in meta_cols:
            if col not in group.columns:
                group[col] = np.nan

        group_features["listing_quality"] = group.apply(_compute_listing_quality, axis=1)
        features.append(group_features)

    feature_df = pd.concat(features, ignore_index=True)
    return feature_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily ASIN features")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    settings = load_settings()
    logger.info("Building features for %s", args.date)
    _ = settings


if __name__ == "__main__":
    main()
