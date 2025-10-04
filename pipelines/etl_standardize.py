"""Standardize raw time series data."""
from __future__ import annotations

import argparse
from datetime import date
from typing import Iterable

import pandas as pd

from utils.logging import get_logger
from utils.validators import DataValidator

logger = get_logger(__name__)


REQUIRED_COLUMNS = [
    "asin",
    "site",
    "dt",
    "price",
    "bsr",
    "rating",
    "reviews_count",
    "stock_est",
    "buybox_seller",
]


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values("_ingested_at").drop_duplicates(
        subset=["asin", "site", "dt"], keep="last"
    )
    price_valid = df["price"].apply(lambda v: v is not None and v > 0)
    bsr_valid = df["bsr"].apply(lambda v: v is not None and 0 < v < 5_000_000)

    df["price_valid"] = price_valid.astype(object)
    df["bsr_valid"] = bsr_valid.astype(object)

    df["price"] = df["price"].astype(float).astype(object)
    df["bsr"] = df["bsr"].astype(float).astype(object)
    df["rating"] = df["rating"].astype(float)
    df.loc[~price_valid, "price"] = None
    df.loc[~bsr_valid, "bsr"] = None
    df["dt"] = pd.to_datetime(df["dt"]).dt.normalize()
    return df


def run(raw_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    if combined.empty:
        return combined

    standardized = standardize(combined)
    validator = DataValidator()
    result = validator.validate_timeseries(standardized, REQUIRED_COLUMNS)
    logger.info("Standardization validation status: %s", result.status)
    return standardized


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize raw timeseries data")
    parser.add_argument("--date", default=date.today().isoformat())
    _ = parser.parse_args()
    logger.info("Run ETL standardization for date %s", _)


if __name__ == "__main__":
    main()
