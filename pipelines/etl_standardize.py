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

    price_numeric = pd.to_numeric(df["price"], errors="coerce")
    rating_numeric = pd.to_numeric(df["rating"], errors="coerce")
    bsr_numeric = pd.to_numeric(df["bsr"], errors="coerce")

    price_valid = price_numeric.notna() & (price_numeric > 0)
    bsr_valid = bsr_numeric.notna() & (bsr_numeric > 0) & (bsr_numeric < 5_000_000)

    df = df.assign(
        price=price_numeric,
        rating=rating_numeric,
        bsr=bsr_numeric,
        price_valid=price_valid,
        bsr_valid=bsr_valid,
    )

    df.loc[~df["price_valid"], "price"] = None
    df.loc[~df["bsr_valid"], "bsr"] = None

    df["price"] = df["price"].astype(object)
    df["bsr"] = df["bsr"].astype(object)
    df.loc[~df["price_valid"], "price"] = None
    df.loc[~df["bsr_valid"], "bsr"] = None
    df["price_valid"] = df["price_valid"].astype(object)
    df["bsr_valid"] = df["bsr_valid"].astype(object)

    df["dt"] = pd.to_datetime(df["dt"]).dt.date
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
