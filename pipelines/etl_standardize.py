"""Standardize raw time series data."""
from __future__ import annotations

import argparse
import math
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

def _is_positive(value: object) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _is_bsr_valid(value: object) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    try:
        val = float(value)
    except (TypeError, ValueError):
        return False
    return 0 < val < 5_000_000


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values("_ingested_at").drop_duplicates(
        subset=["asin", "site", "dt"], keep="last"
    )
    df["price_valid"] = df["price"].apply(_is_positive)
    df["bsr_valid"] = df["bsr"].apply(_is_bsr_valid)

    invalid_price = ~df["price_valid"]
    invalid_bsr = ~df["bsr_valid"]
    df.loc[invalid_price, "price"] = None
    df.loc[invalid_bsr, "bsr"] = None

    df["price"] = df["price"].astype(float)
    df["rating"] = df["rating"].astype(float)
    df["bsr"] = df["bsr"].astype(float).astype(object)
    df.loc[invalid_bsr, "bsr"] = None
    df["dt"] = pd.to_datetime(df["dt"]).dt.date
    df["price_valid"] = df["price_valid"].map(lambda x: True if bool(x) else False).astype(object)
    df["bsr_valid"] = df["bsr_valid"].map(lambda x: True if bool(x) else False).astype(object)
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
