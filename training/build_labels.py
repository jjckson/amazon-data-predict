"""Build daily training labels with explicit Parquet dependency checks.

The label builder writes Parquet outputs which requires an engine such as
``pyarrow`` (preferred) or ``fastparquet``. Use :func:`ensure_parquet_engine`
before attempting to persist results so that missing optional dependencies are
surfaced with a clear message.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from utils.logging import get_logger
from utils.parquet import ensure_parquet_engine, normalize_engine_preference
from utils.validators import DataValidator

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "asin",
    "site",
    "dt",
    "bsr",
    "price",
    "reviews_count",
]


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sanitized copy of the standardised frame."""

    if frame.empty:
        return frame.copy()

    prepared = frame.copy()
    prepared["dt"] = pd.to_datetime(prepared["dt"]).dt.date
    prepared = prepared.sort_values(["asin", "site", "dt", "_ingested_at"], ascending=True)
    prepared = prepared.drop_duplicates(subset=["asin", "site", "dt"], keep="last")
    return prepared


def build_labels(standardised: pd.DataFrame) -> pd.DataFrame:
    """Generate next-day BSR improvement labels for each ASIN/site combination."""

    if standardised.empty:
        return standardised.assign(future_bsr=pd.Series(dtype="float"), label=pd.Series(dtype="int64"))

    missing = [col for col in REQUIRED_COLUMNS if col not in standardised.columns]
    if missing:
        raise ValueError(f"Missing required columns for label generation: {missing}")

    prepared = _prepare_frame(standardised)
    group_keys = ["asin", "site"]
    future_bsr = prepared.groupby(group_keys)["bsr"].shift(-1)
    label = (future_bsr.notna() & (future_bsr < prepared["bsr"])).astype("int64")

    result = prepared.assign(future_bsr=future_bsr, label=label)
    return result


def run(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine standardised frames and generate labels with validation."""

    materialized: List[pd.DataFrame] = [f for f in frames if f is not None]
    combined = pd.concat(materialized, ignore_index=True) if materialized else pd.DataFrame()
    if combined.empty:
        return combined

    labels = build_labels(combined)
    validator = DataValidator()
    validation = validator.validate_timeseries(labels, REQUIRED_COLUMNS + ["label"])
    logger.info("Label validation status: %s", validation.status)
    return labels


def save_labels(labels: pd.DataFrame, output: Path, engine: str = "pyarrow") -> None:
    """Persist labels to Parquet after verifying the selected engine is installed."""

    preferred = normalize_engine_preference([engine])
    ensure_parquet_engine(preferred)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output, engine=preferred, index=False)
    logger.info("Wrote %d label rows to %s", len(labels), output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training labels from standardised data")
    parser.add_argument("--input", nargs="+", help="Input Parquet files containing standardised metrics")
    parser.add_argument("--output", required=True, type=Path, help="Destination Parquet file for labels")
    parser.add_argument(
        "--engine",
        default="pyarrow",
        help="Preferred Parquet engine (pyarrow or fastparquet). Defaults to pyarrow.",
    )

    args = parser.parse_args()
    engine = normalize_engine_preference([args.engine])
    ensure_parquet_engine(engine)

    frames = []
    for path_str in args.input or []:
        path = Path(path_str)
        logger.info("Loading standardised data from %s", path)
        frames.append(pd.read_parquet(path, engine=engine))

    labels = run(frames)
    save_labels(labels, args.output, engine=engine)


if __name__ == "__main__":
    main()
