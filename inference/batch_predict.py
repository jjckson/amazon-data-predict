"""Batch prediction helpers with clear Parquet dependency messaging."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from utils.logging import get_logger
from utils.parquet import ensure_parquet_engine, normalize_engine_preference

logger = get_logger(__name__)


def format_predictions(raw_predictions: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate prediction frames and standardise column order."""

    frames = [frame for frame in raw_predictions if frame is not None]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        return combined

    preferred_order = [col for col in ["asin", "site", "score", "dt"] if col in combined.columns]
    ordered = combined[preferred_order + [c for c in combined.columns if c not in preferred_order]]
    return ordered


def save_predictions(predictions: pd.DataFrame, output: Path, engine: str = "pyarrow") -> None:
    """Write predictions to Parquet after ensuring the engine is installed."""

    preferred = normalize_engine_preference([engine])
    ensure_parquet_engine(preferred)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output, engine=preferred, index=False)
    logger.info("Saved %d predictions to %s", len(predictions), output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predict explosive scores")
    parser.add_argument("--input", nargs="+", help="Prediction CSV files to combine")
    parser.add_argument("--output", required=True, type=Path, help="Destination Parquet file")
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
        logger.info("Loading predictions from %s", path)
        frames.append(pd.read_csv(path))

    formatted = format_predictions(frames)
    save_predictions(formatted, args.output, engine=engine)


if __name__ == "__main__":
    main()
