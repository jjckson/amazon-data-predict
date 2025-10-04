"""Batch inference utilities with explicit Parquet dependency messaging.

Predicted outputs are written to Parquet, which requires a serializer such as
`pyarrow` (default) or `fastparquet`. Use :func:`ensure_parquet_engine` to
validate the chosen engine before attempting to serialise results.
"""
from __future__ import annotations

from importlib import import_module
from typing import Tuple

SUPPORTED_PARQUET_ENGINES: Tuple[str, ...] = ("pyarrow", "fastparquet")


def ensure_parquet_engine(engine: str = "pyarrow") -> None:
    """Check that the requested Parquet engine can be imported."""

    try:
        import_module(engine)
    except ImportError as exc:  # pragma: no cover - exercised indirectly
        options = ", ".join(SUPPORTED_PARQUET_ENGINES)
        raise RuntimeError(
            "Missing dependency for Parquet export. Install one of: " f"{options}."
        ) from exc
