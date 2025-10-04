"""Build training labels and document Parquet dependencies.

This module expects a Parquet serializer such as `pyarrow` (preferred) or
`fastparquet` to be installed. Call :func:`ensure_parquet_engine` before
writing Parquet output to surface a helpful error message when the runtime is
missing the required dependency.
"""
from __future__ import annotations

from importlib import import_module
from typing import Tuple

SUPPORTED_PARQUET_ENGINES: Tuple[str, ...] = ("pyarrow", "fastparquet")


def ensure_parquet_engine(engine: str = "pyarrow") -> None:
    """Ensure the configured Parquet engine is available.

    Parameters
    ----------
    engine:
        Name of the Parquet engine to check. Defaults to ``"pyarrow"``.

    Raises
    ------
    RuntimeError
        If the specified engine cannot be imported. The error message lists
        the supported choices so that users know which extra dependencies to
        install.
    """

    try:
        import_module(engine)
    except ImportError as exc:  # pragma: no cover - exercised indirectly
        options = ", ".join(SUPPORTED_PARQUET_ENGINES)
        raise RuntimeError(
            "Missing dependency for Parquet export. Install one of: " f"{options}."
        ) from exc
