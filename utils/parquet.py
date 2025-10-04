"""Utilities for working with Parquet serializers."""
from __future__ import annotations

from importlib import import_module
from typing import Iterable, Tuple

SUPPORTED_PARQUET_ENGINES: Tuple[str, ...] = ("pyarrow", "fastparquet")


def ensure_parquet_engine(engine: str = "pyarrow") -> None:
    """Ensure the configured Parquet engine can be imported.

    Parameters
    ----------
    engine:
        Name of the Parquet engine to check. Defaults to ``"pyarrow"``.

    Raises
    ------
    RuntimeError
        If the specified engine cannot be imported. The error lists the
        supported options to guide dependency installation.
    """

    try:
        import_module(engine)
    except ImportError as exc:  # pragma: no cover - exercised indirectly
        options = ", ".join(SUPPORTED_PARQUET_ENGINES)
        raise RuntimeError(
            "Missing dependency for Parquet support. Install one of: " f"{options}."
        ) from exc


def normalize_engine_preference(preference: Iterable[str] | None = None) -> str:
    """Return the first supported engine from the provided preference list.

    Parameters
    ----------
    preference:
        Optional ordered list of preferred engines. ``None`` defaults to the
        recommended ``("pyarrow", "fastparquet")`` ordering.

    Returns
    -------
    str
        The first engine from ``preference`` that is supported.
    """

    candidates = list(preference or SUPPORTED_PARQUET_ENGINES)
    for candidate in candidates:
        if candidate in SUPPORTED_PARQUET_ENGINES:
            return candidate
    return SUPPORTED_PARQUET_ENGINES[0]
