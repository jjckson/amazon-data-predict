"""Helpers for constructing supervised learning targets."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import datetime as dt
from typing import Any

import numpy as np
import pandas as pd


def _prepare_frame(mart_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the mart frame prior to label construction."""

    if mart_df.empty:
        return mart_df.copy()

    prepared = mart_df.copy()
    if "dt" in prepared.columns:
        prepared["dt"] = pd.to_datetime(prepared["dt"])
    if "bsr" in prepared.columns:
        prepared["bsr"] = pd.to_numeric(prepared["bsr"], errors="coerce")
    sort_columns = [col for col in ["asin", "site", "dt"] if col in prepared.columns]
    if sort_columns:
        prepared = prepared.sort_values(sort_columns)  # type: ignore[assignment]
    return prepared


def build_labels(mart_df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """Create future looking labels for BSR performance."""

    prepared = _prepare_frame(mart_df)
    if prepared.empty:
        prepared = prepared.copy()
        prepared["future_bsr"] = pd.Series(dtype="float64")
        prepared["bsr_improved"] = pd.Series(dtype="bool")
        return prepared

    horizon = max(int(horizon_days), 1)
    prepared = prepared.copy()
    group_keys = [col for col in ["asin", "site"] if col in prepared.columns]
    if group_keys:
        prepared["future_bsr"] = (
            prepared.groupby(group_keys, sort=False)["bsr"].shift(-horizon)
        )
    else:
        prepared["future_bsr"] = prepared["bsr"].shift(-horizon)
    comparison = prepared["future_bsr"] < prepared["bsr"]
    prepared["bsr_improved"] = comparison.fillna(False)
    return prepared


def _compute_rank_target(group: pd.DataFrame) -> pd.DataFrame:
    """Assign a percentile rank target based on future sales.

    The input ``group`` is expected to contain a ``future_sales`` column. Rows
    are ranked in descending order such that the highest ``future_sales`` value
    receives the largest percentile. The resulting percentile is stored in the
    ``y_rank`` column alongside the original data.
    """

    if "future_sales" not in group:
        raise KeyError("future_sales column is required to compute rank target")

    ranked = group.copy()
    order = ranked["future_sales"].rank(method="min", ascending=False)
    max_rank = order.max()
    if pd.isna(max_rank) or max_rank == 0:
        ranked["y_rank"] = pd.NA
        return ranked

    ranked["y_rank"] = (max_rank - order + 1) / max_rank
    return ranked


def _normalise_for_json(value: Any) -> Any:
    """Normalise objects so that they can be serialised via ``json.dumps``.

    The helper mirrors the behaviour expected by the training jobs where
    metadata values may include scalars, sequences or nested mappings. Pandas
    ``isna`` cannot be called on list-like objects because it tries to evaluate
    their truthiness which raises ``ValueError``. To avoid that we convert any
    iterable structures before falling back to ``pd.isna`` for scalar values.
    """

    if value is None:
        return None

    if isinstance(value, (str, bytes, bytearray)):
        return value

    if isinstance(value, Mapping):
        return {str(key): _normalise_for_json(item) for key, item in value.items()}

    if isinstance(value, pd.Series):
        return [_normalise_for_json(item) for item in value.tolist()]

    if isinstance(value, pd.Index):
        return [_normalise_for_json(item) for item in value.tolist()]

    if isinstance(value, np.ndarray):
        return [_normalise_for_json(item) for item in value.tolist()]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise_for_json(item) for item in list(value)]

    if isinstance(value, set):
        return [_normalise_for_json(item) for item in list(value)]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    if isinstance(value, (np.bool_,)):  # ``json`` cannot serialise numpy bools
        return bool(value)

    return value
