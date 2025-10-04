"""Helpers for constructing supervised learning targets."""
from __future__ import annotations

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
