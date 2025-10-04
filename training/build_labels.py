"""Helpers for constructing supervised learning targets."""
from __future__ import annotations

import pandas as pd


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
