"""Time window utilities."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, List


def rolling_window_dates(
    end_date: date,
    window: int,
    lookback: int | None = None,
) -> List[date]:
    """Return ordered list of dates covering the window."""
    effective_window = lookback or window
    start_date = end_date - timedelta(days=effective_window - 1)
    return [start_date + timedelta(days=i) for i in range(effective_window)]


def window_pairs(dates: Iterable[date], window: int) -> List[tuple[date, date]]:
    ordered = sorted(dates)
    results: List[tuple[date, date]] = []
    for idx in range(len(ordered)):
        start = ordered[idx]
        end = start + timedelta(days=window - 1)
        if end <= ordered[-1]:
            results.append((start, end))
    return results
