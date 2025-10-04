"""Data validation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd


@dataclass
class ValidationResult:
    status: str
    details: dict


class DataValidator:
    """Simple validation routines for pipeline outputs."""

    def __init__(self, coverage_threshold: float = 0.95) -> None:
        self.coverage_threshold = coverage_threshold

    def validate_timeseries(
        self,
        df: pd.DataFrame,
        required_columns: Iterable[str],
    ) -> ValidationResult:
        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            return ValidationResult("failed", {"missing_columns": missing_cols})

        df_sorted = df.sort_values("dt")
        coverage = df_sorted["dt"].diff().dt.days.fillna(1).eq(1).mean()
        status = "passed" if coverage >= self.coverage_threshold else "warning"

        anomalies = {}
        if "price" in df.columns:
            anomalies["invalid_price"] = int((df["price"] <= 0).sum())
        if "bsr" in df.columns:
            anomalies["invalid_bsr"] = int(
                ((df["bsr"] <= 0) | (df["bsr"] >= 5_000_000)).sum()
            )

        return ValidationResult(status, {"coverage": coverage, **anomalies})

    def validate_consistency(
        self,
        raw_counts: Mapping[str, int],
        processed_counts: Mapping[str, int],
        tolerance: float = 0.01,
    ) -> ValidationResult:
        mismatches = {}
        for key, raw_count in raw_counts.items():
            processed = processed_counts.get(key, 0)
            if raw_count == 0:
                continue
            delta = abs(raw_count - processed) / raw_count
            if delta > tolerance:
                mismatches[key] = {"raw": raw_count, "processed": processed}
        status = "passed" if not mismatches else "warning"
        return ValidationResult(status, mismatches)
