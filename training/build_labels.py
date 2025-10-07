"""Helpers for constructing supervised learning targets."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Mapping, Sequence
import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd


Window = tuple[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class LabelWindows:
    """Configuration for observation (W) and forecast (H) windows."""

    observation_days: int = 28
    forecast_days: int = 30

    def __post_init__(self) -> None:  # pragma: no cover - trivial validation
        if self.observation_days <= 0:
            raise ValueError("observation_days must be positive")
        if self.forecast_days <= 0:
            raise ValueError("forecast_days must be positive")


@dataclass(frozen=True)
class Thresholds:
    """Threshold parameters used by the different targets."""

    r: float = 1.6  # sales ratio threshold for the binary label
    p: float = 0.2  # BSR percentile drop threshold (0-1 scale) for the binary label
    delta: float = 1.0  # BSR rank improvement threshold for the binary label


@dataclass(frozen=True)
class SplitConfig:
    """Time based split ratios for Train/Valid/Test partitions."""

    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    names: tuple[str, str, str] = ("train", "valid", "test")

    def __post_init__(self) -> None:  # pragma: no cover - trivial validation
        if len(self.ratios) != 3 or len(self.names) != 3:
            raise ValueError("Exactly three ratios and names are required")
        if any(r < 0 for r in self.ratios):
            raise ValueError("Split ratios must be non-negative")
        if sum(self.ratios) <= 0:
            raise ValueError("At least one split must have a positive ratio")


@dataclass
class BuildLabelsResult:
    """Structured output of the label construction pipeline."""

    samples: pd.DataFrame
    train_samples_bin: pd.DataFrame
    train_samples_rank: pd.DataFrame
    train_samples_reg: pd.DataFrame
    reports: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        """Proxy DataFrame-like attribute access to :attr:`samples`.

        ``build_labels`` historically returned a ``pd.DataFrame`` and some
        consumers—including tests—still expect dataframe accessors such as
        ``iloc`` to be available on the result.  Delegating unknown attribute
        lookups keeps backwards compatibility while we expose the richer
        structured output.
        """

        try:
            return getattr(self.samples, name)
        except AttributeError as exc:  # pragma: no cover - defensive branch
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from exc

    def __len__(self) -> int:
        """Allow ``len(result)`` to reflect the underlying samples."""

        return len(self.samples)

    def __iter__(self):
        """Iterate over the sample columns like a DataFrame."""

        return iter(self.samples)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise common columns such as ``dt`` and ensure sorting."""

    if frame.empty:
        return frame.copy()

    prepared = frame.copy()
    if "dt" in prepared.columns:
        prepared["dt"] = pd.to_datetime(prepared["dt"])
    if "t_ref" in prepared.columns:
        prepared["t_ref"] = pd.to_datetime(prepared["t_ref"])
    if "sales" in prepared.columns:
        prepared["sales"] = pd.to_numeric(prepared["sales"], errors="coerce")
    sort_columns = [col for col in ["asin", "site", "dt"] if col in prepared.columns]
    if sort_columns:
        prepared = prepared.sort_values(sort_columns)  # type: ignore[assignment]
    return prepared


def _expected_window(reference: pd.Timestamp, days: int, *, include_reference: bool) -> Window:
    """Return the start/end timestamps for a rolling window."""

    if include_reference:
        start = reference - pd.Timedelta(days=days - 1)
        end = reference
    else:
        start = reference + pd.Timedelta(days=1)
        end = reference + pd.Timedelta(days=days)
    return (start, end)


def _slice_window(group: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Return the rows of ``group`` that fall inside ``window`` (inclusive)."""

    start, end = window
    return group.loc[(group["dt"] >= start) & (group["dt"] <= end)]


def _compute_window_aggregates(
    group: pd.DataFrame,
    t_ref: pd.Timestamp,
    windows: LabelWindows,
) -> Mapping[str, Any]:
    """Compute past/future aggregates for a single ``(asin, site)`` slice."""

    observation_window = _expected_window(t_ref, windows.observation_days, include_reference=True)
    forecast_window = _expected_window(t_ref, windows.forecast_days, include_reference=False)

    past_slice = _slice_window(group, observation_window)
    future_slice = _slice_window(group, forecast_window)

    expected_past_days = windows.observation_days
    expected_future_days = windows.forecast_days

    unique_past_days = past_slice["dt"].nunique()
    unique_future_days = future_slice["dt"].nunique()

    past_sales = past_slice.get("sales", pd.Series(dtype=float)).sum(min_count=1)
    future_sales = future_slice.get("sales", pd.Series(dtype=float)).sum(min_count=1)

    aggregates: dict[str, Any] = {
        "t_ref": t_ref,
        "past_sales": past_sales if pd.notna(past_sales) else 0.0,
        "future_sales": future_sales if pd.notna(future_sales) else 0.0,
        "past_coverage": unique_past_days / expected_past_days if expected_past_days else np.nan,
        "future_coverage": unique_future_days / expected_future_days if expected_future_days else np.nan,
    }

    for column in ("bsr_percentile", "bsr_rank"):
        past_column = f"past_{column}"
        future_column = f"future_{column}"
        if column in past_slice.columns:
            aggregates[past_column] = past_slice[column].mean()
        else:
            aggregates[past_column] = np.nan
        if column in future_slice.columns:
            aggregates[future_column] = future_slice[column].mean()
        else:
            aggregates[future_column] = np.nan

    if not past_slice.empty:
        aggregates["past_mean_sales"] = past_slice["sales"].mean()
        aggregates["past_std_sales"] = past_slice["sales"].std(ddof=0)
    else:
        aggregates["past_mean_sales"] = np.nan
        aggregates["past_std_sales"] = np.nan

    if not future_slice.empty:
        aggregates["future_mean_sales"] = future_slice["sales"].mean()
    else:
        aggregates["future_mean_sales"] = np.nan

    return aggregates


def _daily_rolling_sampler(
    mart_df: pd.DataFrame,
    windows: LabelWindows,
) -> pd.DataFrame:
    """Materialise ``(asin, site, t_ref)`` slices for the rolling sampler."""

    if mart_df.empty:
        return pd.DataFrame(columns=["asin", "site", "t_ref"])

    prepared = _prepare_frame(mart_df)
    required_cols = {"asin", "site", "dt"}
    missing = required_cols - set(prepared.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise KeyError(f"mart_df is missing required columns: {missing_columns}")

    samples: list[dict[str, Any]] = []
    for (asin, site), group in prepared.groupby(["asin", "site"], sort=False):
        if group.empty:
            continue

        group = group.sort_values("dt")
        dates = group["dt"].drop_duplicates().sort_values()
        min_ref = dates.min() + pd.Timedelta(days=max(windows.observation_days - 1, 0))
        max_ref = dates.max() - pd.Timedelta(days=windows.forecast_days)
        candidate_dates = dates[(dates >= min_ref) & (dates <= max_ref)]

        for t_ref in candidate_dates:
            aggregates = _compute_window_aggregates(group, t_ref, windows)
            aggregates.update({"asin": asin, "site": site})
            samples.append(aggregates)

    if not samples:
        return pd.DataFrame(columns=["asin", "site", "t_ref"])

    return pd.DataFrame(samples)


def _attach_features(
    samples: pd.DataFrame,
    features_df: pd.DataFrame | None,
    facts_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach feature vectors and static facts to the sampled slices."""

    if samples.empty:
        return samples.copy()

    enriched = samples.sort_values(["asin", "site", "t_ref"]).reset_index(drop=True)

    if features_df is not None and not features_df.empty:
        features = _prepare_frame(features_df)
        features = features.sort_values(["asin", "site", "dt"])
        feature_cols = [col for col in features.columns if col not in {"asin", "site", "dt"}]
        merged = pd.merge_asof(
            enriched,
            features,
            left_on="t_ref",
            right_on="dt",
            by=["asin", "site"],
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.drop(columns=["dt"] if "dt" in merged.columns else [])
        enriched = merged
        if feature_cols:
            # Create a serialisable feature payload for downstream tasks.
            enriched["feature_vector"] = (
                enriched[feature_cols]
                .apply(
                    lambda row: {
                        col: row[col]
                        for col in feature_cols
                        if col in row and pd.notna(row[col])
                    },
                    axis=1,
                )
                .apply(_normalise_for_json)
            )
        else:
            enriched["feature_vector"] = None
    else:
        enriched["feature_vector"] = None

    if facts_df is not None and not facts_df.empty:
        facts = _prepare_frame(facts_df)
        fact_cols = [col for col in facts.columns if col not in {"asin", "site", "dt"}]
        enriched = enriched.merge(facts, on=["asin", "site"], how="left", suffixes=("", "_fact"))
        if fact_cols:
            present_cols = [col for col in fact_cols if col in enriched.columns]
            if present_cols:
                enriched["metadata"] = (
                    enriched[present_cols]
                    .apply(
                        lambda row: {
                            col: row[col]
                            for col in present_cols
                            if pd.notna(row[col])
                        },
                        axis=1,
                    )
                    .apply(_normalise_for_json)
                )
            else:
                enriched["metadata"] = None
        else:
            enriched["metadata"] = None
    else:
        enriched["metadata"] = None

    return enriched


def _derive_targets(
    samples: pd.DataFrame,
    thresholds: Thresholds,
) -> pd.DataFrame:
    """Compute target columns for binary, ranking and regression tasks."""

    if samples.empty:
        for target in ("y_bin", "y_rank", "y_reg"):
            samples[target] = pd.Series(dtype="float64")
        return samples

    enriched = samples.copy()
    enriched["y_reg"] = enriched["future_sales"].astype(float)

    site_series = enriched.get("site")
    if site_series is None:
        site_labels = pd.Series("unknown_site", index=enriched.index, dtype="string")
    else:
        site_labels = site_series.fillna("unknown_site").astype("string")

    category_series = enriched.get("category_id")
    if category_series is None:
        category_labels = pd.Series("unknown_category", index=enriched.index, dtype="string")
    else:
        category_labels = category_series.fillna("unknown_category").astype("string")

    group_components: list[pd.Series] = [site_labels, category_labels]

    price_band_series = enriched.get("price_band")
    if price_band_series is not None and price_band_series.notna().any():
        price_labels = price_band_series.fillna("unknown_price_band").astype("string")
        group_components.append(price_labels)

    group_id = (
        pd.concat(group_components, axis=1)
        .astype("string")
        .agg("|".join, axis=1)
    )
    enriched["group_id"] = group_id

    enriched["y_rank"] = (
        enriched.groupby(["group_id", "t_ref"], sort=False)["y_reg"]
        .transform(lambda col: col.rank(method="average", ascending=False, pct=True))
    )

    past_sales = enriched.get("past_sales", pd.Series(dtype=float)).astype(float)
    future_sales = enriched.get("future_sales", pd.Series(dtype=float)).astype(float)

    past_sales_arr = past_sales.to_numpy()
    future_sales_arr = future_sales.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        sales_ratio_arr = np.divide(
            future_sales_arr,
            past_sales_arr,
            out=np.full_like(future_sales_arr, np.nan, dtype=float),
            where=past_sales_arr > 0,
        )
    zero_or_negative_past = past_sales_arr <= 0
    positive_future = future_sales_arr > 0
    sales_ratio_arr = np.where(
        zero_or_negative_past & positive_future,
        np.inf,
        sales_ratio_arr,
    )
    sales_ratio_arr = np.where(
        zero_or_negative_past & ~positive_future,
        np.nan,
        sales_ratio_arr,
    )
    enriched["sales_ratio"] = pd.Series(sales_ratio_arr, index=enriched.index)

    past_bsr_pct = enriched.get("past_bsr_percentile", pd.Series(np.nan, index=enriched.index)).astype(float)
    future_bsr_pct = enriched.get("future_bsr_percentile", pd.Series(np.nan, index=enriched.index)).astype(float)
    enriched["bsr_percentile_drop"] = past_bsr_pct - future_bsr_pct

    past_bsr_rank = enriched.get("past_bsr_rank", pd.Series(np.nan, index=enriched.index)).astype(float)
    future_bsr_rank = enriched.get("future_bsr_rank", pd.Series(np.nan, index=enriched.index)).astype(float)
    enriched["bsr_rank_improvement"] = past_bsr_rank - future_bsr_rank

    sales_ratio_values = enriched["sales_ratio"]
    ratio_condition = (sales_ratio_values >= thresholds.r) | sales_ratio_values.isna()

    percentile_values = enriched["bsr_percentile_drop"]
    percentile_condition = (percentile_values >= thresholds.p).fillna(False)

    rank_values = enriched["bsr_rank_improvement"]
    rank_condition = (rank_values > thresholds.delta).fillna(False)

    enriched["y_bin"] = (ratio_condition & (percentile_condition | rank_condition)).astype(int)

    return enriched


def _assign_splits(samples: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Attach deterministic time-based train/valid/test splits."""

    if samples.empty:
        samples["split"] = pd.Series(dtype="string")
        return samples

    enriched = samples.copy()
    unique_dates = np.sort(enriched["t_ref"].dropna().unique())
    if len(unique_dates) == 0:
        enriched["split"] = config.names[0]
        return enriched

    ratios = np.array(config.ratios, dtype=float)
    if ratios.sum() == 0:
        ratios = np.array([1.0, 0.0, 0.0])
    ratios = ratios / ratios.sum()

    counts = np.floor(ratios * len(unique_dates)).astype(int)
    remainder = len(unique_dates) - counts.sum()
    for idx in range(remainder):
        counts[idx % len(counts)] += 1

    boundaries: list[pd.Timestamp] = []
    cursor = 0
    for count in counts[:-1]:
        cursor += count
        if cursor == 0:
            boundaries.append(unique_dates[0])
        elif cursor >= len(unique_dates):
            boundaries.append(unique_dates[-1])
        else:
            boundaries.append(unique_dates[cursor - 1])

    if boundaries:
        train_cutoff = boundaries[0]
        valid_cutoff = boundaries[1] if len(boundaries) > 1 else unique_dates[-1]
    else:
        train_cutoff = unique_dates[-1]
        valid_cutoff = unique_dates[-1]

    conditions = [
        enriched["t_ref"] <= train_cutoff,
        enriched["t_ref"] <= valid_cutoff,
    ]
    choices = [config.names[0], config.names[1]]
    enriched["split"] = np.select(conditions, choices, default=config.names[2])
    return enriched


def _compute_reports(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute diagnostic reports for balance, coverage and missingness."""

    if samples.empty:
        return {
            "balance": pd.DataFrame(columns=["split", "y_bin", "count"]),
            "coverage": pd.DataFrame(columns=["split", "past_coverage", "future_coverage"]),
            "mean_encoders": pd.DataFrame(columns=["split", "site", "site_mean_y_reg"]),
            "global_stats": pd.DataFrame(columns=["split", "y_reg_mean", "y_reg_std", "n"]),
            "bsr_availability": pd.DataFrame(
                columns=[
                    "split",
                    "past_bsr_percentile",
                    "future_bsr_percentile",
                    "past_bsr_rank",
                    "future_bsr_rank",
                ]
            ),
            "missingness": pd.DataFrame(columns=["column", "missing_fraction"]),
        }

    reports: dict[str, pd.DataFrame] = {}

    balance = (
        samples.groupby(["split", "y_bin"], dropna=False)["asin"]
        .count()
        .rename("count")
        .reset_index()
    )
    reports["balance"] = balance

    coverage = (
        samples.groupby("split")[ ["past_coverage", "future_coverage"] ].mean().reset_index()
    )
    reports["coverage"] = coverage

    mean_targets = (
        samples.groupby(["split", "site"], dropna=False)["y_reg"]
        .mean()
        .rename("site_mean_y_reg")
        .reset_index()
    )
    reports["mean_encoders"] = mean_targets

    global_stats = (
        samples.groupby("split", dropna=False)["y_reg"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "y_reg_mean", "std": "y_reg_std", "count": "n"})
    )
    reports["global_stats"] = global_stats

    bsr_columns = [
        col
        for col in [
            "past_bsr_percentile",
            "future_bsr_percentile",
            "past_bsr_rank",
            "future_bsr_rank",
        ]
        if col in samples.columns
    ]
    if bsr_columns:
        bsr_availability = (
            samples.groupby("split", dropna=False)[bsr_columns]
            .agg(lambda series: series.notna().mean())
            .reset_index()
        )
        reports["bsr_availability"] = bsr_availability

    missingness = (
        samples.isna()
        .mean()
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    reports["missingness"] = missingness

    return reports


def _log_and_persist_reports(
    reports: Mapping[str, pd.DataFrame],
    logger: logging.Logger,
    report_callback: Callable[[str, pd.DataFrame], None] | None,
) -> None:
    """Send diagnostic reports to the configured sinks."""

    for name, frame in reports.items():
        logger.info("%s report:\n%s", name, frame.to_string(index=False))
        if report_callback is not None:
            report_callback(name, frame)


def _persist_split_tables(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create the split specific tables for downstream consumption."""

    tables: dict[str, pd.DataFrame] = {}
    for target, column in {
        "train_samples_bin": "y_bin",
        "train_samples_rank": "y_rank",
        "train_samples_reg": "y_reg",
    }.items():
        subset_columns = [
            col
            for col in ["asin", "site", "t_ref", "split", column, "feature_vector", "metadata"]
            if col in samples.columns
        ]
        if "split" not in samples.columns:
            tables[target] = pd.DataFrame(columns=subset_columns)
            continue
        tables[target] = (
            samples.loc[samples["split"] == "train", subset_columns]
            .reset_index(drop=True)
        )
    return tables


def build_labels(
    mart_df: pd.DataFrame,
    *,
    features_df: pd.DataFrame | None = None,
    facts_df: pd.DataFrame | None = None,
    windows: LabelWindows | None = None,
    thresholds: Thresholds | None = None,
    split_config: SplitConfig | None = None,
    logger: logging.Logger | None = None,
    report_callback: Callable[[str, pd.DataFrame], None] | None = None,
) -> BuildLabelsResult:
    """Build supervised targets and diagnostic artifacts for the training job."""

    logger = logger or logging.getLogger(__name__)
    logger.debug("Starting label construction")

    windows = windows or LabelWindows()
    thresholds = thresholds or Thresholds()
    split_config = split_config or SplitConfig()

    samples = _daily_rolling_sampler(mart_df, windows)

    if samples.empty:
        logger.warning("No valid samples were generated for the provided inputs")
        reports = _compute_reports(samples)
        _log_and_persist_reports(reports, logger, report_callback)
        tables = _persist_split_tables(samples)
        return BuildLabelsResult(samples=samples, **tables, reports=reports)

    samples = _attach_features(samples, features_df, facts_df)
    samples = _derive_targets(samples, thresholds)
    samples = _assign_splits(samples, split_config)

    if not samples.empty:
        samples["site_mean_y_reg"] = (
            samples.groupby(["split", "site"], dropna=False)["y_reg"].transform("mean")
        )
        samples["split_mean_y_reg"] = (
            samples.groupby("split", dropna=False)["y_reg"].transform("mean")
        )

    reports = _compute_reports(samples)
    _log_and_persist_reports(reports, logger, report_callback)

    tables = _persist_split_tables(samples)
    logger.debug("Generated %d samples", len(samples))

    return BuildLabelsResult(samples=samples, reports=reports, **tables)


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
