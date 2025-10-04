"""Generate supervised training samples with labels for downstream models."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils.config import load_settings
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LabelConfig:
    lookback_days: int = 28
    horizon_days: int = 30
    ratio_threshold: float = 1.6
    percentile_drop: float = 0.2
    bsr_delta: Optional[float] = None
    min_coverage: float = 0.7
    min_past_sales: float = 1e-6
    split_ratios: Sequence[float] = (0.7, 0.15, 0.15)
    price_band_quantiles: Sequence[float] = (0.0, 0.25, 0.75, 1.0)


class LabelBuilder:
    """Create binary, ranking, and regression labels from historical data."""

    def __init__(self, settings: Optional[Dict] = None) -> None:
        cfg = settings or load_settings()
        training_cfg = cfg.get("training", {}).get("labels", {})
        splits_cfg = cfg.get("training", {}).get("splits", {})
        self.config = LabelConfig(
            lookback_days=training_cfg.get("lookback_days", 28),
            horizon_days=training_cfg.get("horizon_days", 30),
            ratio_threshold=training_cfg.get("ratio_threshold", 1.6),
            percentile_drop=training_cfg.get("percentile_drop", 0.2),
            bsr_delta=training_cfg.get("bsr_delta"),
            min_coverage=training_cfg.get("min_coverage", 0.7),
            min_past_sales=training_cfg.get("min_past_sales", 1e-6),
            split_ratios=(
                splits_cfg.get("train", 0.7),
                splits_cfg.get("valid", 0.15),
                splits_cfg.get("test", 0.15),
            ),
            price_band_quantiles=tuple(
                training_cfg.get(
                    "price_band_quantiles", (0.0, 0.25, 0.75, 1.0)
                )
            ),
        )

    def build(
        self,
        mart_df: pd.DataFrame,
        features_df: pd.DataFrame,
        orders_df: Optional[pd.DataFrame] = None,
        ads_df: Optional[pd.DataFrame] = None,
        dim_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Return label tables keyed by task name."""
        if mart_df.empty:
            logger.warning("Mart dataframe is empty; skipping label generation")
            return {"bin": pd.DataFrame(), "rank": pd.DataFrame(), "reg": pd.DataFrame()}

        mart_df = mart_df.copy()
        mart_df["dt"] = pd.to_datetime(mart_df["dt"])
        mart_df.sort_values(["asin", "site", "dt"], inplace=True)

        category_col = self._infer_category_column(mart_df, dim_df)
        if dim_df is not None and category_col not in mart_df.columns:
            mart_df = mart_df.merge(
                dim_df[["asin", "site", category_col]],
                on=["asin", "site"],
                how="left",
            )

        mart_df["category_key"] = (
            mart_df[category_col].apply(self._normalise_category)
            if category_col in mart_df.columns
            else "unknown"
        )

        mart_df = self._attach_sales(mart_df, orders_df)
        mart_df = self._attach_ads(mart_df, ads_df)
        mart_df = self._compute_bsr_percentile(mart_df)

        features_df = features_df.copy()
        features_df["dt"] = pd.to_datetime(features_df["dt"])

        merged = mart_df.merge(
            features_df,
            on=["asin", "site", "dt"],
            how="left",
            suffixes=("", "_feat"),
        )

        label_frame = self._compute_windows(merged)
        if label_frame.empty:
            return {"bin": label_frame, "rank": label_frame, "reg": label_frame}

        label_frame = self._assign_splits(label_frame)
        label_frame = self._attach_meta(label_frame)
        label_frame = self._attach_feature_vectors(label_frame, features_df)

        outputs = {
            "bin": label_frame[
                ["asin", "site", "t_ref", "y_bin", "split", "feature_vector", "meta"]
            ].rename(columns={"y_bin": "y"}),
            "rank": label_frame[
                [
                    "asin",
                    "site",
                    "t_ref",
                    "y_rank",
                    "group_id",
                    "split",
                    "feature_vector",
                    "meta",
                ]
            ].rename(columns={"y_rank": "y"}),
            "reg": label_frame[
                ["asin", "site", "t_ref", "y_reg", "split", "feature_vector", "meta"]
            ].rename(columns={"y_reg": "y"}),
        }

        return outputs

    def _attach_sales(
        self, mart_df: pd.DataFrame, orders_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        if orders_df is not None and not orders_df.empty:
            orders = orders_df.copy()
            if "order_date" in orders.columns and "dt" not in orders.columns:
                orders["dt"] = pd.to_datetime(orders["order_date"])
            else:
                orders["dt"] = pd.to_datetime(orders["dt"])
            agg = (
                orders.groupby(["asin", "site", "dt"], as_index=False)["ordered_units"]
                .sum()
                .rename(columns={"ordered_units": "sales"})
            )
            mart_df = mart_df.merge(agg, on=["asin", "site", "dt"], how="left")
        if "sales" not in mart_df.columns:
            if "est_sales" in mart_df.columns:
                mart_df.rename(columns={"est_sales": "sales"}, inplace=True)
            else:
                mart_df["sales"] = np.nan
        return mart_df

    def _attach_ads(
        self, mart_df: pd.DataFrame, ads_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        if ads_df is None or ads_df.empty:
            return mart_df
        ads_df = ads_df.copy()
        if "dt" in ads_df.columns:
            ads_df["dt"] = pd.to_datetime(ads_df["dt"])
        elif "date" in ads_df.columns:
            ads_df["dt"] = pd.to_datetime(ads_df["date"])
        else:
            return mart_df
        grouped = (
            ads_df.groupby(["asin", "site", "dt"], as_index=False)["spend"].sum()
            if "spend" in ads_df.columns
            else pd.DataFrame()
        )
        if not grouped.empty:
            mart_df = mart_df.merge(grouped, on=["asin", "site", "dt"], how="left")
        return mart_df

    def _compute_bsr_percentile(self, mart_df: pd.DataFrame) -> pd.DataFrame:
        if "bsr" not in mart_df.columns:
            mart_df["bsr_percentile"] = np.nan
            return mart_df
        mart_df["bsr_percentile"] = (
            mart_df.groupby(["site", "dt"])["bsr"].rank(pct=True, method="max")
        )
        return mart_df

    def _compute_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        lookback = cfg.lookback_days
        horizon = cfg.horizon_days
        min_past = math.ceil(lookback * cfg.min_coverage)
        min_future = math.ceil(horizon * cfg.min_coverage)

        results: List[pd.DataFrame] = []
        for (asin, site), group in df.groupby(["asin", "site"], sort=False):
            group = group.sort_values("dt")
            group["sales"] = group["sales"].astype(float)
            past_sales = (
                group["sales"].rolling(window=lookback, min_periods=min_past).sum()
            )
            future_sales = (
                group["sales"].iloc[::-1]
                .rolling(window=horizon, min_periods=min_future)
                .sum()
                .iloc[::-1]
                .shift(-1)
            )

            past_bsr_pct = (
                group["bsr_percentile"]
                .rolling(window=lookback, min_periods=min_past)
                .median()
            )
            future_bsr_pct = (
                group["bsr_percentile"].iloc[::-1]
                .rolling(window=horizon, min_periods=min_future)
                .median()
                .iloc[::-1]
                .shift(-1)
            )

            past_bsr = (
                group["bsr"].rolling(window=lookback, min_periods=min_past).median()
                if "bsr" in group.columns
                else np.nan
            )
            future_bsr = (
                group["bsr"].iloc[::-1]
                .rolling(window=horizon, min_periods=min_future)
                .median()
                .iloc[::-1]
                .shift(-1)
                if "bsr" in group.columns
                else np.nan
            )

            slice_df = group[["asin", "site", "dt", "category_key"]].copy()
            if "price" in group.columns:
                slice_df["price"] = group["price"].astype(float).to_numpy()
            slice_df["past_sales"] = past_sales
            slice_df["future_sales"] = future_sales
            slice_df["past_sales_points"] = (
                group["sales"].rolling(window=lookback, min_periods=1).count()
            )
            slice_df["future_sales_points"] = (
                group["sales"].iloc[::-1]
                .rolling(window=horizon, min_periods=1)
                .count()
                .iloc[::-1]
                .shift(-1)
            )
            slice_df["past_bsr_pct"] = past_bsr_pct
            slice_df["future_bsr_pct"] = future_bsr_pct
            slice_df["past_bsr"] = past_bsr
            slice_df["future_bsr"] = future_bsr
            slice_df["t_ref"] = slice_df["dt"]
            results.append(slice_df)

        if not results:
            return pd.DataFrame()
        frame = pd.concat(results, ignore_index=True)
        frame.dropna(subset=["future_sales"], inplace=True)

        cfg = self.config
        frame["sales_ratio"] = frame["future_sales"] / frame["past_sales"].replace(0, np.nan)
        frame.loc[frame["past_sales"] < cfg.min_past_sales, "sales_ratio"] = np.nan
        frame["bsr_percentile_drop"] = frame["past_bsr_pct"] - frame["future_bsr_pct"]
        frame["bsr_delta"] = frame["future_bsr"] - frame["past_bsr"]

        frame["y_bin"] = self._compute_binary_label(frame)
        frame["y_rank"] = self._compute_rank_target(frame)
        frame["y_reg"] = frame["future_sales"]
        frame["group_id"] = frame.apply(self._group_id, axis=1)
        return frame.dropna(subset=["y_bin", "y_rank", "y_reg"], how="all")

    def _compute_binary_label(self, frame: pd.DataFrame) -> pd.Series:
        cfg = self.config
        ratio_condition = frame["sales_ratio"] >= cfg.ratio_threshold
        drop_condition = frame["bsr_percentile_drop"] >= cfg.percentile_drop
        if cfg.bsr_delta is not None:
            delta_condition = frame["bsr_delta"] <= -abs(cfg.bsr_delta)
        else:
            delta_condition = pd.Series(False, index=frame.index)
        label = ratio_condition & (drop_condition | delta_condition)
        return label.astype(int)

    def _compute_rank_target(self, frame: pd.DataFrame) -> pd.Series:
        grouping_cols = ["site", "category_key", "t_ref"]
        ranks = (
            frame.groupby(grouping_cols)["future_sales"].rank(pct=True, ascending=True)
        )
        return 1 - ranks

    def _group_id(self, row: pd.Series) -> int:
        key = f"{row['site']}|{row['category_key']}"
        return abs(hash(key)) % (10**12)

    def _assign_splits(self, frame: pd.DataFrame) -> pd.DataFrame:
        ratios = self.config.split_ratios
        if not math.isclose(sum(ratios), 1.0, rel_tol=1e-3):
            raise ValueError("Split ratios must sum to 1.0")
        unique_dates = sorted(frame["t_ref"].dropna().unique())
        n_dates = len(unique_dates)
        if n_dates == 0:
            frame["split"] = "train"
            return frame
        train_cut = int(n_dates * ratios[0])
        valid_cut = int(n_dates * (ratios[0] + ratios[1]))
        train_dates = set(unique_dates[:train_cut or 1])
        valid_dates = set(unique_dates[train_cut:valid_cut or train_cut + 1])
        test_dates = set(unique_dates[valid_cut:])

        def label_split(ts: pd.Timestamp) -> str:
            if ts in train_dates:
                return "train"
            if ts in valid_dates:
                return "valid"
            return "test"

        frame["split"] = frame["t_ref"].apply(label_split)
        return frame

    def _attach_meta(self, frame: pd.DataFrame) -> pd.DataFrame:
        price_bands = self._price_bands(frame)
        frame = frame.merge(
            price_bands, on=["asin", "site"], how="left", suffixes=("", "_band")
        )
        frame["meta"] = frame.apply(
            lambda row: {
                "category": row.get("category_key", "unknown"),
                "price_band": row.get("price_band", "unknown"),
            },
            axis=1,
        )
        return frame

    def _price_bands(self, frame: pd.DataFrame) -> pd.DataFrame:
        price_col = "price"
        if price_col not in frame.columns:
            return frame[["asin", "site"]].drop_duplicates().assign(price_band="unknown")
        quantiles = self.config.price_band_quantiles
        summary = (
            frame.dropna(subset=[price_col])
            .groupby(["asin", "site"], as_index=False)[price_col]
            .median()
            .rename(columns={price_col: "median_price"})
        )
        if summary.empty:
            return frame[["asin", "site"]].drop_duplicates().assign(price_band="unknown")

        results = []
        for site, site_group in summary.groupby("site", sort=False):
            prices = site_group["median_price"]
            try:
                labels = pd.qcut(
                    prices,
                    q=quantiles,
                    duplicates="drop",
                    labels=False,
                )
                label_map = {
                    idx: f"q{int(quantiles[i]*100)}-{int(quantiles[i+1]*100)}"
                    for idx, i in enumerate(range(len(quantiles) - 1))
                }
                price_band = labels.map(label_map)
            except ValueError:
                price_band = pd.Series("unknown", index=site_group.index)
            result = site_group[["asin", "site"]].copy()
            result["price_band"] = price_band.fillna("unknown")
            results.append(result)

        if not results:
            return frame[["asin", "site"]].drop_duplicates().assign(price_band="unknown")
        return pd.concat(results, ignore_index=True)

    def _attach_feature_vectors(
        self, frame: pd.DataFrame, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        feature_cols = [
            col
            for col in features_df.columns
            if col not in {"asin", "site", "dt"}
        ]
        features_df = features_df[["asin", "site", "dt"] + feature_cols]
        features_df = features_df.rename(columns={"dt": "t_ref"})
        merged = frame.merge(features_df, on=["asin", "site", "t_ref"], how="left")
        frame_cols = frame.columns.tolist()
        frame = merged[frame_cols + feature_cols]
        feature_vectors = (
            merged[feature_cols]
            .apply(lambda row: {k: row[k] for k in feature_cols if pd.notna(row[k])}, axis=1)
            .apply(lambda d: d if d else {})
        )
        frame["feature_vector"] = feature_vectors.apply(json.dumps)
        return frame

    def _infer_category_column(
        self, mart_df: pd.DataFrame, dim_df: Optional[pd.DataFrame]
    ) -> str:
        candidates = [
            "category_id",
            "category",
            "category_path",
        ]
        for col in candidates:
            if col in mart_df.columns:
                return col
        if dim_df is not None:
            for col in candidates:
                if col in dim_df.columns:
                    return col
        return "category"

    def _normalise_category(self, value: object) -> str:
        if isinstance(value, (list, tuple)):
            return ">".join(str(v) for v in value)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "unknown"
        return str(value)


def load_frame(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def write_outputs(outputs: Dict[str, pd.DataFrame], output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        if df.empty:
            continue
        out_path = path / f"train_samples_{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Wrote %s samples to %s", name, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training labels")
    parser.add_argument("--mart", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--orders")
    parser.add_argument("--ads")
    parser.add_argument("--dim")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    mart_df = load_frame(args.mart)
    features_df = load_frame(args.features)
    orders_df = load_frame(args.orders) if args.orders else None
    ads_df = load_frame(args.ads) if args.ads else None
    dim_df = load_frame(args.dim) if args.dim else None

    builder = LabelBuilder()
    outputs = builder.build(mart_df, features_df, orders_df, ads_df, dim_df)
    write_outputs(outputs, args.output_dir)


if __name__ == "__main__":
    main()
