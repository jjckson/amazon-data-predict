"""Train a LightGBM ranker using prepared training samples."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb
except ModuleNotFoundError:  # pragma: no cover
    lgb = None  # type: ignore

from utils.logging import get_logger

logger = get_logger(__name__)


def load_data(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    if file_path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def build_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.Series, List[int]]:
    missing = set(feature_cols + [label_col, group_col]) - set(df.columns)
    if missing:
        raise KeyError(f"Columns missing from dataset: {sorted(missing)}")
    grouped = df.groupby(group_col, sort=False)
    group_sizes = grouped.size().tolist()
    X = df[feature_cols]
    y = df[label_col]
    return X, y, group_sizes


def fit_lgb_ranker(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    group_col: str,
    params: dict,
) -> "lgb.Booster":
    if lgb is None:
        raise ModuleNotFoundError(
            "lightgbm is not installed. Please add it to requirements before training."
        )

    X_tr, y_tr, g_tr = build_matrix(train_df, feature_cols, label_col, group_col)
    X_va, y_va, g_va = build_matrix(valid_df, feature_cols, label_col, group_col)

    dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, group=g_va, reference=dtrain, free_raw_data=False)

    logger.info("Starting LightGBM training with %d features", len(feature_cols))
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        num_boost_round=params.get("num_boost_round", 2000),
        early_stopping_rounds=params.get("early_stopping_rounds", 100),
        verbose_eval=params.get("verbose_eval", 100),
    )
    return model


def default_params() -> dict:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10, 20],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "num_boost_round": 2000,
        "early_stopping_rounds": 100,
        "verbose_eval": 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM ranking model")
    parser.add_argument("--data", required=True, help="Path to train_samples_rank dataset")
    parser.add_argument("--features", nargs="*", help="Feature columns to use")
    parser.add_argument("--label", default="y", help="Label column name")
    parser.add_argument("--group", default="group_id", help="Grouping column name")
    parser.add_argument("--output", default="artifacts/lgb_rank.txt")
    args = parser.parse_args()

    df = load_data(args.data)
    if df.empty:
        raise ValueError("Training dataset is empty")

    feature_cols = args.features or [
        col
        for col in df.columns
        if col not in {"asin", "site", "t_ref", args.label, args.group, "split", "meta"}
        and not col.startswith("feature_vector")
    ]
    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]
    if valid_df.empty:
        raise ValueError("Validation split is empty; adjust split ratios in label builder.")

    params = default_params()
    model = fit_lgb_ranker(train_df, valid_df, feature_cols, args.label, args.group, params)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    logger.info("Saved LightGBM ranker to %s", output_path)


if __name__ == "__main__":
    main()
