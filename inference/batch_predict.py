"""Batch inference script for LightGBM ranking model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb
except ModuleNotFoundError:  # pragma: no cover
    lgb = None  # type: ignore

from utils.logging import get_logger

logger = get_logger(__name__)


def load_features(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    if file_path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def load_model(path: str | Path) -> "lgb.Booster":
    if lgb is None:
        raise ModuleNotFoundError("lightgbm is required for batch prediction")
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return lgb.Booster(model_file=str(file_path))


def predict(
    model: "lgb.Booster",
    features: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
) -> pd.DataFrame:
    if features.empty:
        logger.warning("Feature dataframe is empty; nothing to predict")
        return features
    if not feature_cols:
        raise ValueError("No feature columns provided for prediction")

    scores = model.predict(features[feature_cols], num_iteration=model.best_iteration)
    features = features.copy()
    features["lgbm_score"] = scores
    features["rank_in_group"] = (
        features.groupby(group_col)["lgbm_score"].rank(ascending=False, method="first")
    )
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch LightGBM inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-col", default="group_id")
    parser.add_argument("--feature-cols", nargs="*")
    args = parser.parse_args()

    model = load_model(args.model)
    df = load_features(args.features)
    feature_cols = args.feature_cols or [
        col
        for col in df.columns
        if col not in {"asin", "site", "dt", args.group_col, "group_id", "lgbm_score", "rank_in_group"}
    ]
    result = predict(model, df, feature_cols, args.group_col)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix in {".parquet", ".pq"}:
        result.to_parquet(output_path, index=False)
    else:
        result.to_csv(output_path, index=False)
    logger.info("Wrote predictions to %s", output_path)


if __name__ == "__main__":
    main()
