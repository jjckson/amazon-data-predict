"""Regression baselines using gradient boosting libraries."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from training.eval_metrics import regression_metrics
from training.run_logger import RunLogger


LOGGER = logging.getLogger(__name__)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.is_file():
        return _read_frame(path)
    frames = []
    for file in sorted(path.glob("*")):
        if file.suffix.lower() not in {".csv", ".parquet", ".txt"}:
            continue
        frames.append(_read_frame(file))
    if not frames:
        raise FileNotFoundError(f"No compatible files located in {path}")
    return pd.concat(frames, ignore_index=True)


def _prepare_features(df: pd.DataFrame, target_col: str, drop_columns: Sequence[str]) -> pd.DataFrame:
    excluded = {target_col, *drop_columns}
    features = [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]
    if not features:
        raise ValueError("No numeric features available for regression training")
    return df[features]


def _train_validation_split(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    n_test = max(1, int(np.ceil(len(df) * test_size)))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Validation split produced empty partition; adjust fraction")
    return train_df, test_df


def _train_lightgbm(train_X, train_y, val_X, val_y, args):
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LightGBM must be installed for lightgbm model") from exc

    params = {
        "objective": "regression",
        "metric": ["rmse"],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 1,
        "lambda_l2": args.lambda_l2,
        "random_state": args.random_state,
        "verbose": -1,
    }

    train_dataset = lgb.Dataset(train_X, label=train_y)
    val_dataset = lgb.Dataset(val_X, label=val_y, reference=train_dataset)

    booster = lgb.train(
        params,
        train_set=train_dataset,
        valid_sets=[train_dataset, val_dataset],
        valid_names=["train", "validation"],
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
    )

    best_iteration = booster.best_iteration or args.num_boost_round
    train_pred = booster.predict(train_X, num_iteration=best_iteration)
    val_pred = booster.predict(val_X, num_iteration=best_iteration)
    feature_importances = pd.DataFrame(
        {
            "feature": train_X.columns,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)

    return booster, train_pred, val_pred, feature_importances


def _train_xgboost(train_X, train_y, val_X, val_y, args):
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("XGBoost must be installed for xgboost model") from exc

    dtrain = xgb.DMatrix(train_X, label=train_y)
    dval = xgb.DMatrix(val_X, label=val_y)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.bagging_fraction,
        "colsample_bytree": args.feature_fraction,
        "lambda": args.lambda_l2,
        "min_child_weight": args.min_data_in_leaf,
        "seed": args.random_state,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
    )

    best_ntree = booster.best_ntree_limit or args.num_boost_round
    train_pred = booster.predict(dtrain, ntree_limit=best_ntree)
    val_pred = booster.predict(dval, ntree_limit=best_ntree)
    importance = booster.get_score(importance_type="gain")
    feature_importances = pd.DataFrame(
        {
            "feature": list(importance.keys()),
            "gain": list(importance.values()),
        }
    ).sort_values("gain", ascending=False)

    return booster, train_pred, val_pred, feature_importances


def _train_catboost(train_X, train_y, val_X, val_y, args):
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CatBoost must be installed for catboost model") from exc

    model = CatBoostRegressor(
        iterations=args.num_boost_round,
        learning_rate=args.learning_rate,
        depth=args.max_depth,
        l2_leaf_reg=args.lambda_l2,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=args.random_state,
        verbose=False,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    train_pool = Pool(train_X, label=train_y)
    val_pool = Pool(val_X, label=val_y)
    model.fit(train_pool, eval_set=val_pool, verbose=args.verbose_eval)

    train_pred = model.predict(train_X)
    val_pred = model.predict(val_X)
    feature_importances = pd.DataFrame(
        {
            "feature": train_X.columns,
            "importance": model.get_feature_importance(train_pool),
        }
    ).sort_values("importance", ascending=False)

    return model, train_pred, val_pred, feature_importances


MODEL_TRAINERS = {
    "lightgbm": _train_lightgbm,
    "xgboost": _train_xgboost,
    "catboost": _train_catboost,
}


def train_model(args: argparse.Namespace) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data_path = Path(args.data_path)
    df = _load_dataset(data_path)
    if args.target_column not in df.columns:
        raise KeyError(f"Target column {args.target_column!r} not present in dataset")

    drop_columns = set(args.drop_columns or [])
    train_df, val_df = _train_validation_split(df, args.validation_fraction, args.random_state)
    train_features = _prepare_features(train_df, args.target_column, drop_columns)
    val_features = _prepare_features(val_df, args.target_column, drop_columns)

    trainer = MODEL_TRAINERS.get(args.model_type)
    if trainer is None:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model, train_pred, val_pred, feature_importances = trainer(
        train_features,
        train_df[args.target_column],
        val_features,
        val_df[args.target_column],
        args,
    )

    train_metrics = regression_metrics(train_df[args.target_column], train_pred)
    val_metrics = regression_metrics(val_df[args.target_column], val_pred)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"train": train_metrics, "validation": val_metrics}

    with RunLogger(args.experiment_name, args.run_name) as run:
        run.log_params(
            {
                "model_type": args.model_type,
                "learning_rate": args.learning_rate,
                "num_boost_round": args.num_boost_round,
                "num_leaves": args.num_leaves,
                "max_depth": args.max_depth,
                "min_data_in_leaf": args.min_data_in_leaf,
                "feature_fraction": args.feature_fraction,
                "bagging_fraction": args.bagging_fraction,
                "lambda_l2": args.lambda_l2,
            }
        )
        run.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        run.log_metrics({f"validation_{k}": v for k, v in val_metrics.items()})

        feature_path = output_dir / f"{args.model_type}_feature_importances.csv"
        feature_importances.to_csv(feature_path, index=False)
        run.log_artifact(feature_path)

        preds_path = output_dir / f"{args.model_type}_validation_predictions.csv"
        pd.DataFrame(
            {
                "prediction": val_pred,
                "label": val_df[args.target_column].to_numpy(),
            }
        ).to_csv(preds_path, index=False)
        run.log_artifact(preds_path)

        metrics_path = output_dir / f"{args.model_type}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        run.log_artifact(metrics_path)

    model_path = output_dir / f"{args.model_type}_model"
    try:
        if hasattr(model, "save_model"):
            model.save_model(str(model_path))  # type: ignore[attr-defined]
        elif hasattr(model, "save_model_to_file"):
            model.save_model_to_file(str(model_path))
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to persist model artifact: %s", exc)

    return val_metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train regression baseline")
    parser.add_argument("--data-path", type=str, default="train_samples_reg")
    parser.add_argument("--output-dir", type=str, default="artifacts/reg")
    parser.add_argument("--experiment-name", type=str, default="regression-baseline")
    parser.add_argument("--run-name", type=str, default="run")
    parser.add_argument("--target-column", type=str, default="target")
    parser.add_argument("--drop-columns", nargs="*", default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-type", type=str, default="lightgbm", choices=list(MODEL_TRAINERS.keys()))
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--verbose-eval", type=int, default=50)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    val_metrics = train_model(args)
    LOGGER.info("Validation metrics: %s", val_metrics)


if __name__ == "__main__":  # pragma: no cover
    main()

