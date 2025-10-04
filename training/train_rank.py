"""Train a LambdaRank model using LightGBM."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from training.eval_metrics import compute_ranking_metrics
from training.run_logger import RunLogger


LOGGER = logging.getLogger(__name__)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def _load_rank_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"No training data found at {data_path}")
    if data_path.is_file():
        return _read_frame(data_path)

    frames = []
    for file in sorted(data_path.glob("*")):
        if file.suffix.lower() not in {".csv", ".parquet", ".txt"}:
            continue
        frames.append(_read_frame(file))
    if not frames:
        raise FileNotFoundError(f"No compatible files found in {data_path}")
    return pd.concat(frames, ignore_index=True)


def _detect_group_column(df: pd.DataFrame, provided: str | None) -> str:
    if provided and provided in df.columns:
        return provided
    for candidate in ("group_id", "query", "query_id", "asin", "item_group"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Unable to determine group column; please specify explicitly")


def _detect_label_column(df: pd.DataFrame, provided: str | None) -> str:
    if provided and provided in df.columns:
        return provided
    for candidate in ("y_rank", "label", "target", "relevance"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Unable to determine label column; please specify explicitly")


def _prepare_features(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    drop_columns: Sequence[str],
) -> pd.DataFrame:
    excluded = {label_col, group_col, *drop_columns}
    features = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not features:
        raise ValueError("No numeric features found for training")
    return df[features]


def _group_sizes(groups: Iterable) -> list[int]:
    sizes: list[int] = []
    current = None
    count = 0
    for item in groups:
        if current is None:
            current = item
            count = 1
            continue
        if item == current:
            count += 1
        else:
            sizes.append(count)
            current = item
            count = 1
    if current is not None:
        sizes.append(count)
    return sizes


def _train_validation_split(
    df: pd.DataFrame,
    group_col: str,
    validation_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    groups = df[group_col].drop_duplicates().to_numpy()
    permutation = rng.permutation(len(groups))
    n_val = max(1, int(np.ceil(len(groups) * validation_fraction)))
    val_groups = set(groups[permutation[:n_val]])
    train_df = df[~df[group_col].isin(val_groups)].copy()
    val_df = df[df[group_col].isin(val_groups)].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Validation split produced empty dataset; adjust fraction")
    return train_df, val_df


def _build_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    weights: pd.Series | None = None,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("LightGBM is required for ranking training") from exc

    group_sizes = _group_sizes(groups.to_numpy())
    dataset = lgb.Dataset(features, label=labels, group=group_sizes, weight=weights)
    return dataset


def train_model(args: argparse.Namespace) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data_path = Path(args.data_path)
    df = _load_rank_data(data_path)
    group_col = _detect_group_column(df, args.group_column)
    label_col = _detect_label_column(df, args.label_column)
    weight_col = args.weight_column if args.weight_column and args.weight_column in df.columns else None

    train_df, val_df = _train_validation_split(df, group_col, args.validation_fraction, args.random_state)

    drop_columns = set(args.drop_columns or [])
    train_df = train_df.sort_values(group_col).reset_index(drop=True)
    val_df = val_df.sort_values(group_col).reset_index(drop=True)

    train_features = _prepare_features(train_df, label_col, group_col, drop_columns)
    val_features = _prepare_features(val_df, label_col, group_col, drop_columns)

    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("LightGBM must be installed to run ranking training") from exc

    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [5, 10, 20],
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

    train_dataset = _build_dataset(
        train_features,
        train_df[label_col],
        train_df[group_col],
        train_df[weight_col] if weight_col else None,
    )
    val_dataset = _build_dataset(
        val_features,
        val_df[label_col],
        val_df[group_col],
        val_df[weight_col] if weight_col else None,
    )

    evals_result: dict[str, dict[str, list[float]]] = {}

    with RunLogger(args.experiment_name, args.run_name) as run:
        run.log_params(default_params)
        run.set_tags({"data_path": str(data_path), "group_column": group_col, "label_column": label_col})

        booster = lgb.train(
            default_params,
            train_set=train_dataset,
            valid_sets=[train_dataset, val_dataset],
            valid_names=["train", "validation"],
            evals_result=evals_result,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=args.verbose_eval,
        )

        best_iteration = booster.best_iteration or args.num_boost_round
        LOGGER.info("Best iteration: %s", best_iteration)

        val_predictions = booster.predict(val_features, num_iteration=best_iteration)
        ranking_metrics = compute_ranking_metrics(val_df[label_col], val_predictions, val_df[group_col], ks=(5, 10, 20, 50))
        run.log_metrics(ranking_metrics)

        feature_importances = pd.DataFrame(
            {
                "feature": train_features.columns,
                "gain": booster.feature_importance(importance_type="gain"),
                "split": booster.feature_importance(importance_type="split"),
            }
        ).sort_values("gain", ascending=False)
        feature_path = Path(args.output_dir) / "feature_importances.csv"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_importances.to_csv(feature_path, index=False)
        run.log_artifact(feature_path)

        shap_sample = min(len(val_features), args.shap_sample_size)
        if shap_sample > 0:
            sample = val_features.sample(n=shap_sample, random_state=args.random_state)
            shap_values = booster.predict(sample, num_iteration=best_iteration, pred_contrib=True)
            shap_df = pd.DataFrame(shap_values, columns=list(sample.columns) + ["bias"])
            shap_path = Path(args.output_dir) / "shap_values.csv"
            shap_df.to_csv(shap_path, index=False)
            run.log_artifact(shap_path)

        model_path = Path(args.output_dir) / "model.txt"
        booster.save_model(model_path)
        run.log_artifact(model_path)

        metrics_path = Path(args.output_dir) / "metrics.json"
        metrics_path.write_text(json.dumps(ranking_metrics, indent=2), encoding="utf-8")
        run.log_artifact(metrics_path)

    return ranking_metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaRank model")
    parser.add_argument("--data-path", type=str, default="train_samples_rank", help="Path to training samples directory or file")
    parser.add_argument("--output-dir", type=str, default="artifacts/rank", help="Directory to store model artifacts")
    parser.add_argument("--experiment-name", type=str, default="rank-training", help="Experiment name for MLflow or local logging")
    parser.add_argument("--run-name", type=str, default="run", help="Run name for logging")
    parser.add_argument("--group-column", type=str, default=None, help="Column containing group identifiers")
    parser.add_argument("--label-column", type=str, default=None, help="Column containing rank labels")
    parser.add_argument("--weight-column", type=str, default=None, help="Optional sample weight column")
    parser.add_argument("--drop-columns", nargs="*", default=None, help="Additional columns to drop from features")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Fraction of groups used for validation")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for deterministic behaviour")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--verbose-eval", type=int, default=50)
    parser.add_argument("--shap-sample-size", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = train_model(args)
    LOGGER.info("Validation metrics: %s", metrics)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

