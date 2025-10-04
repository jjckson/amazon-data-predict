"""Hyper-parameter search for the LightGBM ranker using Optuna."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb
except ModuleNotFoundError:  # pragma: no cover
    lgb = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import optuna
except ModuleNotFoundError:  # pragma: no cover
    optuna = None  # type: ignore

from training.train_rank import build_matrix, load_data
from utils.logging import get_logger

logger = get_logger(__name__)


def objective_factory(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    group_col: str,
):
    if lgb is None:
        raise ModuleNotFoundError("lightgbm is required for tuning")

    X_tr, y_tr, g_tr = build_matrix(train_df, feature_cols, label_col, group_col)
    X_va, y_va, g_va = build_matrix(valid_df, feature_cols, label_col, group_col)
    dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, group=g_va, reference=dtrain, free_raw_data=False)

    def _objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [10, 20],
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 5),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "verbosity": -1,
        }
        booster = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            valid_names=["valid"],
            num_boost_round=2000,
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        best_score = booster.best_score.get("valid", {}).get("ndcg@20")
        if best_score is None:
            raise RuntimeError("LightGBM did not report ndcg@20")
        return best_score

    return _objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune LightGBM ranker with Optuna")
    parser.add_argument("--data", required=True)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--label", default="y")
    parser.add_argument("--group", default="group_id")
    parser.add_argument("--features", nargs="*")
    parser.add_argument("--study-name", default="lgb_rank_tuning")
    parser.add_argument("--storage")
    args = parser.parse_args()

    if optuna is None:
        raise ModuleNotFoundError("optuna is not installed. Add it to requirements to run tuning.")

    df = load_data(args.data)
    feature_cols = args.features or [
        col
        for col in df.columns
        if col not in {"asin", "site", "t_ref", args.label, args.group, "split", "meta"}
        and not col.startswith("feature_vector")
    ]
    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]
    if valid_df.empty:
        raise ValueError("Validation split is empty; adjust label configuration")

    objective = objective_factory(train_df, valid_df, feature_cols, args.label, args.group)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials)

    logger.info("Best trial: score=%.4f params=%s", study.best_value, study.best_params)


if __name__ == "__main__":
    main()
