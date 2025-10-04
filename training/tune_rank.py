"""Hyper-parameter tuning for the LambdaRank model using Optuna."""
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
from training.train_rank import _detect_group_column, _detect_label_column, _load_rank_data, _prepare_features


LOGGER = logging.getLogger(__name__)


def _group_kfold_split(groups: Sequence, n_splits: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.array(list(dict.fromkeys(groups)))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if unique_groups.size < n_splits:
        raise ValueError("Not enough groups to perform requested splits")
    fold_sizes = np.full(n_splits, unique_groups.size // n_splits, dtype=int)
    fold_sizes[: unique_groups.size % n_splits] += 1
    current = 0
    group_to_indices: dict[object, np.ndarray] = {}
    groups_np = np.asarray(groups)
    for group in unique_groups:
        group_to_indices[group] = np.where(groups_np == group)[0]

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_groups = unique_groups[start:stop]
        val_indices = np.concatenate([group_to_indices[group] for group in val_groups])
        train_groups = np.concatenate([unique_groups[:start], unique_groups[stop:]])
        train_indices = np.concatenate([group_to_indices[group] for group in train_groups])
        yield train_indices, val_indices
        current = stop


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for LambdaRank")
    parser.add_argument("--data-path", type=str, default="train_samples_rank")
    parser.add_argument("--output-dir", type=str, default="artifacts/rank_tuning")
    parser.add_argument("--experiment-name", type=str, default="rank-tuning")
    parser.add_argument("--run-name", type=str, default="tuning")
    parser.add_argument("--group-column", type=str, default=None)
    parser.add_argument("--label-column", type=str, default=None)
    parser.add_argument("--drop-columns", nargs="*", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args(argv)


def _build_dataset(features: pd.DataFrame, labels: pd.Series, groups: pd.Series):
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("LightGBM must be installed for tuning") from exc

    from training.train_rank import _group_sizes

    dataset = lgb.Dataset(features, label=labels, group=_group_sizes(groups.to_numpy()))
    return dataset


def tune_model(args: argparse.Namespace) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data_path = Path(args.data_path)
    df = _load_rank_data(data_path)
    group_col = _detect_group_column(df, args.group_column)
    label_col = _detect_label_column(df, args.label_column)
    df = df.sort_values(group_col).reset_index(drop=True)

    drop_columns = set(args.drop_columns or [])
    features = _prepare_features(df, label_col, group_col, drop_columns)
    labels = df[label_col]
    groups = df[group_col]

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Optuna is required for tuning") from exc

    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("LightGBM must be installed for tuning") from exc

    rng = np.random.default_rng(args.random_state)
    unique_groups = groups.drop_duplicates().to_numpy()
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    groups_mapping = {group: idx for idx, group in enumerate(shuffled_groups)}
    remapped_groups = groups.map(groups_mapping)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_at": [5, 10, 20],
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "random_state": args.random_state,
            "verbose": -1,
        }

        scores: list[float] = []
        for fold, (train_idx, val_idx) in enumerate(_group_kfold_split(remapped_groups.to_numpy(), args.n_splits)):
            train_features = features.iloc[train_idx]
            val_features = features.iloc[val_idx]
            train_labels = labels.iloc[train_idx]
            val_labels = labels.iloc[val_idx]
            train_groups = remapped_groups.iloc[train_idx]
            val_groups = remapped_groups.iloc[val_idx]

            train_dataset = _build_dataset(train_features, train_labels, train_groups)
            val_dataset = _build_dataset(val_features, val_labels, val_groups)

            booster = lgb.train(
                params,
                train_set=train_dataset,
                valid_sets=[val_dataset],
                valid_names=[f"fold{fold}"],
                num_boost_round=1000,
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            preds = booster.predict(val_features, num_iteration=booster.best_iteration or 1000)
            metrics = compute_ranking_metrics(val_labels, preds, val_groups, ks=(20,))
            scores.append(metrics["ndcg@20"])

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=args.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best_params = study.best_params
    best_params.update(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_at": [5, 10, 20],
            "random_state": args.random_state,
        }
    )

    metrics = {"best_value": study.best_value, "trials": len(study.trials)}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    metrics_path = output_dir / "study_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with RunLogger(args.experiment_name, args.run_name) as run:
        run.log_params(best_params)
        run.log_metrics(metrics)
        run.log_artifact(params_path)
        run.log_artifact(metrics_path)

        if study.best_trial:
            trials_summary = [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                }
                for trial in study.trials
            ]
            run.log_text(json.dumps(trials_summary, indent=2), "trials.json")

    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = tune_model(args)
    LOGGER.info("Best tuning metrics: %s", metrics)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

