"""Shared evaluation metrics utilities for training scripts."""
from __future__ import annotations

import math
from typing import Iterable, MutableMapping, Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


def _to_numpy(array: ArrayLike) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array.astype(float)
    return np.asarray(list(array), dtype=float)


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    if relevance.size == 0 or k <= 0:
        return 0.0
    topk = relevance[:k]
    discounts = 1.0 / np.log2(np.arange(2, topk.size + 2))
    gains = np.power(2.0, topk) - 1.0
    return float(np.sum(gains * discounts))


def ndcg_at_k(relevance: ArrayLike, ideal_relevance: ArrayLike | None = None, k: int = 20) -> float:
    rel = _to_numpy(relevance)
    ideal = _to_numpy(ideal_relevance) if ideal_relevance is not None else np.sort(rel)[::-1]
    normaliser = dcg_at_k(ideal, k)
    if normaliser == 0:
        return 0.0
    return dcg_at_k(rel, k) / normaliser


def recall_at_k(relevance: np.ndarray, k: int) -> float:
    if relevance.size == 0 or k <= 0:
        return 0.0
    total_relevant = float(np.sum(relevance > 0))
    if total_relevant == 0:
        return 0.0
    retrieved = relevance[:k]
    return float(np.sum(retrieved > 0) / total_relevant)


def average_precision(relevance: np.ndarray) -> float:
    if relevance.size == 0:
        return 0.0
    hits = relevance > 0
    n_relevant = float(np.sum(hits))
    if n_relevant == 0:
        return 0.0
    cumulative_hits = np.cumsum(hits)
    precision_at_k = cumulative_hits / (np.arange(relevance.size) + 1)
    ap = np.sum(precision_at_k * hits) / n_relevant
    return float(ap)


def _group_indices(group: Iterable[int] | Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    group_np = np.asarray(list(group))
    unique_groups, inverse = np.unique(group_np, return_inverse=True)
    return unique_groups, inverse


def lift_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    baseline_score: ArrayLike,
    group: Sequence[int] | Sequence[str],
    ks: Sequence[int] = (5, 10, 20),
) -> list[dict[str, float]]:
    y_true_np = _to_numpy(y_true)
    y_score_np = _to_numpy(y_score)
    baseline_np = _to_numpy(baseline_score)

    if y_true_np.shape != y_score_np.shape or y_true_np.shape != baseline_np.shape:
        raise ValueError("y_true, y_score, and baseline_score must have the same shape")

    unique_groups, inverse = _group_indices(group)

    curve: list[dict[str, float]] = []
    for k in ks:
        recalls: list[float] = []
        baseline_recalls: list[float] = []
        lifts: list[float] = []

        for group_idx, _ in enumerate(unique_groups):
            mask = inverse == group_idx
            group_true = y_true_np[mask]
            if group_true.size == 0:
                continue

            sorted_true = group_true[np.argsort(y_score_np[mask])[::-1]]
            recall = recall_at_k(sorted_true, k)

            baseline_sorted_true = group_true[np.argsort(baseline_np[mask])[::-1]]
            baseline_recall = recall_at_k(baseline_sorted_true, k)

            recalls.append(recall)
            baseline_recalls.append(baseline_recall)
            lifts.append(recall - baseline_recall)

        curve.append(
            {
                "k": int(k),
                "recall": float(np.mean(recalls)) if recalls else float("nan"),
                "baseline_recall": float(np.mean(baseline_recalls)) if baseline_recalls else float("nan"),
                "lift": float(np.mean(lifts)) if lifts else float("nan"),
            }
        )

    return curve


def compute_ranking_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    group: Sequence[int] | Sequence[str],
    ks: Sequence[int] = (5, 10, 20),
    baseline_score: ArrayLike | None = None,
) -> dict[str, float]:
    y_true_np = _to_numpy(y_true)
    y_score_np = _to_numpy(y_score)
    group_np = np.asarray(list(group))

    if baseline_score is not None:
        baseline_np = _to_numpy(baseline_score)
        if baseline_np.shape != y_true_np.shape:
            raise ValueError("baseline_score must match the shape of y_true")
    else:
        baseline_np = None

    metrics: MutableMapping[str, list[float]] = {f"ndcg@{k}": [] for k in ks}
    recalls: MutableMapping[str, list[float]] = {f"recall@{k}": [] for k in ks}
    baseline_recalls: MutableMapping[str, list[float]] = {f"baseline_recall@{k}": [] for k in ks} if baseline_np is not None else {}
    lifts: MutableMapping[str, list[float]] = {f"lift@{k}": [] for k in ks} if baseline_np is not None else {}
    average_precisions: list[float] = []
    baseline_average_precisions: list[float] = [] if baseline_np is not None else []

    for group_id in np.unique(group_np):
        mask = group_np == group_id
        group_true = y_true_np[mask]
        group_scores = y_score_np[mask]
        if group_true.size == 0:
            continue
        order = np.argsort(group_scores)[::-1]
        sorted_true = group_true[order]
        ideal_true = np.sort(group_true)[::-1]
        average_precisions.append(average_precision(sorted_true))
        for k in ks:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(sorted_true, ideal_true, k))
            recalls[f"recall@{k}"].append(recall_at_k(sorted_true, k))

        if baseline_np is not None:
            baseline_scores = baseline_np[mask]
            baseline_sorted_true = group_true[np.argsort(baseline_scores)[::-1]]
            baseline_average_precisions.append(average_precision(baseline_sorted_true))
            for k in ks:
                baseline_value = recall_at_k(baseline_sorted_true, k)
                baseline_recalls[f"baseline_recall@{k}"].append(baseline_value)
                lifts[f"lift@{k}"].append(recalls[f"recall@{k}"][-1] - baseline_value)

    averaged = {metric: float(np.mean(values)) if values else float("nan") for metric, values in metrics.items()}
    averaged.update({metric: float(np.mean(values)) if values else float("nan") for metric, values in recalls.items()})

    averaged["map"] = float(np.mean(average_precisions)) if average_precisions else float("nan")

    if baseline_np is not None:
        averaged.update(
            {
                metric: float(np.mean(values)) if values else float("nan")
                for metric, values in baseline_recalls.items()
            }
        )
        averaged["baseline_map"] = (
            float(np.mean(baseline_average_precisions)) if baseline_average_precisions else float("nan")
        )
        averaged.update({metric: float(np.mean(values)) if values else float("nan") for metric, values in lifts.items()})

    return averaged


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------


def _binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def _binary_probabilities(y_pred_proba: ArrayLike) -> np.ndarray:
    probs = _to_numpy(y_pred_proba)
    if probs.ndim == 2 and probs.shape[1] == 2:
        probs = probs[:, 1]
    return np.clip(probs, 1e-15, 1 - 1e-15)


def roc_auc_score(y_true: ArrayLike, y_score: ArrayLike) -> float:
    y_true_np = _to_numpy(y_true)
    y_score_np = _to_numpy(y_score)
    pos = y_true_np == 1
    neg = y_true_np == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score_np)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score_np) + 1)
    sum_ranks_pos = np.sum(ranks[pos])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def log_loss(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
    y_true_np = _to_numpy(y_true)
    probs = _binary_probabilities(y_pred_proba)
    losses = y_true_np * np.log(probs) + (1 - y_true_np) * np.log(1 - probs)
    return float(-np.mean(losses))


def binary_classification_metrics(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_np = _to_numpy(y_true)
    probs = _binary_probabilities(y_pred_proba)
    preds = (probs >= threshold).astype(int)

    tp, tn, fp, fn = _binary_confusion(y_true_np, preds)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    balanced_accuracy = (recall + specificity) / 2 if not math.isnan(recall) and not math.isnan(specificity) else float("nan")

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision) if not math.isnan(precision) else float("nan"),
        "recall": float(recall) if not math.isnan(recall) else float("nan"),
        "f1": float(f1) if not math.isnan(f1) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy) if not math.isnan(balanced_accuracy) else float("nan"),
        "roc_auc": roc_auc_score(y_true_np, probs),
        "log_loss": log_loss(y_true_np, probs),
    }
    return metrics


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    diff = y_true_np - y_pred_np
    mse = float(np.mean(np.square(diff)))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(diff)))
    if y_true_np.size == 0:
        r2 = float("nan")
    else:
        total_var = float(np.sum(np.square(y_true_np - np.mean(y_true_np))))
        r2 = 1 - (np.sum(np.square(diff)) / total_var) if total_var else float("nan")
    return {"rmse": rmse, "mae": mae, "mse": mse, "r2": float(r2)}


__all__ = [
    "binary_classification_metrics",
    "regression_metrics",
    "compute_ranking_metrics",
    "recall_at_k",
    "average_precision",
    "lift_curve",
    "ndcg_at_k",
    "roc_auc_score",
    "log_loss",
]

