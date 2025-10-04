"""Utility to compute ranking evaluation metrics from predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


def dcg(scores: Iterable[float]) -> float:
    scores = np.asarray(list(scores), dtype=float)
    if scores.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, scores.size + 2))
    return float(np.sum(scores * discounts))


def ndcg_at_k(relevance: Iterable[float], k: int) -> float:
    rel = np.asarray(list(relevance), dtype=float)[:k]
    ideal = np.sort(rel)[::-1]
    denom = dcg(ideal)
    if denom == 0.0:
        return 0.0
    return dcg(rel) / denom


def recall_at_k(relevance: Iterable[int], k: int) -> float:
    rel = np.asarray(list(relevance), dtype=int)
    if rel.sum() == 0:
        return 0.0
    return float(rel[:k].sum() / rel.sum())


def evaluate_group(df: pd.DataFrame, k: int) -> Tuple[float, float]:
    return ndcg_at_k(df["label"], k), recall_at_k(df["label"], k)


def evaluate(df: pd.DataFrame, group_col: str, k_values: Iterable[int]) -> pd.DataFrame:
    results = []
    for k in k_values:
        ndcgs: list[float] = []
        recalls: list[float] = []
        for _, group in df.groupby(group_col):
            ndcg_k, recall_k = evaluate_group(group.sort_values("score", ascending=False), k)
            ndcgs.append(ndcg_k)
            recalls.append(recall_k)
        results.append({"k": k, "ndcg": float(np.mean(ndcgs)), "recall": float(np.mean(recalls))})
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ranking metrics")
    parser.add_argument("--predictions", required=True, help="CSV/Parquet predictions with labels")
    parser.add_argument("--group-col", default="group_id")
    parser.add_argument("--score-col", default="lgbm_score")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--k", nargs="*", type=int, default=[10, 20])
    parser.add_argument("--output")
    args = parser.parse_args()

    path = Path(args.predictions)
    if path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if args.score_col != "score":
        df = df.rename(columns={args.score_col: "score"})
    if args.label_col != "label":
        df = df.rename(columns={args.label_col: "label"})

    metrics = evaluate(df, args.group_col, args.k)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_json(out_path, orient="records", lines=False)
        logger.info("Wrote metrics to %s", out_path)
    else:
        logger.info("Metrics:\n%s", metrics.to_string(index=False))


if __name__ == "__main__":
    main()
