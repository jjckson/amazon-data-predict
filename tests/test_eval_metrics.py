from __future__ import annotations

import math

import numpy as np
import pytest

from training.eval_metrics import compute_ranking_metrics, lift_curve


def test_compute_ranking_metrics_includes_recall_and_map() -> None:
    y_true = [1, 0, 1, 0, 1, 0]
    y_score = [0.9, 0.1, 0.8, 0.2, 0.7, 0.05]
    baseline_score = [0.2, 0.3, 0.1, 0.1, 0.2, 0.05]
    groups = ["g1", "g1", "g1", "g2", "g2", "g2"]

    metrics = compute_ranking_metrics(y_true, y_score, groups, ks=(1, 2), baseline_score=baseline_score)

    assert pytest.approx(metrics["ndcg@1"], rel=1e-6) == 1.0
    assert pytest.approx(metrics["recall@1"], rel=1e-6) == 0.75
    assert pytest.approx(metrics["recall@2"], rel=1e-6) == 1.0
    assert pytest.approx(metrics["map"], rel=1e-6) == 1.0
    assert pytest.approx(metrics["baseline_recall@1"], rel=1e-6) == 0.5
    assert pytest.approx(metrics["baseline_recall@2"], rel=1e-6) == 0.75
    assert pytest.approx(metrics["lift@1"], rel=1e-6) == 0.25
    assert pytest.approx(metrics["lift@2"], rel=1e-6) == 0.25
    assert pytest.approx(metrics["baseline_map"], rel=1e-6) == 0.7916666666666666


def test_compute_ranking_metrics_handles_singleton_and_empty_groups() -> None:
    y_true: list[float] = [1, 0]
    y_score = [0.9, 0.1]
    groups = ["solo", "all_zero"]

    metrics = compute_ranking_metrics(y_true, y_score, groups, ks=(1,))

    assert pytest.approx(metrics["recall@1"], rel=1e-6) == 0.5
    assert pytest.approx(metrics["map"], rel=1e-6) == 0.5


def test_compute_ranking_metrics_empty_input_returns_nan() -> None:
    metrics = compute_ranking_metrics([], [], [], ks=(1,))

    assert math.isnan(metrics["ndcg@1"])
    assert math.isnan(metrics["recall@1"])
    assert math.isnan(metrics["map"])


def test_lift_curve_outputs_mean_difference() -> None:
    y_true = [1, 0, 0, 1]
    y_score = [0.9, 0.8, 0.7, 0.1]
    baseline = [0.4, 0.3, 0.2, 0.1]
    groups = ["a", "a", "b", "b"]

    curve = lift_curve(y_true, y_score, baseline, groups, ks=(1, 2))

    assert len(curve) == 2
    k1 = next(item for item in curve if item["k"] == 1)
    assert pytest.approx(k1["recall"], rel=1e-6) == 0.5
    assert pytest.approx(k1["baseline_recall"], rel=1e-6) == 0.5
    assert pytest.approx(k1["lift"], rel=1e-6) == 0.0


def test_lift_curve_requires_matching_shapes() -> None:
    with pytest.raises(ValueError):
        lift_curve([1, 0], [0.9, 0.8], [0.1], ["a", "a"], ks=(1,))


def test_lift_curve_with_zero_baseline_recall() -> None:
    y_true = np.array([1, 0, 0])
    y_score = np.array([0.9, 0.1, 0.05])
    baseline = np.array([0.01, 0.1, 0.05])
    groups = np.array(["g1", "g1", "g1"])

    curve = lift_curve(y_true, y_score, baseline, groups, ks=(1,))

    assert pytest.approx(curve[0]["lift"], rel=1e-6) == curve[0]["recall"]
