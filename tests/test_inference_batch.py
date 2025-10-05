from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inference import batch_predict
from inference.batch_predict import BatchPredictor, compute_scores
from inference.schemas import BatchScoreResponse
from utils.config import load_settings


class DummyBooster:
    def __init__(self) -> None:
        self.best_iteration = 3
        self._feature_names = [
            "bsr_trend_30",
            "est_sales_30",
            "review_vel_14",
            "price_vol_30",
            "listing_quality",
        ]

    def feature_name(self) -> list[str]:
        return self._feature_names

    def predict(self, data, num_iteration=None, pred_contrib=False):
        assert list(data.columns) == self._feature_names
        base = np.linspace(0.9, 0.5, len(data), dtype=float)
        if pred_contrib:
            shap_base = np.array([0.05, 0.04, 0.03, -0.02, 0.01, 0.0])
            return np.tile(shap_base, (len(data), 1))
        return base


@pytest.fixture(autouse=True)
def _reset_booster_cache():
    batch_predict._get_cached_booster.cache_clear()
    yield
    batch_predict._get_cached_booster.cache_clear()


@pytest.fixture
def default_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asin": ["A1", "A2"],
            "site": ["US", "US"],
            "dt": ["2024-01-01", "2024-01-01"],
            "category": ["Home", "Home"],
            "price_band": ["20-30", "20-30"],
            "bsr_trend_30": [0.1, 0.2],
            "est_sales_30": [200, 150],
            "review_vel_14": [5, 10],
            "price_vol_30": [0.05, 0.04],
            "listing_quality": [0.8, 0.7],
        }
    )


@pytest.fixture
def default_settings() -> dict:
    return load_settings()


def test_compute_scores_generates_shap(monkeypatch, default_frame, default_settings):
    monkeypatch.setattr(batch_predict, "_load_rank_booster", lambda path: DummyBooster())
    scored = compute_scores(default_frame, default_settings, rank_min=10)
    assert "rank_in_group" in scored.columns
    top_reason = scored.loc[0, "_reason_dict"]
    assert isinstance(top_reason, dict)
    assert len(top_reason) == 5
    assert "lgbm_score" in scored.columns


def test_compute_scores_missing_model(monkeypatch, default_frame, default_settings):
    def _raise(_):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(batch_predict, "_load_rank_booster", _raise)
    scored = compute_scores(default_frame, default_settings, rank_min=10)
    assert scored.empty


def test_compute_scores_missing_columns(monkeypatch, default_frame, default_settings):
    monkeypatch.setattr(batch_predict, "_load_rank_booster", lambda path: DummyBooster())
    incomplete = default_frame.drop(columns=["listing_quality"])
    scored = compute_scores(incomplete, default_settings, rank_min=10)
    assert scored.empty


def test_compute_scores_empty_frame(default_settings):
    empty = pd.DataFrame()
    scored = compute_scores(empty, default_settings, rank_min=10)
    assert scored.empty


def test_batch_predictor_exports(tmp_path, monkeypatch, default_frame, default_settings):
    monkeypatch.setattr(batch_predict, "_load_rank_booster", lambda path: DummyBooster())
    project_root = tmp_path
    predictor = BatchPredictor.from_settings(settings=default_settings, project_root=project_root)

    features_dir = predictor.feature_repo.base_path
    features_dir.mkdir(parents=True, exist_ok=True)
    feature_frame = default_frame.copy()
    feature_frame["dt"] = ["2024-02-01", "2024-02-01"]
    (features_dir / "2024-02-01.csv").write_text(feature_frame.to_csv(index=False))

    response = predictor.run(as_of=date(2024, 2, 1))
    assert isinstance(response, BatchScoreResponse)
    assert response.items
    top_item = response.items[0]
    assert top_item.reason
    assert response.feature_snapshot_path
    assert response.score_artifacts
    assert all(Path(path).exists() for path in response.score_artifacts.values())

    for view in predictor.score_views:
        sql_path = view.sql_path
        assert sql_path.exists()
        content = sql_path.read_text()
        assert view.name in content

    if predictor.feature_view is not None:
        assert predictor.feature_view.sql_path.exists()
