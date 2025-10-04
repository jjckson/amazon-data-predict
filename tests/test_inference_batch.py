from __future__ import annotations

from datetime import date

import pandas as pd

from inference.batch_predict import BatchPredictor, compute_scores
from inference.schemas import BatchScoreResponse
from utils.config import load_settings


def test_compute_scores_generates_shap():
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "asin": ["A1", "A2"],
            "site": ["US", "US"],
            "dt": ["2024-01-01", "2024-01-01"],
            "category": ["Home", "Home"],
            "bsr_trend_30": [0.1, 0.2],
            "est_sales_30": [200, 150],
            "review_vel_14": [5, 10],
            "price_vol_30": [0.05, 0.04],
            "listing_quality": [0.8, 0.7],
        }
    )

    scored = compute_scores(frame, settings, rank_min=10)
    assert "group_rank" in scored.columns
    assert scored.loc[0, "_reason_dict"]


def test_batch_predictor_exports(tmp_path):
    project_root = tmp_path
    settings = load_settings()
    predictor = BatchPredictor.from_settings(settings=settings, project_root=project_root)

    features_dir = predictor.feature_repo.base_path
    features_dir.mkdir(parents=True, exist_ok=True)
    feature_frame = pd.DataFrame(
        {
            "asin": ["B1", "B2"],
            "site": ["US", "US"],
            "dt": ["2024-02-01", "2024-02-01"],
            "category": ["Kitchen", "Kitchen"],
            "bsr_trend_30": [0.4, 0.3],
            "est_sales_30": [400, 380],
            "review_vel_14": [15, 12],
            "price_vol_30": [0.06, 0.07],
            "listing_quality": [0.9, 0.85],
        }
    )
    (features_dir / "2024-02-01.csv").write_text(feature_frame.to_csv(index=False))

    response = predictor.run(as_of=date(2024, 2, 1))
    assert isinstance(response, BatchScoreResponse)
    assert response.items
    top_item = response.items[0]
    assert top_item.reason

    for view in predictor.score_views:
        sql_path = view.sql_path
        assert sql_path.exists()
        content = sql_path.read_text()
        assert view.name in content

    if predictor.feature_view is not None:
        assert predictor.feature_view.sql_path.exists()
