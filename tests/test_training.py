from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")

from training.build_labels import build_labels


def test_meta_serialised_as_string_round_trip():
    frame = pd.DataFrame(
        {
            "asin": ["A", "B"],
            "site": ["US", "US"],
            "dt": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "target": [1, 0],
            "feature_one": [0.1, 0.2],
            "feature_two": [10, 20],
            "brand": ["BrandX", None],
            "category": ["Home", "Home"],
            "first_seen": [pd.Timestamp("2023-12-01"), pd.NaT],
        }
    )
    settings = {
        "training": {
            "label_column": "target",
            "feature_columns": ["feature_one", "feature_two"],
            "meta_columns": ["brand", "category", "first_seen"],
        }
    }

    labels = build_labels(frame, settings)

    assert labels["meta"].map(type).eq(str).all()
    assert labels["feature_vector"].map(type).eq(str).all()

    first_meta = json.loads(labels.loc[0, "meta"])
    assert first_meta == {
        "brand": "BrandX",
        "category": "Home",
        "first_seen": "2023-12-01T00:00:00",
    }

    second_meta = json.loads(labels.loc[1, "meta"])
    assert second_meta == {
        "brand": None,
        "category": "Home",
        "first_seen": None,
    }
