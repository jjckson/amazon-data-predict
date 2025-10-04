from __future__ import annotations

import json

import pandas as pd

from training.build_labels import _compute_rank_target, _normalise_for_json


def test_highest_future_sales_receives_largest_rank() -> None:
    df = pd.DataFrame(
        {
            "asin": ["A", "A", "A", "B", "B"],
            "future_sales": [10.0, 20.0, 5.0, 3.0, 7.0],
        }
    )

    for asin, group in df.groupby("asin", sort=False):
        ranked = _compute_rank_target(group)
        top_row = ranked.loc[group["future_sales"].idxmax()]
        assert top_row["y_rank"] == ranked["y_rank"].max(), asin


def test_meta_serialised_as_string_round_trip() -> None:
    meta = pd.Series(
        [
            {
                "features": ["wireless", "bluetooth"],
                "dimensions": {"width": 2.3, "height": [1, 2, None]},
            },
            {"features": []},
        ]
    )

    serialised = meta.apply(lambda value: json.dumps(_normalise_for_json(value)))
    round_tripped = serialised.apply(json.loads)
    expected = meta.apply(_normalise_for_json)

    assert list(round_tripped) == list(expected)
