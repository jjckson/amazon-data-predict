from __future__ import annotations

import pandas as pd

from training.build_labels import _compute_rank_target


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
