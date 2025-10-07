"""Dry run helper that validates dashboard SQL against SQLite."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
import pandas as pd

SQL_FILE = Path(__file__).with_name("dashboard_queries.sql")


def _load_sql() -> str:
    if not SQL_FILE.exists():
        raise FileNotFoundError(f"Dashboard SQL file not found: {SQL_FILE}")
    return SQL_FILE.read_text(encoding="utf-8")


def _create_sample_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS predictions_daily (
            snapshot_date TEXT,
            asin TEXT,
            site TEXT,
            category TEXT,
            predicted_revenue REAL,
            predicted_units REAL,
            confidence_score REAL
        );

        CREATE TABLE IF NOT EXISTS category_snapshot_daily (
            snapshot_date TEXT,
            category TEXT,
            avg_category_revenue REAL,
            avg_category_units REAL
        );

        CREATE TABLE IF NOT EXISTS feature_history (
            asin TEXT,
            site TEXT,
            feature_name TEXT,
            feature_value REAL,
            snapshot_ts TEXT
        );

        CREATE TABLE IF NOT EXISTS ai_comment_summaries (
            snapshot_date TEXT,
            asin TEXT,
            site TEXT,
            summary_text TEXT
        );

        CREATE TABLE IF NOT EXISTS ai_keyword_clusters (
            snapshot_date TEXT,
            asin TEXT,
            site TEXT,
            cluster_label TEXT
        );
        """
    )

    today = dt.date.today()
    rows = []
    for offset in range(0, 7):
        date_value = today - dt.timedelta(days=offset)
        for idx in range(1, 6):
            rows.append(
                (
                    date_value.isoformat(),
                    f"ASIN{idx}",
                    "US",
                    "Electronics" if idx % 2 == 0 else "Toys",
                    1000 + idx * 50 + offset * 10,
                    20 + idx * 2,
                    0.8 + idx * 0.01,
                )
            )
    connection.executemany(
        "INSERT INTO predictions_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    category_rows = []
    for offset in range(0, 14):
        date_value = today - dt.timedelta(days=offset)
        category_rows.append(
            (date_value.isoformat(), "Electronics", 1200 + offset * 5, 25 + offset * 0.5)
        )
        category_rows.append(
            (date_value.isoformat(), "Toys", 900 + offset * 3, 18 + offset * 0.4)
        )
    connection.executemany(
        "INSERT INTO category_snapshot_daily VALUES (?, ?, ?, ?)",
        category_rows,
    )

    feature_rows = []
    for idx in range(1, 4):
        feature_rows.append(
            (
                f"ASIN{idx}",
                "US",
                "feature_a",
                0.1 * idx,
                (dt.datetime.combine(today, dt.time.min) - dt.timedelta(hours=idx)).isoformat(),
            )
        )
        feature_rows.append(
            (
                f"ASIN{idx}",
                "US",
                "feature_a",
                0.1 * idx + 0.05,
                (dt.datetime.combine(today, dt.time.min) - dt.timedelta(hours=idx - 1)).isoformat(),
            )
        )
    connection.executemany(
        "INSERT INTO feature_history VALUES (?, ?, ?, ?, ?)",
        feature_rows,
    )

    comment_rows = []
    keyword_rows = []
    for offset in range(0, 7):
        date_value = today - dt.timedelta(days=offset)
        comment_rows.append(
            (
                date_value.isoformat(),
                "ASIN1",
                "US",
                "Great attachment quality, trending upward",
            )
        )
        keyword_rows.append(
            (
                date_value.isoformat(),
                "ASIN1",
                "US",
                "travel accessories",
            )
        )
    connection.executemany(
        "INSERT INTO ai_comment_summaries VALUES (?, ?, ?, ?)",
        comment_rows,
    )
    connection.executemany(
        "INSERT INTO ai_keyword_clusters VALUES (?, ?, ?, ?)",
        keyword_rows,
    )


def execute_dashboard_views(connection: sqlite3.Connection) -> None:
    """Execute the dashboard SQL script against the provided connection."""

    sql = _load_sql()
    connection.executescript(sql)


def run_dry_run(connection: sqlite3.Connection | None = None) -> dict[str, pd.DataFrame]:
    """Run the dashboard SQL against a temporary in-memory SQLite database."""

    owned_connection = connection is None
    connection = connection or sqlite3.connect(":memory:")
    if owned_connection:
        connection.row_factory = sqlite3.Row
    try:
        _create_sample_tables(connection)
        execute_dashboard_views(connection)
        result: dict[str, pd.DataFrame] = {}
        for view_name in ("vw_top_candidates_daily", "vw_features_latest", "pred_rank_daily"):
            frame = pd.read_sql_query(f"SELECT * FROM {view_name} LIMIT 5", connection)
            result[view_name] = frame
        return result
    finally:
        if owned_connection:
            connection.close()


def main() -> None:
    outputs = run_dry_run()
    for view_name, frame in outputs.items():
        print(f"{view_name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
