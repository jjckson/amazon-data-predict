"""Airflow DAG definitions for the ASIN pipeline."""
from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="asin_daily_pipeline",
    schedule_interval="0 1 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1},
    tags=["asin", "analytics"],
) as dag:
    ingest = BashOperator(
        task_id="ingest_raw",
        bash_command="python -m pipelines.ingest_raw --asin A1 --site US --storage-root raw_data",
    )
    etl = BashOperator(
        task_id="etl_standardize",
        bash_command="python -m pipelines.etl_standardize --date {{ ds }}",
    )
    features = BashOperator(
        task_id="features_build",
        bash_command="python -m pipelines.features_build --date {{ ds }}",
    )
    score = BashOperator(
        task_id="score_baseline",
        bash_command="python -m pipelines.score_baseline --date {{ ds }}",
    )

    ingest >> etl >> features >> score
