# Amazon ASIN Data Pipeline

This repository provides the ingestion-to-scoring backbone that collects daily Amazon ASIN
signals from multiple data sources, standardises the information into relational tables,
builds analytical features, and produces a baseline "Explosive Score" ranking for down-stream
dashboards and machine learning training.

## Repository Layout

```
config/                 # Environment settings and credential templates
connectors/             # API clients with rate limiting + retry logic
pipelines/              # Ingestion, ETL, feature, and scoring orchestrators
training/               # Label generation, ranker training, hyper-parameter tuning
inference/              # Batch scoring utilities for trained models
storage/                # Database DDL, feature-store schema, migrations
utils/                  # Shared helpers for logging, config, validation
jobs/                   # Scheduling assets (cron + Airflow DAG)
tests/                  # Unit tests for connectors + pipelines
```

## Getting Started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # create as needed
   ```
2. **Configure settings**
   - Copy `config/secrets.example.env` to `.env` and fill tokens/passwords.
   - Adjust `config/settings.yaml` with environment, rate limits, windows, and storage targets.
3. **Run tests**
   ```bash
   pytest
   ```

## Pipelines

| Script | Purpose | Key Responsibilities |
| --- | --- | --- |
| `pipelines/ingest_raw.py` | Fetch ASIN data from the unified API | Calls product core/timeseries/review/keyword endpoints, persists raw JSON, validates counts |
| `pipelines/etl_standardize.py` | Cleanse raw facts into daily mart | Deduplicate, handle anomalies, enforce schema, produce quality report |
| `pipelines/features_build.py` | Generate rolling features | Compute BSR trends, price volatility, review velocity, listing quality |
| `pipelines/score_baseline.py` | Compute baseline explosive score | Robust z-scoring per site/category with configurable weights |
| `training/build_labels.py` | Generate supervised samples | Produces binary/ranking/regression labels with temporal splits |

Run label generation once the mart/features tables are populated:

```bash
python -m training.build_labels --mart mart_timeseries_daily.parquet \
  --features features_daily.parquet --output-dir artifacts/samples
```

The script emits `train_samples_bin|rank|reg` parquet files with split metadata, feature JSON blobs, and summary logs.

## Model Training & Evaluation

Once labels exist, train the baseline LightGBM ranker:

```bash
python -m training.train_rank --data artifacts/samples/train_samples_rank.parquet \
  --output artifacts/lgb_rank.txt
```

- Hyper-parameter search (optional): `python -m training.tune_rank --data ... --trials 50`
- Metrics reporting: `python -m training.eval_metrics --predictions predictions.csv --k 10 20`

Batch inference on fresh feature snapshots:

```bash
python -m inference.batch_predict --model artifacts/lgb_rank.txt \
  --features features_daily_2025-10-04.parquet \
  --output exports/pred_rank_daily.parquet
```

Outputs include the model score and group-wise rank, ready to be joined with `dim_asin` for dashboards or downstream triage.

Each script exposes a `run` helper that can be reused inside orchestrators and a CLI entry point
(see `jobs/cron.md`).

## Storage

- `storage/ddl.sql` – Primary Postgres schema for dimensions, raw facts, marts, features, and scoring tables.
- `storage/feature_store_schema.sql` – Convenience views for dashboards and model training.
- `storage/migrations/` – Placeholder directory for future Flyway/Alembic migrations.

## Scheduling

- `jobs/cron.md` documents the production cron cadence (UTC) for daily processing.
- `jobs/airflow_dags.py` sketches an Airflow DAG that enforces sequential task execution with retries.

## Monitoring & Quality

- Logging is centralised via `utils/logging.py` with optional JSON output for ingestion to ELK/Datadog.
- `utils/rate_limiter.py` and `utils/backoff.py` ensure connectors respect upstream quotas and recover from 429/5xx responses.
- `utils/validators.py` delivers daily coverage/anomaly checks; incorporate into alerting webhooks.

## Exports

The scoring pipeline materialises data into `score_baseline_daily`. Use the views defined in
`storage/feature_store_schema.sql` to power BI dashboards or CSV exports (e.g. `exports/top_candidates_YYYYMMDD.csv`).

## Local Development

- Use `pipelines/ingest_raw.py` with a small ASIN set (≤500) when iterating to avoid throttling.
- Mock connectors in tests using `tests/` fixtures; see provided unit tests as templates.
- Keep credentials exclusively in `.env` (never commit secrets).

## Troubleshooting

| Symptom | Suggested Action |
| --- | --- |
| Frequent 429 responses | Lower QPM in `config/settings.yaml` or verify rate limiter tenant separation |
| Missing daily rows | Inspect validator coverage report and raw payload counts |
| Score volatility | Review site/category MAD values; adjust weights in settings if needed |
