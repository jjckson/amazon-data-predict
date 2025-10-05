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

## Ranking Model Training

- `training/train_rank.py` trains the LambdaRank model and now reports group-based `ndcg@k`,
  `recall@k`, mean average precision (MAP), and lift against an optional baseline score column.
- Supply historical baseline predictions via `--baseline-score-column` to unlock lift curve
  reporting. The script exports three artefacts under the specified `--output-dir`:
  - `metrics.json` with aggregated metrics and the lift curve payload.
  - `metrics.csv` containing a tabular snapshot of the aggregated metrics for spreadsheets.
  - `lift_curve.csv` describing the mean recall, baseline recall, and lift at each requested cut.
- All metrics are logged through `RunLogger` for experiment tracking.

## A/B Testing Utilities

Two helper modules under `abtest/` support deterministic traffic routing and offline
evaluation of experiment variants:

- `abtest/traffic_split.py` exposes `TrafficSplitter` for hashing ASINs (or any
  identifier) into experiment buckets. Allocations can be customised globally or
  per-category and every assignment is recorded via an in-memory audit trail that
  can optionally be forwarded to the standard logging stack.
- `abtest/uplift_eval.py` includes `evaluate_uplift`/`UpliftEvaluator` for
  summarising offline impression logs. The evaluator aggregates CTR, CR, GMV per
  impression, and ROAS, computes lifts versus a control, and performs
  significance testing using normal approximations.

### Offline Evaluation Workflow

1. Build an aggregated dataframe with at least `variant`, `impressions`,
   `clicks`, `conversions`, `gmv`, and `spend` columns. Optionally include
   `gmv_sq` (sum of squared per-impression GMV values), and `roas` / `roas_sq`
   (sum and squared sum of per-impression ROAS values) to unlock p-values for
   the revenue metrics.
2. Call `report = evaluate_uplift(df, control_variant="control")` or use the
   `UpliftEvaluator` class inside notebooks/pipelines.
3. Export results through `report.to_dataframe()` for further processing or
   `report.to_markdown()` for lightweight reporting.

See `tests/test_abtest.py` for an end-to-end example that mirrors offline
evaluation and validates the reporting surface.

## Troubleshooting

| Symptom | Suggested Action |
| --- | --- |
| Frequent 429 responses | Lower QPM in `config/settings.yaml` or verify rate limiter tenant separation |
| Missing daily rows | Inspect validator coverage report and raw payload counts |
| Score volatility | Review site/category MAD values; adjust weights in settings if needed |

## Model Card Governance

- The canonical template lives in [`docs/model_card_template.md`](docs/model_card_template.md). Copy it
  for each release (for example `docs/model_card_v1.2.0.md`) and replace the placeholders with the
  latest training data, metrics, and approvals.
- Update the **Change History** table in the published model card with a new row that captures the
  release tag, deployment date, concise summary of behavioural changes, and the accountable owner.
- Refresh the **Data Scope**, **Metrics**, **Known Limits**, and **Compliance & Governance** sections
  using evidence from the release validation package (offline evaluation notebook, monitoring
  dashboards, policy reviews).
- After merging the release branch, drop the link to the updated model card into the analytics
  registry or deployment tracker so downstream consumers can find it alongside the promoted model
  artefacts.

## Model Registry Operations

`training/registry.py` exposes a lightweight interface for managing model versions either in the
MLflow Model Registry (when a tracking server is reachable) or via a local JSON file for
disconnected development environments. The CLI mirrors the most common flows:

```bash
# Register a model version using the current MLflow tracking URI or the --registry-file override
python -m training.registry register \
  --name rank_v1 \
  --artifact-uri runs:/abc123/model \
  --signature signature.json \
  --metric auc=0.91 \
  --data-span 2024-01-01:2024-01-15 \
  --feature-version features:v5

# Promote a version to production after signature validation against the active deployment
python -m training.registry promote --name rank_v1 --version 2 --to prod

# List tracked versions with their metadata
python -m training.registry list --name rank_v1
```

If MLflow is unavailable (or `mlflow` is not installed), pass `--local --registry-file runs/registry.json`
to persist metadata locally. Promotions to *Staging* or *Production* validate the stored signature and
feature contract so that only compatible models can replace the active deployment. The JSON registry
captures the recorded `artifact_uri`, `metrics`, `data_span`, `feature_version`, and `signature` for
each version to support reproducibility outside MLflow.
