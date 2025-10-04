# Scheduled Jobs (UTC)

```
0 * * * *  python -m pipelines.ingest_raw --asin-file asin_list.csv --site US UK DE --storage-root s3://raw-json/
30 1 * * * python -m pipelines.etl_standardize --date $(date -u +\%F)
0 2 * * *  python -m pipelines.features_build --date $(date -u +\%F)
20 2 * * * python -m pipelines.score_baseline --date $(date -u +\%F)
```

- All jobs run under UTC to align with upstream data collection windows.
- Monitoring hooks should page the on-call channel when any job exits non-zero.
