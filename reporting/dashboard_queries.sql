-- Canonical Business Intelligence views for weekly dashboards.
--
-- vw_top_candidates_daily:
--   Exposes the daily top predicted ASINs alongside confidence and velocity
--   metrics used by merchandising and marketing stakeholders.
-- vw_features_latest:
--   Provides the most recent feature vector for each ASIN/site pair.
-- pred_rank_daily:
--   Supplies row level ranks for daily predictions to support percentile based
--   slicing in downstream dashboards.
--
-- Usage notes:
--   * These views are designed to run on the analytic replica. They rely on
--     the tables populated by the scoring pipelines: ``predictions_daily``,
--     ``category_snapshot_daily`` and ``feature_history``.
--   * Refresh cadence is daily; consumers should filter by ``snapshot_date``
--     to restrict the data range.
--   * All joins are performed on ``asin`` and ``site`` ensuring that ASINs
--     remain unique per marketplace.
--   * Additional filters (category, vendor, family) can be applied by
--     downstream consumers without additional indexes.
--
-- Example:
--   SELECT *
--   FROM vw_top_candidates_daily
--   WHERE snapshot_date = DATE('now', '-1 day')
--   ORDER BY quality_rank
--   LIMIT 100;

DROP VIEW IF EXISTS vw_top_candidates_daily;
CREATE VIEW vw_top_candidates_daily AS
WITH ranked AS (
    SELECT
        p.snapshot_date,
        p.asin,
        p.site,
        p.category,
        p.predicted_revenue,
        p.predicted_units,
        p.confidence_score,
        ROW_NUMBER() OVER (
            PARTITION BY p.snapshot_date, p.site
            ORDER BY p.predicted_revenue DESC, p.confidence_score DESC
        ) AS revenue_rank,
        ROW_NUMBER() OVER (
            PARTITION BY p.snapshot_date, p.site
            ORDER BY p.confidence_score DESC
        ) AS quality_rank
    FROM predictions_daily AS p
)
SELECT
    ranked.snapshot_date,
    ranked.asin,
    ranked.site,
    ranked.category,
    ranked.predicted_revenue,
    ranked.predicted_units,
    ranked.confidence_score,
    ranked.revenue_rank,
    ranked.quality_rank,
    c.avg_category_revenue,
    c.avg_category_units,
    (ranked.predicted_revenue - c.avg_category_revenue) AS revenue_vs_category,
    (ranked.predicted_units - c.avg_category_units) AS units_vs_category
FROM ranked
LEFT JOIN category_snapshot_daily AS c
    ON c.snapshot_date = ranked.snapshot_date
    AND c.category = ranked.category
WHERE ranked.revenue_rank <= 500;

DROP VIEW IF EXISTS vw_features_latest;
CREATE VIEW vw_features_latest AS
WITH latest AS (
    SELECT
        f.asin,
        f.site,
        f.feature_name,
        f.feature_value,
        f.snapshot_ts,
        ROW_NUMBER() OVER (
            PARTITION BY f.asin, f.site, f.feature_name
            ORDER BY f.snapshot_ts DESC
        ) AS feature_rank
    FROM feature_history AS f
)
SELECT
    latest.asin,
    latest.site,
    latest.feature_name,
    latest.feature_value,
    latest.snapshot_ts
FROM latest
WHERE latest.feature_rank = 1;

DROP VIEW IF EXISTS pred_rank_daily;
CREATE VIEW pred_rank_daily AS
SELECT
    p.snapshot_date,
    p.asin,
    p.site,
    p.category,
    p.predicted_revenue,
    p.predicted_units,
    PERCENT_RANK() OVER (
        PARTITION BY p.snapshot_date, p.site
        ORDER BY p.predicted_revenue
    ) AS revenue_percentile,
    PERCENT_RANK() OVER (
        PARTITION BY p.snapshot_date, p.site
        ORDER BY p.predicted_units
    ) AS units_percentile,
    NTILE(100) OVER (
        PARTITION BY p.snapshot_date, p.site
        ORDER BY p.predicted_revenue DESC
    ) AS revenue_ntile,
    NTILE(100) OVER (
        PARTITION BY p.snapshot_date, p.site
        ORDER BY p.predicted_units DESC
    ) AS units_ntile
FROM predictions_daily AS p;
