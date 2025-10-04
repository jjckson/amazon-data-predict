-- Feature store schema for latest feature snapshots
CREATE MATERIALIZED VIEW vw_features_latest AS
SELECT DISTINCT ON (asin, site)
  asin,
  site,
  dt,
  bsr_trend_7,
  bsr_trend_30,
  price_vol_30,
  review_vel_14,
  rating_mean_30,
  est_sales_30,
  listing_quality
FROM features_daily
ORDER BY asin, site, dt DESC;

CREATE VIEW vw_top_candidates_daily AS
SELECT
  s.asin,
  s.site,
  s.dt,
  s.explosive_score,
  s.rank_in_cat,
  d.title,
  d.brand,
  d.category_path,
  m.bsr,
  m.price,
  m.reviews_count
FROM score_baseline_daily s
JOIN dim_asin d USING (asin, site)
JOIN mart_timeseries_daily m USING (asin, site, dt);
