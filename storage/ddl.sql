-- PostgreSQL schema for ASIN analytics
CREATE TABLE dim_asin (
  asin TEXT NOT NULL,
  site TEXT NOT NULL,
  title TEXT,
  brand TEXT,
  category_path TEXT[],
  first_seen TIMESTAMPTZ DEFAULT NOW(),
  last_seen TIMESTAMPTZ,
  PRIMARY KEY (asin, site)
);

CREATE TABLE fact_timeseries_raw (
  asin TEXT NOT NULL,
  site TEXT NOT NULL,
  dt DATE NOT NULL,
  price NUMERIC,
  bsr INTEGER,
  rating NUMERIC,
  reviews_count INTEGER,
  stock_est INTEGER,
  buybox_seller TEXT,
  _ingested_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (asin, site, dt)
);

CREATE TABLE mart_timeseries_daily (
  asin TEXT NOT NULL,
  site TEXT NOT NULL,
  dt DATE NOT NULL,
  price NUMERIC,
  bsr INTEGER,
  rating NUMERIC,
  reviews_count INTEGER,
  stock_est INTEGER,
  buybox_seller TEXT,
  price_valid BOOL,
  bsr_valid BOOL,
  PRIMARY KEY (asin, site, dt)
);

CREATE TABLE fact_reviews (
  asin TEXT NOT NULL,
  site TEXT NOT NULL,
  review_id TEXT NOT NULL,
  dt DATE,
  rating INTEGER,
  title TEXT,
  text TEXT,
  verified BOOL,
  _ingested_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (asin, site, review_id)
);

CREATE TABLE fact_keywords (
  asin TEXT NOT NULL,
  site TEXT NOT NULL,
  keyword TEXT NOT NULL,
  est_search_volume INTEGER,
  cpc NUMERIC,
  difficulty NUMERIC,
  _ingested_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (asin, site, keyword)
);

CREATE TABLE features_daily (
  asin TEXT NOT NULL,
  site TEXT NOT NULL,
  dt DATE NOT NULL,
  bsr_trend_7 NUMERIC,
  bsr_trend_30 NUMERIC,
  price_vol_30 NUMERIC,
  review_vel_14 INTEGER,
  rating_mean_30 NUMERIC,
  est_sales_30 NUMERIC,
  listing_quality NUMERIC,
  PRIMARY KEY (asin, site, dt)
);

CREATE TABLE score_baseline_daily (
  asin TEXT NOT NULL,
  site TEXT NOT NULL,
  dt DATE NOT NULL,
  explosive_score NUMERIC,
  reason JSONB,
  rank_in_cat INTEGER,
  PRIMARY KEY (asin, site, dt)
);
