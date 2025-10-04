"""Ingest raw ASIN data from the unified API."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from connectors.unified_api_client import UnifiedAPIClient
from utils.config import load_settings
from utils.logging import get_logger
from utils.validators import DataValidator

logger = get_logger(__name__)


@dataclass
class IngestResult:
    core: List[Dict]
    timeseries: pd.DataFrame
    reviews: pd.DataFrame
    keywords: pd.DataFrame


class RawStorageWriter:
    """Persist raw JSON payloads to disk/S3-like targets."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, asin: str, site: str, kind: str, payload: Dict) -> None:
        dt_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        path = self.root / site / kind
        path.mkdir(parents=True, exist_ok=True)
        fname = path / f"{asin}_{dt_str}.json"
        fname.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def run(
    *,
    client: UnifiedAPIClient,
    asins: Iterable[str],
    sites: Iterable[str],
    storage_root: str,
    reference_date: date | None = None,
) -> Dict[str, IngestResult]:
    reference_date = reference_date or date.today()
    storage = RawStorageWriter(storage_root)
    validator = DataValidator()

    results: Dict[str, IngestResult] = {}

    for asin in asins:
        for site in sites:
            logger.info("Ingesting asin=%s site=%s", asin, site)
            core = client.get_product_core(asin, site)
            storage.write_json(asin, site, "core", core)

            timeseries = client.get_product_timeseries(
                asin,
                site,
                start=(reference_date - timedelta(days=2)).isoformat(),
                end=reference_date.isoformat(),
            )
            storage.write_json(asin, site, "timeseries", {"items": timeseries})
            ts_df = pd.DataFrame(timeseries)
            if not ts_df.empty:
                ts_df["asin"] = asin
                ts_df["site"] = site
                ts_df["dt"] = pd.to_datetime(ts_df["dt"]).dt.date

            reviews = client.get_reviews_batch(
                asin,
                site,
                since=(reference_date - timedelta(days=7)).isoformat(),
            )
            storage.write_json(asin, site, "reviews", {"items": reviews})
            reviews_df = pd.DataFrame(reviews)
            if not reviews_df.empty:
                reviews_df["asin"] = asin
                reviews_df["site"] = site

            keywords = list(client.get_keywords(asin, site))
            storage.write_json(asin, site, "keywords", {"items": keywords})
            keywords_df = pd.DataFrame(keywords)
            if not keywords_df.empty:
                keywords_df["asin"] = asin
                keywords_df["site"] = site

            validation = validator.validate_consistency(
                {"timeseries": len(timeseries)},
                {"timeseries": len(ts_df)},
            )
            logger.info("Validation status: %s", validation.status)

            results[f"{asin}:{site}"] = IngestResult(
                core=[core],
                timeseries=ts_df,
                reviews=reviews_df,
                keywords=keywords_df,
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw ingestion pipeline")
    parser.add_argument("--asin", nargs="+", required=True)
    parser.add_argument("--site", nargs="+", required=True)
    parser.add_argument("--storage-root", default="raw_data")
    args = parser.parse_args()

    settings = load_settings()
    client = UnifiedAPIClient(
        base_url=settings.get("unified_api_base_url", "https://api"),
        token="${UNIFIED_API_TOKEN}",
    )
    run(
        client=client,
        asins=args.asin,
        sites=args.site,
        storage_root=args.storage_root,
    )


if __name__ == "__main__":
    main()
