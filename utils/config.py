"""Configuration loading helpers with YAML fallback."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - offline fallback
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

DEFAULT_SETTINGS: Dict[str, Any] = {
    "retry_policy": {
        "max_attempts": 5,
        "base_delay_ms": 500,
        "jitter_ms": 300,
        "max_delay_ms": 60000,
    },
    "rate_limits": {
        "unified_api": {"qpm": 200, "burst": 200, "timeout_sec": 10},
    },
    "features": {"rolling": [7, 14, 30]},
    "scoring": {
        "model_version": "rank-lgbm",
        "model_path": "artifacts/rank/model.txt",
        "price_band_column": "price_band",
        "identifier_columns": ["asin", "site", "dt", "category", "price_band"],
        "group_by_columns": ["site", "category", "price_band"],
        "weights": {
            "bsr_trend_30": 0.35,
            "est_sales_30": 0.25,
            "review_vel_14": 0.20,
            "price_vol_30": -0.10,
            "listing_quality": 0.30,
        }
    },
}


def load_settings(path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    """Load YAML configuration if available, else return defaults."""
    if yaml is None:
        return DEFAULT_SETTINGS

    cfg_path = Path(path or "config/settings.yaml")
    if not cfg_path.exists():
        return DEFAULT_SETTINGS
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
