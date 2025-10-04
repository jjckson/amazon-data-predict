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
        "weights": {
            "bsr_trend_30": 0.35,
            "est_sales_30": 0.25,
            "review_vel_14": 0.20,
            "price_vol_30": -0.10,
            "listing_quality": 0.30,
        }
    },
    "training": {
        "labels": {
            "lookback_days": 28,
            "horizon_days": 30,
            "ratio_threshold": 1.6,
            "percentile_drop": 0.2,
            "min_coverage": 0.7,
            "min_past_sales": 1.0,
            "price_band_quantiles": [0.0, 0.25, 0.75, 1.0],
        },
        "splits": {"train": 0.7, "valid": 0.15, "test": 0.15},
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
