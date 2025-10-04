"""Static Keepa payload used for connector tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict


def _ts(days: int) -> int:
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return int((base + timedelta(days=days)).timestamp())


def sample_product_payload() -> Dict:
    return {
        "asin": "B0001",
        "domainId": 1,
        "title": "Sample Product",
        "brand": "BrandCo",
        "categoryTree": [
            {"name": "Root"},
            {"name": "Sub"},
        ],
        "imagesCSV": "https://example.com/a.jpg,https://example.com/b.jpg",
        "history": {
            "salesRanks": {
                "default": [
                    {"timestamp": _ts(0), "value": 1500},
                    {"timestamp": _ts(1), "value": 1400},
                ]
            },
            "buyBoxPrice": [
                {"timestamp": _ts(0), "value": 1999},
                {"timestamp": _ts(1), "value": 1899},
            ],
            "rating": [
                {"timestamp": _ts(0), "value": 450},
            ],
            "reviewCount": [
                {"timestamp": _ts(0), "value": 200},
                {"timestamp": _ts(1), "value": 210},
            ],
            "stock": [
                {"timestamp": _ts(0), "value": 25},
            ],
            "buyBoxSeller": [
                {"timestamp": _ts(0), "value": "SELLER1"},
            ],
        },
        "reviews": [
            {
                "reviewId": "R1",
                "timestamp": _ts(0),
                "rating": 5,
                "title": "Great",
                "text": "Works as advertised",
                "verified": True,
            }
        ],
        "keywords": [
            {
                "keyword": "widget",
                "searchVolume": 12000,
                "cpc": 1.5,
                "difficulty": 0.6,
            }
        ],
    }
