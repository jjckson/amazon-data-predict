"""Lightweight schema utilities shared by batch inference components."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd


def _ensure_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"Unsupported date value: {value!r}")


@dataclass
class FeatureRecord:
    """Canonical representation of a feature vector for an ASIN."""

    asin: str
    site: str
    dt: date
    category: Optional[str] = None
    bsr_trend_30: Optional[float] = None
    est_sales_30: Optional[float] = None
    review_vel_14: Optional[float] = None
    price_vol_30: Optional[float] = None
    listing_quality: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dt = _ensure_date(self.dt)

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "FeatureRecord":
        base_keys = {
            "asin",
            "site",
            "dt",
            "category",
            "bsr_trend_30",
            "est_sales_30",
            "review_vel_14",
            "price_vol_30",
            "listing_quality",
        }
        data = {key: payload.get(key) for key in base_keys if key in payload}
        data.setdefault("asin", payload["asin"])
        data.setdefault("site", payload["site"])
        data.setdefault("dt", payload["dt"])
        data["dt"] = _ensure_date(data["dt"])
        extras = {k: v for k, v in payload.items() if k not in base_keys}
        return cls(extra=extras, **data)  # type: ignore[arg-type]

    def to_record(self) -> Dict[str, Any]:
        record = {
            "asin": self.asin,
            "site": self.site,
            "dt": self.dt,
            "category": self.category,
            "bsr_trend_30": self.bsr_trend_30,
            "est_sales_30": self.est_sales_30,
            "review_vel_14": self.review_vel_14,
            "price_vol_30": self.price_vol_30,
            "listing_quality": self.listing_quality,
        }
        record.update(self.extra)
        return record


@dataclass
class BatchScoreRequest:
    """Request payload for batch scoring."""

    items: List[FeatureRecord]
    as_of: Optional[date] = None
    model_version: str = "baseline"
    rank_min: int = 20

    def __post_init__(self) -> None:
        if self.as_of is not None:
            self.as_of = _ensure_date(self.as_of)
        if self.rank_min < 1:
            raise ValueError("rank_min must be positive")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BatchScoreRequest":
        items_payload = payload.get("items", [])
        items = [FeatureRecord.from_mapping(item) for item in items_payload]
        as_of = payload.get("as_of")
        model_version = payload.get("model_version", "baseline")
        rank_min = payload.get("rank_min", 20)
        return cls(items=items, as_of=as_of, model_version=model_version, rank_min=rank_min)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.items:
            return pd.DataFrame()
        records = [item.to_record() for item in self.items]
        frame = pd.DataFrame(records)
        if "dt" in frame.columns:
            frame["dt"] = pd.to_datetime(frame["dt"])
        return frame


@dataclass
class SummaryRequestSchema:
    """Request payload for invoking the AI summarisation endpoint."""

    text: str
    max_sentences: Optional[int] = None
    language: Optional[str] = None

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("text must not be empty")
        if self.max_sentences is not None and self.max_sentences < 1:
            raise ValueError("max_sentences must be positive when provided")


@dataclass
class KeywordClusterRequestSchema:
    """Request payload for keyword clustering via an AI provider."""

    keywords: List[str]
    language: Optional[str] = None
    top_k: Optional[int] = None

    def __post_init__(self) -> None:
        cleaned_keywords = [keyword.strip() for keyword in self.keywords if keyword and keyword.strip()]
        if not cleaned_keywords:
            raise ValueError("keywords must contain at least one non-empty value")
        self.keywords = cleaned_keywords
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive when provided")


@dataclass
class ScoredItem:
    asin: str
    site: str
    dt: date
    category: Optional[str]
    explosive_score: float
    lgbm_score: float
    rank_in_cat: Optional[int]
    rank_in_group: Optional[int]
    reason: Dict[str, float]
    group_id: Optional[str] = None


@dataclass
class BatchScoreResponse:
    run_date: date
    model_version: str
    items: List[ScoredItem]
    feature_snapshot_path: Optional[str] = None
    score_artifacts: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        run_date: date,
        model_version: str,
        feature_snapshot_path: Optional[Any] = None,
        score_artifacts: Optional[Dict[str, Any]] = None,
    ) -> "BatchScoreResponse":
        feature_snapshot = str(feature_snapshot_path) if feature_snapshot_path is not None else None
        serialized_artifacts = {key: str(value) for key, value in (score_artifacts or {}).items()}

        if frame.empty:
            return cls(
                run_date=run_date,
                model_version=model_version,
                items=[],
                feature_snapshot_path=feature_snapshot,
                score_artifacts=serialized_artifacts,
            )

        items: List[ScoredItem] = []
        for _, row in frame.iterrows():
            reason = row.get("_reason_dict")
            if not isinstance(reason, dict):
                reason_field = row.get("reason")
                if isinstance(reason_field, str):
                    try:
                        reason = json.loads(reason_field)
                    except Exception:  # pragma: no cover
                        reason = {}
                elif isinstance(reason_field, dict):
                    reason = reason_field
                else:
                    reason = {}

            rank_in_cat = row.get("rank_in_cat")
            group_rank = row.get("rank_in_group", row.get("group_rank"))
            dt_value = row.get("dt")
            try:
                parsed_dt = _ensure_date(dt_value)
            except Exception:  # pragma: no cover - defensive fallback
                parsed_dt = run_date

            raw_explosive = row.get("explosive_score")
            raw_lgbm = row.get("lgbm_score")
            explosive_score = float(raw_explosive if raw_explosive is not None else raw_lgbm)
            lgbm_score = float(raw_lgbm if raw_lgbm is not None else explosive_score)

            items.append(
                ScoredItem(
                    asin=row.get("asin"),
                    site=row.get("site"),
                    dt=parsed_dt,
                    category=row.get("category"),
                    explosive_score=explosive_score,
                    lgbm_score=lgbm_score,
                    rank_in_cat=int(rank_in_cat) if _is_finite(rank_in_cat) else None,
                    rank_in_group=int(group_rank) if _is_finite(group_rank) else None,
                    reason=reason,
                    group_id=row.get("group_id") if isinstance(row.get("group_id"), str) else None,
                )
            )
        return cls(
            run_date=run_date,
            model_version=model_version,
            items=items,
            feature_snapshot_path=feature_snapshot,
            score_artifacts=serialized_artifacts,
        )


def _is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not pd.isna(value)
    except Exception:
        return True


import json  # noqa: E402  # isort:skip
