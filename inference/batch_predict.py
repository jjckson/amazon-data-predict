"""Batch scoring utilities for daily inference jobs."""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from inference.schemas import BatchScoreResponse
from utils.config import load_settings
from utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - defer import failures to runtime
    lgb = None


def _load_rank_booster(model_path: Path):
    """Load a LightGBM booster from disk."""

    if lgb is None:  # pragma: no cover - depends on optional dependency
        raise ImportError("LightGBM is required for ranking inference")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing ranking model at {model_path}")

    return lgb.Booster(model_file=str(model_path))


@lru_cache(maxsize=4)
def _get_cached_booster(model_path: str):
    """Cached loader to avoid re-reading the same model multiple times."""

    path = Path(model_path)
    return _load_rank_booster(path)


@dataclass
class ViewSpec:
    """Configuration for a SQL view export."""

    name: str
    path: Path
    artifact: str = "scores"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.path / "daily"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def sql_path(self) -> Path:
        return self.path / f"{self.name}.sql"

    def data_path_for(self, run_date: date) -> Path:
        return self.data_dir / f"{run_date.isoformat()}.csv"

    def persist(self, frame: pd.DataFrame, run_date: date) -> Path:
        path = self.data_path_for(run_date)
        frame.to_csv(path, index=False)
        return path

    def render_view(self, target: Path) -> str:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        sql = textwrap.dedent(
            f"""
            -- Auto-generated on {timestamp}
            CREATE OR REPLACE VIEW {self.name} AS
            SELECT *
            FROM read_csv_auto('{target.as_posix()}');
            """
        ).strip()
        return sql + "\n"

    def write_view(self, target: Path) -> Path:
        sql_content = self.render_view(target)
        self.sql_path.write_text(sql_content)
        return self.sql_path


class FeatureRepository:
    """File-system backed repository for feature snapshots."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _iter_snapshots(self) -> Iterable[Tuple[date, Path]]:
        for path in sorted(self.base_path.glob("*")):
            if not path.is_file():
                continue
            stem = path.stem
            try:
                snap_date = date.fromisoformat(stem)
            except ValueError:
                continue
            yield snap_date, path

    def load_latest(self, as_of: Optional[date] = None) -> Tuple[pd.DataFrame, Optional[date], Optional[Path]]:
        candidates: List[Tuple[date, Path]] = list(self._iter_snapshots())
        if not candidates:
            return pd.DataFrame(), None, None

        if as_of is not None:
            candidates = [item for item in candidates if item[0] <= as_of]
            if not candidates:
                return pd.DataFrame(), None, None

        snap_date, path = max(candidates, key=lambda item: item[0])
        frame = self._read_snapshot(path)
        return frame, snap_date, path

    def _read_snapshot(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".csv":
            return pd.read_csv(path, parse_dates=["dt"])
        if path.suffix == ".json":
            return pd.read_json(path)
        raise ValueError(f"Unsupported feature snapshot format: {path.suffix}")

    def persist(self, frame: pd.DataFrame, run_date: date) -> Path:
        frame_to_write = frame.copy()
        if "dt" in frame_to_write.columns:
            frame_to_write["dt"] = pd.to_datetime(frame_to_write["dt"]).dt.strftime("%Y-%m-%d")
        path = self.base_path / f"{run_date.isoformat()}.csv"
        frame_to_write.to_csv(path, index=False)
        return path


def _compute_group_id(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    safe_frame = frame.copy()
    for column in columns:
        if column not in safe_frame.columns:
            safe_frame[column] = "UNKNOWN"
    composed = (
        safe_frame[columns]
        .fillna("UNKNOWN")
        .astype(str)
        .agg("|".join, axis=1)
    )
    return composed


def _shap_top_contributors(values: pd.Series, limit: int = 5) -> Dict[str, float]:
    ordered = sorted(
        ((name, float(values[name])) for name in values.index),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top_items = ordered[:limit]
    return {name: score for name, score in top_items}


def compute_scores(features: pd.DataFrame, settings: Dict, rank_min: int) -> pd.DataFrame:
    if features.empty:
        logger.info("Received empty feature frame; skipping scoring")
        return features

    scoring_config = settings.get("scoring", {})
    model_path = Path(scoring_config.get("model_path", "artifacts/rank/model.txt"))
    price_band_column = scoring_config.get("price_band_column", "price_band")
    identifier_columns: Sequence[str] = scoring_config.get(
        "identifier_columns",
        ("asin", "site", "dt", "category", price_band_column),
    )

    try:
        booster = _get_cached_booster(str(model_path))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Unable to load ranking model at %s: %s", model_path, exc)
        return pd.DataFrame()

    feature_columns: Sequence[str] = scoring_config.get("feature_columns") or booster.feature_name()
    if not feature_columns:
        logger.warning("Model does not expose feature names; cannot score")
        return pd.DataFrame()

    missing_features = [column for column in feature_columns if column not in features.columns]
    if missing_features:
        logger.warning("Feature frame missing required columns: %s", ", ".join(sorted(missing_features)))
        return pd.DataFrame()

    missing_identifiers = [column for column in identifier_columns if column not in features.columns]
    if missing_identifiers:
        logger.warning(
            "Feature frame missing identifier columns: %s", ", ".join(sorted(missing_identifiers))
        )
        return pd.DataFrame()

    feature_matrix = features[feature_columns].copy()

    best_iteration = getattr(booster, "best_iteration", None) or None
    try:
        predictions = booster.predict(feature_matrix, num_iteration=best_iteration)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.error("Model prediction failed: %s", exc)
        return pd.DataFrame()

    scored = features.copy()
    scored["lgbm_score"] = predictions
    scored["explosive_score"] = scored["lgbm_score"]

    shap_payload: Optional[pd.DataFrame]
    try:
        shap_values = booster.predict(
            feature_matrix,
            num_iteration=best_iteration,
            pred_contrib=True,
        )
        shap_columns = list(feature_columns) + ["bias"]
        shap_payload = pd.DataFrame(shap_values, columns=shap_columns, index=feature_matrix.index)
    except Exception as exc:  # pragma: no cover - shap optional
        logger.warning("Unable to compute SHAP contributions: %s", exc)
        shap_payload = None

    if shap_payload is not None:
        shap_frame = shap_payload.drop(columns=["bias"], errors="ignore")
        shap_series = shap_frame.apply(_shap_top_contributors, axis=1)
    else:
        shap_series = pd.Series([{} for _ in range(len(scored))], index=scored.index)

    scored["_reason_dict"] = shap_series
    scored["reason"] = scored["_reason_dict"].apply(lambda value: json.dumps(value))

    if "dt" in scored.columns:
        scored["dt"] = pd.to_datetime(scored["dt"])  # type: ignore[assignment]

    group_columns: Sequence[str] = scoring_config.get(
        "group_by_columns",
        ("site", "category", price_band_column),
    )
    scored["group_id"] = _compute_group_id(scored, group_columns)

    scored["rank_in_group"] = (
        scored.groupby("group_id", dropna=False)["lgbm_score"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    scored["group_rank"] = scored["rank_in_group"]

    scored["rank_in_cat"] = (
        scored.groupby(["site", "category"], dropna=False)["lgbm_score"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    scored.loc[scored["rank_in_cat"] > rank_min, "rank_in_cat"] = pd.NA

    scored["model_version"] = scoring_config.get("model_version", "rank-lgbm")
    return scored


class BatchPredictor:
    """Coordinates loading features, scoring, and exporting artifacts."""

    def __init__(
        self,
        feature_repo: FeatureRepository,
        score_views: List[ViewSpec],
        settings: Dict,
        feature_view: Optional[ViewSpec] = None,
        model_version: str = "baseline",
    ) -> None:
        self.feature_repo = feature_repo
        self.score_views = score_views
        self.feature_view = feature_view
        self.settings = settings
        self.model_version = model_version
        self.rank_min = settings.get("scoring", {}).get("rank_min", 20)

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Dict] = None,
        project_root: Path = Path("."),
    ) -> "BatchPredictor":
        settings = settings or load_settings()
        views_config = settings.get("exports", {}).get("views", [])
        view_specs: List[ViewSpec] = []
        feature_view: Optional[ViewSpec] = None
        for raw in views_config:
            name = raw["name"]
            artifact = raw.get("artifact")
            if artifact is None:
                artifact = "features" if "feature" in name.lower() else "scores"
            spec = ViewSpec(name=name, path=project_root / raw["path"], artifact=artifact)
            if spec.artifact == "features":
                feature_view = spec
            else:
                view_specs.append(spec)
        if feature_view is None:
            feature_repo = FeatureRepository(project_root / "exports" / "features" / "daily")
        else:
            feature_repo = FeatureRepository(feature_view.data_dir)
        return cls(
            feature_repo=feature_repo,
            score_views=view_specs,
            settings=settings,
            feature_view=feature_view,
            model_version=settings.get("scoring", {}).get("model_version", "rank-lgbm"),
        )

    def run(self, as_of: Optional[date] = None) -> BatchScoreResponse:
        features, feature_date, _ = self.feature_repo.load_latest(as_of=as_of)
        if features.empty or feature_date is None:
            logger.warning("No feature snapshots found for %s", as_of or "latest")
            run_date = as_of or date.today()
            return BatchScoreResponse(
                run_date=run_date,
                model_version=self.model_version,
                items=[],
                feature_snapshot_path=None,
                score_artifacts={},
            )

        logger.info("Scoring %s rows for %s", len(features), feature_date.isoformat())
        scored = compute_scores(features, self.settings, rank_min=self.rank_min)

        feature_path = self.feature_repo.persist(features, feature_date)
        artifact_paths: Dict[str, str] = {}
        if self.feature_view is not None:
            self.feature_view.persist(features, feature_date)
            self.feature_view.write_view(feature_path)

        if not scored.empty:
            write_frame = scored.drop(columns=["_reason_dict"], errors="ignore").copy()
            if "dt" in write_frame.columns:
                write_frame["dt"] = pd.to_datetime(write_frame["dt"]).dt.strftime("%Y-%m-%d")
            if "model_version" not in write_frame.columns:
                write_frame["model_version"] = self.model_version
            for view in self.score_views:
                target = view.persist(write_frame, feature_date)
                view.write_view(target)
                artifact_paths[view.name] = str(target)
        else:
            logger.info("Scored dataframe empty for %s", feature_date.isoformat())

        return BatchScoreResponse.from_dataframe(
            scored,
            run_date=feature_date,
            model_version=self.model_version,
            feature_snapshot_path=feature_path,
            score_artifacts=artifact_paths,
        )


def main() -> None:
    settings = load_settings()
    predictor = BatchPredictor.from_settings(settings=settings, project_root=Path("."))
    result = predictor.run()
    logger.info(
        "Completed batch scoring for %s with %d items", result.run_date.isoformat(), len(result.items)
    )


if __name__ == "__main__":
    main()
