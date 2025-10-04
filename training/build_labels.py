"""Helpers for preparing label datasets for downstream training."""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Dict, MutableMapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_ID_COLUMNS: Sequence[str] = ("asin", "site", "dt")


def _normalise_for_json(value: object) -> object:
    """Normalise scalar values so they can be serialised with ``json.dumps``."""
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return value


def _serialise_payload(row: pd.Series, columns: Sequence[str]) -> str:
    if not columns:
        return "{}"
    payload: MutableMapping[str, object] = {}
    for column in columns:
        payload[column] = _normalise_for_json(row.get(column))
    return json.dumps(payload, sort_keys=True)


def build_labels(frame: pd.DataFrame, settings: Dict | None = None) -> pd.DataFrame:
    """Prepare a label dataset from a feature frame.

    Parameters
    ----------
    frame:
        DataFrame containing identifiers, label target, feature columns, and any
        supplementary metadata columns.
    settings:
        Optional configuration that can specify:

        ``training.label_column``
            Name of the label/target column in ``frame``. Defaults to ``"label"``.
        ``training.feature_columns``
            Explicit list of feature columns to embed into ``feature_vector``. When
            omitted, the columns are inferred by excluding identifiers, the label,
            and metadata columns.
        ``training.meta_columns``
            Metadata columns that should be preserved (serialised as JSON strings)
            for downstream inspection.
        ``training.id_columns``
            Identifier columns, defaulting to ``("asin", "site", "dt")``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing identifier columns, a normalised ``label`` column,
        and JSON serialised ``feature_vector``/``meta`` columns.
    """

    cfg = (settings or {}).get("training", {})
    id_columns: Sequence[str] = cfg.get("id_columns", DEFAULT_ID_COLUMNS)
    label_column: str = cfg.get("label_column", "label")
    meta_columns: Sequence[str] = cfg.get("meta_columns", ())

    missing_ids = [col for col in id_columns if col not in frame.columns]
    if missing_ids:
        raise KeyError(f"Missing identifier columns: {missing_ids}")
    if label_column not in frame.columns:
        raise KeyError(f"Label column '{label_column}' is missing from frame")

    # Determine which columns make up the feature vector.
    feature_columns: Sequence[str] | None = cfg.get("feature_columns")
    if feature_columns is None:
        excluded = set(id_columns) | {label_column} | set(meta_columns)
        feature_columns = [col for col in frame.columns if col not in excluded]

    result = frame[list(id_columns)].copy()
    result["label"] = frame[label_column]

    feature_payloads = frame.apply(lambda row: _serialise_payload(row, feature_columns), axis=1)
    meta_payloads = frame.apply(lambda row: _serialise_payload(row, meta_columns), axis=1)

    result["feature_vector"] = feature_payloads.astype(str)
    result["meta"] = meta_payloads.astype(str)
    return result


__all__ = ["build_labels"]
