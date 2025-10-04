from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from training.registry import EXPECTED_FEATURE_SCHEMA, ModelRegistry


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / "registry.json"


@pytest.fixture
def canonical_signature() -> Dict[str, Dict[str, str]]:
    return {"features": EXPECTED_FEATURE_SCHEMA}


def test_registration_and_stage_transitions(
    registry_path: Path, canonical_signature: Dict[str, Dict[str, str]]
) -> None:
    registry = ModelRegistry(registry_path)

    registry.register(
        "baseline",
        "1",
        signature=canonical_signature,
        data_span="2023-01-01/2023-01-31",
        feature_version="v1",
        artifact_uri="s3://models/baseline/1",
        metrics={"roc_auc": 0.91},
    )

    model_state = registry.describe("baseline")
    assert model_state["stages"]["dev"] == "1"
    assert model_state["versions"]["1"]["metrics"]["roc_auc"] == 0.91
    assert model_state["versions"]["1"]["feature_version"] == "v1"

    registry.promote("baseline", "1", "staging")
    registry.promote("baseline", "1", "prod")
    model_state = registry.describe("baseline")
    assert model_state["stages"]["staging"] == "1"
    assert model_state["stages"]["prod"] == "1"

    registry.demote("baseline", "prod")
    model_state = registry.describe("baseline")
    assert "prod" not in model_state["stages"]

    registry.register(
        "baseline",
        "2",
        signature=canonical_signature,
        data_span="2023-02-01/2023-02-28",
        feature_version="v2",
        artifact_uri="s3://models/baseline/2",
        metrics={"roc_auc": 0.94},
    )
    registry.promote("baseline", "2", "staging")
    model_state = registry.describe("baseline")
    assert model_state["stages"]["staging"] == "2"

    registry.rollback("baseline", "staging")
    model_state = registry.describe("baseline")
    assert model_state["stages"]["staging"] == "1"


def test_promotion_requires_inference_signature(registry_path: Path) -> None:
    registry = ModelRegistry(registry_path)
    uppercase_signature = {
        "features": {key: value.upper() for key, value in EXPECTED_FEATURE_SCHEMA.items()}
    }

    registry.register(
        "baseline",
        "1",
        signature=uppercase_signature,
        data_span="2023-03-01/2023-03-31",
        feature_version="v1",
        artifact_uri="s3://models/baseline/1",
        metrics={"accuracy": 0.8},
    )

    registry.promote("baseline", "1", "staging")
    registry.promote("baseline", "1", "prod")
    model_state = registry.describe("baseline")
    assert model_state["stages"]["prod"] == "1"


def test_promotion_rejects_signature_mismatch(registry_path: Path) -> None:
    registry = ModelRegistry(registry_path)
    mismatched_schema = dict(EXPECTED_FEATURE_SCHEMA)
    mismatched_schema.pop("listing_quality")
    mismatched_schema["unexpected"] = "float"

    registry.register(
        "baseline",
        "1",
        signature={"features": mismatched_schema},
        data_span="2023-04-01/2023-04-30",
        feature_version="v1",
        artifact_uri="s3://models/baseline/1",
        metrics={"precision": 0.75},
    )

    with pytest.raises(ValueError):
        registry.promote("baseline", "1", "staging")

    state = registry.describe("baseline")
    assert "staging" not in state["stages"]
