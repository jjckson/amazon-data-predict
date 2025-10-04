"""Model registry management utilities and CLI commands."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


REGISTRY_ENV = "MODEL_REGISTRY_PATH"
DEFAULT_REGISTRY_PATH = Path(__file__).with_name("model_registry.json")

STAGES = ("dev", "staging", "prod")

EXPECTED_FEATURE_SCHEMA: Dict[str, str] = {
    "asin": "string",
    "site": "string",
    "dt": "datetime",
    "category": "string",
    "bsr_trend_30": "float",
    "est_sales_30": "float",
    "review_vel_14": "float",
    "price_vol_30": "float",
    "listing_quality": "float",
}


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"models": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_registry(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _normalise_schema(schema: Mapping[str, Any]) -> Dict[str, str]:
    return {str(key): str(value).lower() for key, value in schema.items()}


def _extract_signature_features(signature: Mapping[str, Any]) -> Dict[str, Any]:
    if "features" in signature and isinstance(signature["features"], Mapping):
        return dict(signature["features"])
    if "inputs" in signature and isinstance(signature["inputs"], Mapping):
        return dict(signature["inputs"])
    raise ValueError(
        "Model signature must include a mapping of feature names under 'features' or 'inputs'."
    )


def _validate_signature(signature: Mapping[str, Any]) -> Dict[str, str]:
    features = _extract_signature_features(signature)
    normalised = _normalise_schema(features)
    if not normalised:
        raise ValueError("Model signature does not define any features.")
    for name, dtype in normalised.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Feature names in signature must be non-empty strings.")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"Feature '{name}' in signature must define a dtype.")
    return normalised


def _assert_signature_matches_inference(signature: Mapping[str, Any]) -> None:
    normalised = _validate_signature(signature)
    expected = _normalise_schema(EXPECTED_FEATURE_SCHEMA)
    if normalised != expected:
        missing = sorted(set(expected) - set(normalised))
        extra = sorted(set(normalised) - set(expected))
        mismatched = sorted(
            name
            for name in set(expected).intersection(normalised)
            if normalised[name] != expected[name]
        )
        problems = []
        if missing:
            problems.append(f"missing features: {', '.join(missing)}")
        if extra:
            problems.append(f"unexpected features: {', '.join(extra)}")
        if mismatched:
            issues = ", ".join(
                f"{name} (expected {expected[name]}, found {normalised[name]})"
                for name in mismatched
            )
            problems.append(f"type mismatches: {issues}")
        reason = "; ".join(problems) if problems else "schema mismatch"
        raise ValueError(f"Signature does not match inference schema: {reason}")


@dataclass
class ModelVersion:
    """Representation of a registered model version."""

    version: str
    signature: Mapping[str, Any]
    data_span: str
    feature_version: str
    artifact_uri: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "signature": self.signature,
            "data_span": self.data_span,
            "feature_version": self.feature_version,
            "artifact_uri": self.artifact_uri,
            "metrics": dict(self.metrics),
            "registered_at": self.registered_at,
        }


class ModelRegistry:
    """Filesystem-backed registry for managing model lifecycle stages."""

    def __init__(self, path: Optional[Path] = None):
        resolved = path or Path(os.environ.get(REGISTRY_ENV, DEFAULT_REGISTRY_PATH))
        self.path = resolved
        self._state: Dict[str, Any] = _load_registry(self.path)

    def _get_model(self, model_name: str) -> MutableMapping[str, Any]:
        models = self._state.setdefault("models", {})
        return models.setdefault(model_name, {"versions": {}, "stages": {}, "history": []})

    def _get_version(self, model_name: str, version: str) -> Dict[str, Any]:
        model = self._get_model(model_name)
        versions = model["versions"]
        if version not in versions:
            raise KeyError(f"Model '{model_name}' version '{version}' is not registered.")
        return versions[version]

    def _record_history(self, model_name: str, stage: str, version: Optional[str], action: str) -> None:
        model = self._get_model(model_name)
        model.setdefault("history", []).append(
            {
                "stage": stage,
                "version": version,
                "action": action,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def save(self) -> None:
        _save_registry(self.path, self._state)

    def register(
        self,
        model_name: str,
        version: str,
        *,
        signature: Mapping[str, Any],
        data_span: str,
        feature_version: str,
        artifact_uri: str,
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        model = self._get_model(model_name)
        if version in model["versions"]:
            raise ValueError(f"Model '{model_name}' version '{version}' already exists.")
        validated_signature = _validate_signature(signature)
        payload = ModelVersion(
            version=version,
            signature={"features": validated_signature},
            data_span=data_span,
            feature_version=feature_version,
            artifact_uri=artifact_uri,
            metrics=metrics or {},
        )
        model["versions"][version] = payload.to_dict()
        model.setdefault("stages", {})["dev"] = version
        self._record_history(model_name, "dev", version, "register")
        self.save()

    def promote(self, model_name: str, version: str, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Stage '{stage}' is not supported. Choose from {STAGES}.")
        model = self._get_model(model_name)
        version_payload = self._get_version(model_name, version)
        if stage in {"staging", "prod"}:
            _assert_signature_matches_inference(version_payload["signature"])
        model.setdefault("stages", {})[stage] = version
        self._record_history(model_name, stage, version, "promote")
        self.save()

    def demote(self, model_name: str, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Stage '{stage}' is not supported. Choose from {STAGES}.")
        model = self._get_model(model_name)
        stages = model.setdefault("stages", {})
        if stage not in stages:
            raise ValueError(f"Model '{model_name}' has no assignment for stage '{stage}'.")
        previous = stages.pop(stage)
        self._record_history(model_name, stage, previous, "demote")
        self.save()

    def rollback(self, model_name: str, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Stage '{stage}' is not supported. Choose from {STAGES}.")
        model = self._get_model(model_name)
        history = [
            item
            for item in model.get("history", [])
            if item["stage"] == stage and item["action"] == "promote"
        ]
        if len(history) < 2:
            raise ValueError(f"No prior promotions available to roll back stage '{stage}'.")
        current = history[-1]["version"]
        previous = history[-2]["version"]
        model.setdefault("stages", {})[stage] = previous
        self._record_history(model_name, stage, current, "rollback")
        self.save()

    def describe(self, model_name: str) -> Dict[str, Any]:
        return json.loads(json.dumps(self._get_model(model_name)))


def _parse_signature(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.signature and args.signature_file:
        raise ValueError("Provide either --signature or --signature-file, not both.")
    if args.signature:
        return json.loads(args.signature)
    if args.signature_file:
        with Path(args.signature_file).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError("A model signature definition is required for registration.")


def _parse_metrics(metrics: Optional[str]) -> Mapping[str, Any]:
    if not metrics:
        return {}
    loaded = json.loads(metrics)
    if not isinstance(loaded, Mapping):
        raise ValueError("Metrics payload must be a JSON mapping.")
    return loaded


def _add_register_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser("register", help="Register a new model version")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--data-span", required=True)
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--artifact-uri", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--signature")
    parser.add_argument("--signature-file")
    parser.set_defaults(command="register")


def _add_promote_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser("promote", help="Promote a model version to a stage")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.set_defaults(command="promote")


def _add_demote_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser("demote", help="Remove a model version from a stage")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.set_defaults(command="demote")


def _add_rollback_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser("rollback", help="Rollback a stage to the previous version")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.set_defaults(command="rollback")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model registry management")
    subparsers = parser.add_subparsers(dest="command")
    _add_register_parser(subparsers)
    _add_promote_parser(subparsers)
    _add_demote_parser(subparsers)
    _add_rollback_parser(subparsers)
    return parser


def main(args: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    if not parsed.command:
        parser.print_help()
        return

    registry = ModelRegistry()
    if parsed.command == "register":
        signature = _parse_signature(parsed)
        metrics = _parse_metrics(parsed.metrics)
        registry.register(
            parsed.model_name,
            parsed.version,
            signature=signature,
            data_span=parsed.data_span,
            feature_version=parsed.feature_version,
            artifact_uri=parsed.artifact_uri,
            metrics=metrics,
        )
    elif parsed.command == "promote":
        registry.promote(parsed.model_name, parsed.version, parsed.stage)
    elif parsed.command == "demote":
        registry.demote(parsed.model_name, parsed.stage)
    elif parsed.command == "rollback":
        registry.rollback(parsed.model_name, parsed.stage)
    else:
        parser.error(f"Unknown command '{parsed.command}'")


if __name__ == "__main__":
    main()
