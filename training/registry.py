"""Utilities for registering and promoting trained models."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.dataclass_compat import dataclass

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ModelVersion",
    "ModelRegistry",
    "RegistryError",
    "SignatureMismatchError",
    "dataclass",
    "field",
]

field = dataclasses.field

DEFAULT_REGISTRY_FILE = Path("runs/registry.json")
_DEFAULT_STAGE = "None"
_PROMOTION_STAGES = {"Staging", "Production"}
_STAGE_ALIASES = {
    "none": "None",
    "": "None",
    "staging": "Staging",
    "stage": "Staging",
    "testing": "Staging",
    "prod": "Production",
    "production": "Production",
    "archive": "Archived",
    "archived": "Archived",
}
@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Represents a model version stored in the registry."""

    name: str
    version: str
    stage: str
    artifact_uri: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    data_span: Optional[str] = None
    feature_version: Optional[str] = None
    signature: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RegistryError(RuntimeError):
    """Base error for registry operations."""


class SignatureMismatchError(RegistryError):
    """Raised when model signatures are not compatible for promotion."""


class _LocalRegistryBackend:
    """Simple JSON-backed registry for local development."""

    def __init__(self, registry_path: Path) -> None:
        self._path = registry_path

    # ------------------------------------------------------------------
    # JSON persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        if not self._path.exists():
            return {"models": {}}
        return json.loads(self._path.read_text())

    def _save(self, payload: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    # ------------------------------------------------------------------
    # Registry operations
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        artifact_uri: str,
        *,
        signature: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        data_span: Optional[str] = None,
        feature_version: Optional[str] = None,
        stage: Optional[str] = None,
        run_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ModelVersion:
        payload = self._load()
        models = payload.setdefault("models", {})
        entries: List[Dict[str, Any]] = models.setdefault(name, [])

        version_number = len(entries) + 1
        stage_value = _normalise_stage(stage) if stage else _DEFAULT_STAGE
        now = _utc_now()
        feature_contract = _extract_feature_contract(signature)

        record = {
            "version": version_number,
            "stage": stage_value,
            "artifact_uri": artifact_uri,
            "metrics": metrics or {},
            "data_span": data_span,
            "feature_version": feature_version,
            "signature": signature,
            "run_id": run_id,
            "description": description,
            "feature_contract": feature_contract,
            "created_at": now,
            "updated_at": now,
        }

        entries.append(record)
        self._save(payload)
        return _to_model_version(name, record)

    def list_versions(self, name: str) -> List[ModelVersion]:
        payload = self._load()
        entries: Iterable[Dict[str, Any]] = payload.get("models", {}).get(name, [])
        sorted_entries = sorted(entries, key=lambda item: int(item["version"]))
        return [_to_model_version(name, item) for item in sorted_entries]

    def promote(self, name: str, version: str, stage: str) -> ModelVersion:
        return self._update_stage(name, version, stage)

    def demote(self, name: str, version: str, stage: str = "Archived") -> ModelVersion:
        return self._update_stage(name, version, stage)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_stage(self, name: str, version: str, stage: str) -> ModelVersion:
        payload = self._load()
        models = payload.get("models", {})
        entries: List[Dict[str, Any]] = models.get(name, [])
        if not entries:
            raise RegistryError(f"Model '{name}' not found")

        target = _find_version(entries, version)
        if target is None:
            raise RegistryError(f"Version {version} of '{name}' not found")

        target_stage = _normalise_stage(stage)
        if target_stage in _PROMOTION_STAGES:
            _require_signature_contract(target)
            active = _find_stage(entries, target_stage, exclude=target)
            if active is not None:
                if active.get("feature_contract") != target.get("feature_contract"):
                    raise SignatureMismatchError(
                        "Signature mismatch with active version in stage"
                    )
                active["stage"] = "Archived"
                active["updated_at"] = _utc_now()

        target["stage"] = target_stage
        target["updated_at"] = _utc_now()
        self._save(payload)
        return _to_model_version(name, target)


class _MLflowRegistryBackend:
    """Interface to MLflow's Model Registry when available."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def register(
        self,
        name: str,
        artifact_uri: str,
        *,
        signature: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        data_span: Optional[str] = None,
        feature_version: Optional[str] = None,
        stage: Optional[str] = None,
        run_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ModelVersion:
        import mlflow

        _ensure_registered_model(self._client, name)
        mv = self._client.create_model_version(
            name=name,
            source=artifact_uri,
            run_id=run_id,
            tags=_build_tags(
                metrics=metrics,
                data_span=data_span,
                feature_version=feature_version,
                signature=signature,
            ),
            description=description,
        )

        stage_value = _normalise_stage(stage) if stage else mv.current_stage or _DEFAULT_STAGE
        if stage_value and stage_value != mv.current_stage:
            self._client.transition_model_version_stage(
                name=name,
                version=mv.version,
                stage=stage_value,
            )

        return ModelVersion(
            name=name,
            version=str(mv.version),
            stage=stage_value,
            artifact_uri=artifact_uri,
            metrics=metrics,
            data_span=data_span,
            feature_version=feature_version,
            signature=signature,
            run_id=run_id,
            description=description,
        )

    def list_versions(self, name: str) -> List[ModelVersion]:
        versions = self._client.search_model_versions(f"name='{name}'")
        results: List[ModelVersion] = []
        for mv in versions:
            signature = _parse_signature_from_tags(mv.tags)
            metrics = _parse_metrics_from_tags(mv.tags)
            results.append(
                ModelVersion(
                    name=mv.name,
                    version=str(mv.version),
                    stage=mv.current_stage or _DEFAULT_STAGE,
                    artifact_uri=mv.source,
                    metrics=metrics,
                    data_span=mv.tags.get("data_span"),
                    feature_version=mv.tags.get("feature_version"),
                    signature=signature,
                    run_id=mv.run_id,
                    description=mv.description,
                )
            )
        return sorted(results, key=lambda item: int(item.version))

    def promote(self, name: str, version: str, stage: str) -> ModelVersion:
        model_version = self._transition(name, version, stage)
        return model_version

    def demote(self, name: str, version: str, stage: str = "Archived") -> ModelVersion:
        return self._transition(name, version, stage)

    def _transition(self, name: str, version: str, stage: str) -> ModelVersion:
        target_stage = _normalise_stage(stage)
        mv = self._client.get_model_version(name=name, version=version)
        signature = _parse_signature_from_tags(mv.tags)
        if target_stage in _PROMOTION_STAGES:
            _require_signature_contract(
                {
                    "signature": signature,
                    "feature_contract": _extract_feature_contract(signature),
                }
            )
            active = _find_active_mlflow_version(self._client, name, target_stage, version)
            if active is not None and _extract_feature_contract(signature) != _extract_feature_contract(
                _parse_signature_from_tags(active.tags)
            ):
                raise SignatureMismatchError("Signature mismatch with active version in stage")

        self._client.transition_model_version_stage(
            name=name,
            version=version,
            stage=target_stage,
        )
        updated = self._client.get_model_version(name=name, version=version)
        return ModelVersion(
            name=updated.name,
            version=str(updated.version),
            stage=updated.current_stage or _DEFAULT_STAGE,
            artifact_uri=updated.source,
            metrics=_parse_metrics_from_tags(updated.tags),
            data_span=updated.tags.get("data_span"),
            feature_version=updated.tags.get("feature_version"),
            signature=_parse_signature_from_tags(updated.tags),
            run_id=updated.run_id,
            description=updated.description,
        )


class ModelRegistry:
    """Facade that delegates to MLflow if available, otherwise a local file."""

    def __init__(
        self,
        *,
        tracking_uri: Optional[str] = None,
        registry_path: Optional[Path] = None,
        prefer_local: bool = False,
    ) -> None:
        backend = None
        if not prefer_local:
            backend = _maybe_create_mlflow_backend(tracking_uri)

        if backend is None:
            backend = _LocalRegistryBackend(registry_path or DEFAULT_REGISTRY_FILE)

        self._backend = backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        artifact_uri: str,
        *,
        signature: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        data_span: Optional[str] = None,
        feature_version: Optional[str] = None,
        stage: Optional[str] = None,
        run_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ModelVersion:
        return self._backend.register(
            name,
            artifact_uri,
            signature=signature,
            metrics=metrics,
            data_span=data_span,
            feature_version=feature_version,
            stage=stage,
            run_id=run_id,
            description=description,
        )

    def promote(self, name: str, version: str, stage: str) -> ModelVersion:
        return self._backend.promote(name, version, stage)

    def demote(self, name: str, version: str, stage: str = "Archived") -> ModelVersion:
        return self._backend.demote(name, version, stage)

    def list_versions(self, name: str) -> List[ModelVersion]:
        return self._backend.list_versions(name)


# ----------------------------------------------------------------------
# Helper functions shared by both backends
# ----------------------------------------------------------------------


def _maybe_create_mlflow_backend(tracking_uri: Optional[str]) -> Optional[_MLflowRegistryBackend]:
    try:
        import mlflow
        from mlflow import tracking
    except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency
        LOGGER.info("MLflow is not installed; using local registry backend")
        return None

    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif mlflow.get_tracking_uri() is None:
            env_uri = os.getenv("MLFLOW_TRACKING_URI")
            if env_uri:
                mlflow.set_tracking_uri(env_uri)
        client = tracking.MlflowClient()
        client.list_registered_models()
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.warning("Falling back to local registry backend: %s", exc)
        return None

    return _MLflowRegistryBackend(client)


def _normalise_stage(stage: Optional[str]) -> str:
    if not stage:
        return _DEFAULT_STAGE
    alias = _STAGE_ALIASES.get(stage.lower())
    if alias:
        return alias
    title_case = stage.title()
    if title_case in {"None", "Staging", "Production", "Archived"}:
        return title_case
    raise RegistryError(f"Unknown stage '{stage}'")


def _extract_feature_contract(signature: Optional[Dict[str, Any]]) -> List[str]:
    if not signature:
        return []
    inputs = signature.get("inputs")
    if not isinstance(inputs, list):
        raise RegistryError("Signature must contain an 'inputs' list")
    contract: List[str] = []
    for item in inputs:
        if not isinstance(item, dict) or "name" not in item:
            raise RegistryError("Invalid signature payload; expected input name")
        contract.append(str(item["name"]))
    return contract


def _require_signature_contract(record: Dict[str, Any]) -> None:
    signature = record.get("signature")
    if not signature:
        raise RegistryError("Cannot promote a model without a recorded signature")
    contract = record.get("feature_contract")
    if not contract:
        raise RegistryError("Model signature does not contain input features")


def _find_version(entries: Iterable[Dict[str, Any]], version: str) -> Optional[Dict[str, Any]]:
    for entry in entries:
        if str(entry.get("version")) == str(version):
            return entry
    return None


def _find_stage(
    entries: Iterable[Dict[str, Any]],
    stage: str,
    *,
    exclude: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    for entry in entries:
        if exclude is not None and entry is exclude:
            continue
        if entry.get("stage") == stage:
            return entry
    return None


def _to_model_version(name: str, record: Dict[str, Any]) -> ModelVersion:
    return ModelVersion(
        name=name,
        version=str(record.get("version")),
        stage=record.get("stage", _DEFAULT_STAGE),
        artifact_uri=record.get("artifact_uri"),
        metrics=record.get("metrics"),
        data_span=record.get("data_span"),
        feature_version=record.get("feature_version"),
        signature=record.get("signature"),
        run_id=record.get("run_id"),
        description=record.get("description"),
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
    )


def _ensure_registered_model(client: Any, name: str) -> None:
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)


def _build_tags(
    *,
    metrics: Optional[Dict[str, float]],
    data_span: Optional[str],
    feature_version: Optional[str],
    signature: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    if metrics:
        for key, value in metrics.items():
            tags[f"metric_{key}"] = str(value)
    if data_span:
        tags["data_span"] = data_span
    if feature_version:
        tags["feature_version"] = feature_version
    if signature:
        tags["signature"] = json.dumps(signature)
    return tags


def _parse_signature_from_tags(tags: Dict[str, str]) -> Optional[Dict[str, Any]]:
    payload = tags.get("signature")
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RegistryError("Stored signature payload is not valid JSON") from exc


def _parse_metrics_from_tags(tags: Dict[str, str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    prefix = "metric_"
    for key, value in tags.items():
        if key.startswith(prefix):
            try:
                metrics[key[len(prefix) :]] = float(value)
            except ValueError:
                LOGGER.debug("Ignoring non-numeric metric tag %s=%s", key, value)
    return metrics


def _find_active_mlflow_version(client: Any, name: str, stage: str, version: str) -> Optional[Any]:
    for mv in client.search_model_versions(f"name='{name}'"):
        if mv.current_stage == stage and str(mv.version) != str(version):
            return mv
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model registry CLI")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI env var.",
    )
    parser.add_argument(
        "--registry-file",
        type=Path,
        default=DEFAULT_REGISTRY_FILE,
        help="Path to the local registry file when MLflow is unavailable.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force the use of the local JSON registry instead of MLflow.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register", help="Register a new model version")
    register_parser.add_argument("--name", required=True)
    register_parser.add_argument("--artifact-uri", required=True)
    register_parser.add_argument("--signature", help="Path or JSON payload with the model signature")
    register_parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metrics in key=value format (can be supplied multiple times)",
    )
    register_parser.add_argument("--data-span")
    register_parser.add_argument("--feature-version")
    register_parser.add_argument("--stage")
    register_parser.add_argument("--run-id")
    register_parser.add_argument("--description")

    promote_parser = subparsers.add_parser("promote", help="Promote a model to a stage")
    promote_parser.add_argument("--name", required=True)
    promote_parser.add_argument("--version", required=True)
    promote_parser.add_argument("--to", required=True, dest="stage")

    demote_parser = subparsers.add_parser("demote", help="Demote a model version")
    demote_parser.add_argument("--name", required=True)
    demote_parser.add_argument("--version", required=True)
    demote_parser.add_argument("--to", default="Archived", dest="stage")

    list_parser = subparsers.add_parser("list", help="List versions of a model")
    list_parser.add_argument("--name", required=True)

    return parser


def _create_registry_from_args(args: argparse.Namespace) -> ModelRegistry:
    return ModelRegistry(
        tracking_uri=args.tracking_uri,
        registry_path=args.registry_file,
        prefer_local=args.local,
    )


def _handle_register(args: argparse.Namespace) -> None:
    registry = _create_registry_from_args(args)
    signature = _load_signature_payload(args.signature)
    metrics = _parse_metrics_cli(args.metric)
    version = registry.register(
        args.name,
        args.artifact_uri,
        signature=signature,
        metrics=metrics,
        data_span=args.data_span,
        feature_version=args.feature_version,
        stage=args.stage,
        run_id=args.run_id,
        description=args.description,
    )
    print(f"Registered {version.name} v{version.version} in stage {version.stage}")


def _handle_promote(args: argparse.Namespace) -> None:
    registry = _create_registry_from_args(args)
    version = registry.promote(args.name, args.version, args.stage)
    print(f"Promoted {version.name} v{version.version} to {version.stage}")


def _handle_demote(args: argparse.Namespace) -> None:
    registry = _create_registry_from_args(args)
    version = registry.demote(args.name, args.version, args.stage)
    print(f"Demoted {version.name} v{version.version} to {version.stage}")


def _handle_list(args: argparse.Namespace) -> None:
    registry = _create_registry_from_args(args)
    versions = registry.list_versions(args.name)
    if not versions:
        print("No versions found")
        return
    for version in versions:
        metrics = ", ".join(f"{k}={v}" for k, v in (version.metrics or {}).items())
        print(
            f"v{version.version}\t{version.stage}\t{version.artifact_uri}\t"
            f"{metrics}\tdata_span={version.data_span}\tfeature_version={version.feature_version}"
        )


def _load_signature_payload(payload: Optional[str]) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    candidate = Path(payload)
    if candidate.exists():
        return json.loads(candidate.read_text())
    return json.loads(payload)


def _parse_metrics_cli(values: Iterable[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise RegistryError(f"Metric '{item}' is not in key=value format")
        key, value = item.split("=", 1)
        metrics[key] = float(value)
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "register": _handle_register,
        "promote": _handle_promote,
        "demote": _handle_demote,
        "list": _handle_list,
    }

    handler = handlers.get(args.command)
    assert handler is not None  # for mypy
    handler(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
