"""Utility helpers for logging experiment runs.

The project prefers MLflow for tracking but the dependency is optional in the
test environment. ``RunLogger`` therefore attempts to initialise MLflow and
falls back to writing structured files under a ``runs/`` directory when MLflow
is not available. The interface mimics the subset of the MLflow API used by the
training scripts so that the calling code does not have to branch on the
tracking backend being available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


LOGGER = logging.getLogger(__name__)


def _serialise_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=str)


@dataclass
class _FileRunStore:
    """Persist run metadata to the filesystem when MLflow is unavailable."""

    base_dir: Path
    params: MutableMapping[str, Any] = field(default_factory=dict)
    metrics: list[Mapping[str, Any]] = field(default_factory=list)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.params.update(params)
        self._write_file("params.json", self.params)

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int]) -> None:
        entry = {"step": step, **metrics}
        self.metrics.append(entry)
        self._write_file("metrics.json", self.metrics)

    def log_artifact(self, path: Path) -> None:
        destination = self.base_dir / path.name
        if path.resolve() == destination.resolve():
            return
        if path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)

    def log_text(self, text: str, artifact_file: str) -> None:
        target_path = self.base_dir / artifact_file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(text, encoding="utf-8")

    def _write_file(self, filename: str, data: Any) -> None:
        target_path = self.base_dir / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        content = _serialise_json(data if isinstance(data, Mapping) else data)
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            tmp.write(content)
            temp_path = Path(tmp.name)
        temp_path.replace(target_path)


class RunLogger:
    """Context manager mirroring a subset of the MLflow run interface."""

    def __init__(
        self,
        experiment: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        output_dir: Path | str = Path("runs"),
    ) -> None:
        self.experiment = experiment
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.output_dir = Path(output_dir)
        self._mlflow_run = None
        self._mlflow = None
        self._file_store: Optional[_FileRunStore] = None

    def __enter__(self) -> "RunLogger":
        try:
            import mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment)
            self._mlflow = mlflow
            self._mlflow_run = mlflow.start_run(run_name=self.run_name)
            LOGGER.info("Started MLflow run: %s", self._mlflow_run.info.run_id)
        except Exception as exc:  # pragma: no cover - fallback path
            LOGGER.info("Falling back to filesystem run logging: %s", exc)
            run_name = self.run_name or "run"
            run_dir = self.output_dir / self.experiment / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            self._file_store = _FileRunStore(run_dir)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._mlflow is not None:
            self._mlflow.end_run(status="FAILED" if exc else "FINISHED")

    # ------------------------------------------------------------------
    # Public logging helpers
    # ------------------------------------------------------------------
    def log_params(self, params: Mapping[str, Any]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params(params)
        elif self._file_store is not None:
            self._file_store.log_params(params)

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)
        elif self._file_store is not None:
            self._file_store.log_metrics(metrics, step)

    def log_artifact(self, path: Path) -> None:
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path))
        elif self._file_store is not None:
            self._file_store.log_artifact(path)

    def log_text(self, text: str, artifact_file: str) -> None:
        if self._mlflow is not None:
            self._mlflow.log_text(text, artifact_file)
        elif self._file_store is not None:
            self._file_store.log_text(text, artifact_file)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if self._mlflow is not None:
            self._mlflow.set_tags(tags)
        elif self._file_store is not None:
            self._file_store.log_params({f"tag.{key}": value for key, value in tags.items()})

