"""Utilities for registering trained models."""
from __future__ import annotations

import dataclasses
import sys
from typing import Any, Dict, Optional

__all__ = ["ModelVersion", "dataclass", "field"]

field = dataclasses.field

_SLOTS_SUPPORTED = sys.version_info >= (3, 10)


def _call_dataclass(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    return dataclasses.dataclass(*args, **kwargs)


def dataclass(*args: Any, **kwargs: Any) -> Any:
    """A thin wrapper around :func:`dataclasses.dataclass`.

    The wrapper removes the ``slots`` keyword when the standard library
    implementation does not support it (Python < 3.10). It also tolerates
    monkeypatched versions that raise ``TypeError`` when ``slots`` is provided.
    """

    if "slots" not in kwargs:
        return _call_dataclass(args, kwargs)

    call_kwargs = dict(kwargs)

    if _SLOTS_SUPPORTED:
        try:
            return _call_dataclass(args, call_kwargs)
        except TypeError as exc:
            if "slots" not in str(exc):
                raise

    call_kwargs.pop("slots", None)
    return _call_dataclass(args, call_kwargs)


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Represents a model version stored in the registry."""

    name: str
    version: str
    artifact_path: Optional[str] = None
