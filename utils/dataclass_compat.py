"""Compatibility helpers for ``dataclasses.dataclass``.

The standard library added support for the ``slots`` parameter in Python 3.10.
Some execution environments – or monkeypatched ``dataclasses`` modules used in
tests – may still reject the argument.  This module provides a thin wrapper
that gracefully drops ``slots`` when unsupported so callers can use a single
decorator across runtimes.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, Dict, Tuple

__all__ = ["dataclass"]

_SLOTS_SUPPORTED = sys.version_info >= (3, 10)


def _call_dataclass(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    return dataclasses.dataclass(*args, **kwargs)


def dataclass(*args: Any, **kwargs: Any) -> Any:
    """A ``dataclass`` decorator tolerant of unsupported ``slots`` arguments."""

    if "slots" not in kwargs:
        return _call_dataclass(args, kwargs)

    call_kwargs = dict(kwargs)

    if _SLOTS_SUPPORTED:
        try:
            return _call_dataclass(args, call_kwargs)
        except TypeError as exc:  # pragma: no cover - exercised via tests
            if "slots" not in str(exc):
                raise

    call_kwargs.pop("slots", None)
    return _call_dataclass(args, call_kwargs)
