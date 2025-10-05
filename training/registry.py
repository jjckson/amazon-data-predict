"""Utilities for registering trained models and metadata."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, TypeVar, overload

__all__ = ["dataclass", "ModelVersion"]

_T = TypeVar("_T")


@overload
def dataclass(_cls: type[_T], /, **kwargs: Any) -> type[_T]:
    ...


@overload
def dataclass(
    _cls: None = ..., /, **kwargs: Any
) -> Callable[[type[_T]], type[_T]]:  # pragma: no cover - typing overload
    ...


def dataclass(
    _cls: type[_T] | None = None, /, **kwargs: Any
) -> type[_T] | Callable[[type[_T]], type[_T]]:
    """Wrap :func:`dataclasses.dataclass` with backwards compatible ``slots`` support.

    The ``slots`` keyword argument was only introduced in the Python 3.10
    implementation of :func:`dataclasses.dataclass`.  Some of our callers rely on
    the ability to request ``slots=True`` regardless of the interpreter version,
    so this helper swallows the argument on older versions where the standard
    decorator does not understand it.
    """

    sentinel = object()
    kwargs = dict(kwargs)
    slots_value = kwargs.pop("slots", sentinel)

    def _apply(cls: type[_T]) -> type[_T]:
        if slots_value is sentinel:
            return dataclasses.dataclass(cls, **kwargs)

        try:
            return dataclasses.dataclass(cls, slots=slots_value, **kwargs)
        except TypeError as error:
            # On Python < 3.10, ``dataclasses.dataclass`` raises a ``TypeError`` for
            # the unexpected ``slots`` keyword argument.  Only swallow that
            # specific failure; any other TypeError should propagate so the caller
            # is aware of genuine configuration issues.
            message = (error.args[0] if error.args else str(error)).lower()
            accepts_no_keywords = "takes no keyword arguments" in message
            unexpected_slots = "slots" in message and "unexpected keyword argument" in message
            if not (accepts_no_keywords or unexpected_slots):
                raise
            return dataclasses.dataclass(cls, **kwargs)

    if _cls is None:
        return _apply

    return _apply(_cls)


@dataclass
class ModelVersion:
    """Metadata describing a registered model version."""

    name: str
    version: str
