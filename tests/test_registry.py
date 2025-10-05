from __future__ import annotations

import dataclasses

import pytest

from training import registry


def test_dataclass_allows_slots_on_older_interpreters(monkeypatch) -> None:
    original_dataclass = dataclasses.dataclass

    def fake_dataclass(*args, **kwargs):
        if "slots" in kwargs:
            raise TypeError("dataclass() got an unexpected keyword argument 'slots'")
        return original_dataclass(*args, **kwargs)

    monkeypatch.setattr(dataclasses, "dataclass", fake_dataclass)

    @registry.dataclass(slots=True)
    class Example:
        value: int

    instance = Example(42)

    assert dataclasses.is_dataclass(Example)
    assert instance.value == 42


def test_dataclass_handles_no_keyword_arguments_message(monkeypatch) -> None:
    original_dataclass = dataclasses.dataclass

    def fake_dataclass(*args, **kwargs):
        if kwargs:
            raise TypeError("dataclass() takes no keyword arguments")
        return original_dataclass(*args, **kwargs)

    monkeypatch.setattr(dataclasses, "dataclass", fake_dataclass)

    @registry.dataclass(slots=True)
    class Example:
        value: int

    assert dataclasses.is_dataclass(Example)


def test_dataclass_preserves_other_type_errors(monkeypatch) -> None:
    original_dataclass = dataclasses.dataclass

    def fake_dataclass(*args, **kwargs):
        if "slots" in kwargs:
            raise TypeError("slots must be bool")
        return original_dataclass(*args, **kwargs)

    monkeypatch.setattr(dataclasses, "dataclass", fake_dataclass)

    with pytest.raises(TypeError, match="slots must be bool"):
        @registry.dataclass(slots=True)
        class Example:
            value: int
