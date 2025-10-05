from __future__ import annotations

import dataclasses

from training import registry


def test_dataclass_wrapper_ignores_slots_for_legacy_interpreters(monkeypatch) -> None:
    real_dataclass = dataclasses.dataclass
    calls = []

    def fake_dataclass(*args, **kwargs):
        calls.append(kwargs)
        if "slots" in kwargs:
            raise TypeError("dataclass() got an unexpected keyword argument 'slots'")
        return real_dataclass(*args, **kwargs)

    monkeypatch.setattr(dataclasses, "dataclass", fake_dataclass)

    @registry.dataclass(slots=True)
    class Dummy:
        value: int

    instance = Dummy(5)
    assert instance.value == 5
    assert dataclasses.is_dataclass(Dummy)
    assert calls[-1] == {}
