from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from training import registry


def _signature(features: list[str]) -> dict[str, object]:
    return {
        "inputs": [{"name": name, "type": "double"} for name in features],
        "outputs": [{"name": "score", "type": "double"}],
    }


def _local_registry(tmp_path: Path) -> registry.ModelRegistry:
    return registry.ModelRegistry(registry_path=tmp_path / "registry.json", prefer_local=True)


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


def test_register_and_list_versions(tmp_path: Path) -> None:
    reg = _local_registry(tmp_path)
    signature = _signature(["price", "popularity"])
    version = reg.register(
        "rank_v1",
        "s3://bucket/models/rank/1",
        signature=signature,
        metrics={"auc": 0.91},
        data_span="2024-01-01:2024-01-15",
        feature_version="features:v5",
    )

    assert version.version == "1"
    assert version.stage == "None"
    assert version.metrics == {"auc": 0.91}

    versions = reg.list_versions("rank_v1")
    assert len(versions) == 1
    assert versions[0].signature == signature
    assert versions[0].feature_version == "features:v5"


def test_promote_and_demote_versions(tmp_path: Path) -> None:
    reg = _local_registry(tmp_path)
    signature = _signature(["price", "popularity"])

    v1 = reg.register("rank_v1", "s3://bucket/models/rank/1", signature=signature)
    promoted = reg.promote("rank_v1", v1.version, "staging")
    assert promoted.stage == "Staging"

    prod = reg.promote("rank_v1", v1.version, "prod")
    assert prod.stage == "Production"

    v2 = reg.register("rank_v1", "s3://bucket/models/rank/2", signature=signature)
    new_prod = reg.promote("rank_v1", v2.version, "production")
    assert new_prod.stage == "Production"

    archived_versions = {mv.version: mv.stage for mv in reg.list_versions("rank_v1")}
    assert archived_versions["1"] == "Archived"
    assert archived_versions["2"] == "Production"

    demoted = reg.demote("rank_v1", v2.version, "none")
    assert demoted.stage == "None"


def test_promote_without_signature_is_rejected(tmp_path: Path) -> None:
    reg = _local_registry(tmp_path)
    version = reg.register("rank_v1", "s3://bucket/models/rank/1", signature=None)

    with pytest.raises(registry.RegistryError):
        reg.promote("rank_v1", version.version, "prod")


def test_signature_mismatch_blocks_promotion(tmp_path: Path) -> None:
    reg = _local_registry(tmp_path)
    v1 = reg.register("rank_v1", "s3://bucket/models/rank/1", signature=_signature(["price"]))
    reg.promote("rank_v1", v1.version, "prod")

    v2 = reg.register("rank_v1", "s3://bucket/models/rank/2", signature=_signature(["inventory"]))

    with pytest.raises(registry.SignatureMismatchError):
        reg.promote("rank_v1", v2.version, "prod")


def test_cli_register_and_promote(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registry_file = tmp_path / "registry.json"
    signature_path = tmp_path / "signature.json"
    signature_path.write_text(json.dumps(_signature(["price", "popularity"])))

    assert (
        registry.main(
            [
                "--local",
                "--registry-file",
                str(registry_file),
                "register",
                "--name",
                "rank_v1",
                "--artifact-uri",
                "s3://bucket/models/rank/1",
                "--signature",
                str(signature_path),
                "--metric",
                "auc=0.92",
            ]
        )
        == 0
    )

    assert (
        registry.main(
            [
                "--local",
                "--registry-file",
                str(registry_file),
                "promote",
                "--name",
                "rank_v1",
                "--version",
                "1",
                "--to",
                "prod",
            ]
        )
        == 0
    )

    registry.main(
        [
            "--local",
            "--registry-file",
            str(registry_file),
            "list",
            "--name",
            "rank_v1",
        ]
    )
    stdout = capsys.readouterr().out
    assert "v1" in stdout
    assert "Production" in stdout
