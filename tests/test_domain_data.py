import json

import pytest

from gemq.utils.domain_data import (
    build_domain_scenario,
    format_record,
    load_domain_registry,
    load_scenario_tokens,
    save_domain_scenario,
)


class ToyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


def write_registry(tmp_path, allocation_path="general/train.jsonl"):
    registry = {
        "schema_version": 1,
        "registry_id": "test",
        "domains": {
            "general": {
                "source": "test/source",
                "revision": "deadbeef",
                "license": "MIT",
                "allocation": {
                    "path": allocation_path,
                    "format": "jsonl",
                    "id_field": "id",
                    "template": "{text}",
                },
                "held_out": ["general/test.jsonl"],
            }
        },
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def write_records(tmp_path):
    path = tmp_path / "data" / "general" / "train.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(["abcdefgh", "ijklmnop", "qrstuvwx", "yz012345"]):
            handle.write(json.dumps({"id": index, "text": text}) + "\n")
    return path


def test_registry_rejects_unknown_schema(tmp_path):
    path = write_registry(tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        load_domain_registry(path)


def test_format_record_normalizes_lists_and_nulls():
    text = format_record({"tests": ["a", "b"], "context": None}, "{tests}|{context}")
    assert text == "a\nb|"


def test_scenario_is_deterministic_and_auditable(tmp_path):
    source = write_records(tmp_path)
    registry = write_registry(tmp_path)
    first, manifest_first = build_domain_scenario(
        registry_path=registry,
        data_root=tmp_path / "data",
        domain="general",
        tokenizer=ToyTokenizer(),
        nsamples=3,
        seqlen=8,
        seed=7,
    )
    second, manifest_second = build_domain_scenario(
        registry_path=registry,
        data_root=tmp_path / "data",
        domain="general",
        tokenizer=ToyTokenizer(),
        nsamples=3,
        seqlen=8,
        seed=7,
    )
    assert first.equal(second)
    assert manifest_first == manifest_second
    assert manifest_first["effective_tokens"] == 24
    assert manifest_first["allocation_sha256"]

    saved = save_domain_scenario(first, manifest_first, tmp_path / "scenario")
    loader = load_scenario_tokens(saved["tokens_path"])
    assert len(loader) == 3
    assert tuple(loader[0][0].shape) == (1, 8)


def test_scenario_refuses_path_escape(tmp_path):
    write_records(tmp_path)
    registry = write_registry(tmp_path, allocation_path="../outside.jsonl")
    with pytest.raises(ValueError, match="escapes data root"):
        build_domain_scenario(
            registry_path=registry,
            data_root=tmp_path / "data",
            domain="general",
            tokenizer=ToyTokenizer(),
            nsamples=1,
            seqlen=4,
            seed=0,
        )
