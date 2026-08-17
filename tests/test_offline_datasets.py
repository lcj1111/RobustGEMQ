import json

import pytest
from datasets import Dataset

from gemq.utils.data_utils import load_c4_split, load_wikitext2_split


def test_load_wikitext_from_local_parquet(monkeypatch, tmp_path):
    Dataset.from_dict({"text": ["alpha", "beta"]}).to_parquet(
        tmp_path / "train-00000-of-00001.parquet"
    )
    monkeypatch.setenv("GEMQ_WIKITEXT_DIR", str(tmp_path))

    split = load_wikitext2_split("train")

    assert split["text"] == ["alpha", "beta"]


def test_load_c4_from_local_jsonl(monkeypatch, tmp_path):
    source = tmp_path / "c4-train.json"
    source.write_text(
        "\n".join(json.dumps({"text": text}) for text in ["first", "second"]),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMQ_C4_TRAIN_FILE", str(source))

    split = load_c4_split("train")

    assert split["text"] == ["first", "second"]


def test_configured_dataset_path_must_exist(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("GEMQ_WIKITEXT_DIR", str(missing))

    with pytest.raises(FileNotFoundError, match="GEMQ_WIKITEXT_DIR"):
        load_wikitext2_split("test")


def test_local_wikitext_requires_requested_split(monkeypatch, tmp_path):
    Dataset.from_dict({"text": ["train only"]}).to_parquet(
        tmp_path / "train-00000-of-00001.parquet"
    )
    monkeypatch.setenv("GEMQ_WIKITEXT_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="split 'test'"):
        load_wikitext2_split("test")
