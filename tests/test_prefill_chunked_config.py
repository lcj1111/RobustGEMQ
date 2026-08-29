from types import SimpleNamespace

import pytest

from gemq.inference.moe_block import QuantFusedOlmoeMoEBlock


def _config():
    return SimpleNamespace(
        hidden_size=64,
        intermediate_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )


def test_chunked_backend_has_bounded_default(monkeypatch):
    monkeypatch.setenv("GEMQ_PREFILL_BACKEND", "chunked")
    monkeypatch.delenv("GEMQ_PREFILL_CHUNK_TOKENS", raising=False)
    block = QuantFusedOlmoeMoEBlock(_config())
    assert block.prefill_backend == "chunked"
    assert block.prefill_chunk_tokens == 512


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_chunk_token_limit_must_be_positive_integer(monkeypatch, value):
    monkeypatch.setenv("GEMQ_PREFILL_BACKEND", "chunked")
    monkeypatch.setenv("GEMQ_PREFILL_CHUNK_TOKENS", value)
    with pytest.raises(ValueError, match="必须为正整数"):
        QuantFusedOlmoeMoEBlock(_config())


def test_chunk_token_limit_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GEMQ_PREFILL_BACKEND", "chunked")
    monkeypatch.setenv("GEMQ_PREFILL_CHUNK_TOKENS", "256")
    block = QuantFusedOlmoeMoEBlock(_config())
    assert block.prefill_chunk_tokens == 256
