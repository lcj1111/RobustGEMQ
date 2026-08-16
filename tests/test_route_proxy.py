"""Unit tests for the Phase 4 near-boundary route proxy."""

from __future__ import annotations

import numpy as np
import pytest

from gemq.routing.margin_proxy import (
    allocation_cost,
    bootstrap_partial_spearman,
    nested_to_tensor,
    normalize_scenario_tensor,
    partial_spearman,
)


def nested(layers=2, experts=3):
    return {
        layer: {
            expert: {1: 9.0 + expert, 2: 3.0 + expert, 3: 1.0 + expert}
            for expert in range(experts)
        }
        for layer in range(layers)
    }


def test_nested_tensor_and_allocation_cost():
    tensor = nested_to_tensor(nested(), layers=2, experts=3)
    config = {layer: {expert: (expert % 3) + 1 for expert in range(3)} for layer in range(2)}
    expected = sum(nested()[layer][expert][config[layer][expert]] for layer in range(2) for expert in range(3))
    assert allocation_cost(tensor, config) == pytest.approx(expected)


def test_normalization_is_per_token_then_median_bit2():
    raw = nested_to_tensor(nested(), layers=2, experts=3)
    normalized, scale = normalize_scenario_tensor(raw, effective_tokens=8)
    assert scale == pytest.approx(np.median(raw[:, :, 1]) / 8)
    assert np.median(normalized[:, :, 1]) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [0, -1])
def test_normalization_rejects_invalid_token_count(bad):
    with pytest.raises(ValueError, match="positive"):
        normalize_scenario_tensor(np.ones((2, 3, 3)), bad)


def test_partial_spearman_removes_monotone_hamming_confounder():
    control = np.repeat(np.arange(5), 4)
    signal = np.tile(np.arange(4), 5)
    x = 10 * control + signal
    y = 20 * control + signal
    assert partial_spearman(x, y, control) > 0.9


def test_bootstrap_is_deterministic_and_positive_for_strong_signal():
    control = np.repeat(np.arange(5), 8)
    signal = np.tile(np.arange(8), 5)
    x = 5 * control + signal
    y = 7 * control + signal
    left = bootstrap_partial_spearman(x, y, control, iterations=200, seed=11)
    right = bootstrap_partial_spearman(x, y, control, iterations=200, seed=11)
    assert left == right
    assert left[0] > 0
