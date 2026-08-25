from __future__ import annotations

import numpy as np

from gemq.allocation.control_allocations import config_array, layer_balanced_config, usage_only_config


def assert_exact_layer_balance(config):
    values = config_array(config)
    assert values.shape == (16, 64)
    assert int(values.sum()) == 2560
    for layer in range(16):
        assert np.count_nonzero(values[layer] == 2) == 32
        assert np.count_nonzero(values[layer] == 3) == 32


def test_layer_balanced_control_has_exact_per_layer_budget():
    assert_exact_layer_balance(layer_balanced_config())


def test_usage_only_selects_highest_usage_with_stable_ties():
    usage = np.tile(np.arange(64, dtype=np.int64), (16, 1))
    config = usage_only_config(usage)
    assert_exact_layer_balance(config)
    values = config_array(config)
    assert np.all(values[:, :32] == 2)
    assert np.all(values[:, 32:] == 3)


def test_usage_only_breaks_equal_usage_ties_by_expert_id():
    usage = np.ones((16, 64), dtype=np.int64)
    values = config_array(usage_only_config(usage))
    assert np.all(values[:, :32] == 3)
    assert np.all(values[:, 32:] == 2)
