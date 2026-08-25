"""不依赖重建误差求解器的确定性对照 allocation。"""

from __future__ import annotations

import numpy as np


def config_array(config: dict) -> np.ndarray:
    return np.asarray(
        [[int(config[layer][expert]) for expert in range(64)] for layer in range(16)],
        dtype=np.int8,
    )


def array_config(values: np.ndarray) -> dict:
    if values.shape != (16, 64):
        raise ValueError("allocation must have shape [16, 64]")
    return {
        layer: {expert: int(values[layer, expert]) for expert in range(64)}
        for layer in range(16)
    }


def layer_balanced_config() -> dict:
    """每层固定 32 个 2-bit 与 32 个 3-bit，并按层旋转奇偶位置。"""
    values = np.empty((16, 64), dtype=np.int8)
    for layer in range(16):
        for expert in range(64):
            values[layer, expert] = 3 if (layer + expert) % 2 == 0 else 2
    return array_config(values)


def usage_only_config(usage: np.ndarray) -> dict:
    """每层将 usage 最大的 32 个 expert 置为 3-bit，其余置为 2-bit。"""
    if usage.shape != (16, 64) or (usage < 0).any():
        raise ValueError("usage must be a non-negative [16, 64] tensor")
    values = np.full((16, 64), 2, dtype=np.int8)
    for layer in range(16):
        # usage 降序优先；相同时 expert id 小者优先，保证结果确定。
        order = np.lexsort((np.arange(64), -usage[layer]))
        values[layer, order[:32]] = 3
    return array_config(values)
