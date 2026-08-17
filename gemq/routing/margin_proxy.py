"""Pure helpers for the Phase 4 near-boundary route-risk proxy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr


BITS = (1, 2, 3)


def nested_to_tensor(nested: Mapping, *, layers: int = 16, experts: int = 64) -> np.ndarray:
    """Convert the persisted layer/expert/bit mapping to a validated dense tensor."""
    tensor = np.asarray(
        [
            [
                [nested[layer][expert][bit] for bit in BITS]
                for expert in range(experts)
            ]
            for layer in range(layers)
        ],
        dtype=np.float64,
    )
    if tensor.shape != (layers, experts, len(BITS)):
        raise ValueError(f"Unexpected coefficient tensor shape: {tensor.shape}")
    if not np.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError("Route coefficients must be finite and non-negative")
    return tensor


def normalize_scenario_tensor(raw: np.ndarray, effective_tokens: int) -> tuple[np.ndarray, float]:
    """Apply the pre-registered per-token then median-bit2 normalization."""
    raw = np.asarray(raw, dtype=np.float64)
    if effective_tokens <= 0:
        raise ValueError("effective_tokens must be positive")
    if raw.ndim != 3 or raw.shape[-1] != len(BITS):
        raise ValueError(f"Expected [layer, expert, bit] tensor, got {raw.shape}")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Raw route costs must be finite and non-negative")
    per_token = raw / float(effective_tokens)
    scale = float(np.median(per_token[:, :, BITS.index(2)]))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"Invalid median bit-2 route scale: {scale}")
    return per_token / scale, scale


def allocation_cost(tensor: np.ndarray, config: Mapping) -> float:
    """Evaluate one expert-bit allocation against a dense cost tensor."""
    tensor = np.asarray(tensor, dtype=np.float64)
    layers, experts, bit_count = tensor.shape
    if bit_count != len(BITS):
        raise ValueError(f"Expected {len(BITS)} bit choices, got {bit_count}")
    return float(
        sum(
            tensor[layer, expert, BITS.index(int(config[layer][expert]))]
            for layer in range(layers)
            for expert in range(experts)
        )
    )


def partial_spearman(x: Sequence[float], y: Sequence[float], control: Sequence[float]) -> float:
    """Spearman correlation after rank-residualizing one scalar confounder."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if x.shape != y.shape or x.shape != control.shape or x.ndim != 1:
        raise ValueError("x, y, and control must be same-length one-dimensional arrays")
    if len(x) < 4:
        raise ValueError("At least four observations are required")
    x_rank, y_rank, control_rank = rankdata(x), rankdata(y), rankdata(control)
    design = np.column_stack([np.ones(len(control_rank)), control_rank])
    x_residual = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
    y_residual = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
    value = float(spearmanr(x_residual, y_residual).statistic)
    if not np.isfinite(value):
        raise ValueError("Partial Spearman is undefined for the supplied observations")
    return value


def bootstrap_partial_spearman(
    x: Sequence[float],
    y: Sequence[float],
    control: Sequence[float],
    *,
    iterations: int = 4000,
    seed: int = 20260816,
) -> tuple[float, float]:
    """Configuration-level percentile bootstrap for partial Spearman."""
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        indices = rng.integers(0, len(x), len(x))
        if len(np.unique(control[indices])) < 2:
            continue
        try:
            values.append(partial_spearman(x[indices], y[indices], control[indices]))
        except ValueError:
            continue
    if len(values) < iterations // 2:
        raise ValueError("Too few finite bootstrap replicates")
    return tuple(map(float, np.quantile(values, [0.025, 0.975])))
