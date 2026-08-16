"""Router diagnostics and validated proxy utilities."""

from .margin_proxy import (
    BITS,
    allocation_cost,
    bootstrap_partial_spearman,
    nested_to_tensor,
    normalize_scenario_tensor,
    partial_spearman,
)

__all__ = [
    "BITS",
    "allocation_cost",
    "bootstrap_partial_spearman",
    "nested_to_tensor",
    "normalize_scenario_tensor",
    "partial_spearman",
]
