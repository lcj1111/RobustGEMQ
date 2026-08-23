"""面向 MoE 专家的可审计鲁棒 bit 分配。

所有场景共享同一组二元决策变量。Domain-Mean、Domain-Worst 和
Domain-CVaR 仅风险目标不同，使用相同的重构误差张量与可行域。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp


OBJECTIVES = ("mean", "worst", "cvar")


def _as_tensor(value, bits: tuple[int, ...]) -> np.ndarray:
    """接收稠密 `[layer, expert, bit]` 张量或 GEMQ 的嵌套字典。"""
    if isinstance(value, np.ndarray):
        tensor = np.asarray(value, dtype=np.float64)
    else:
        layers = sorted(value)
        experts = sorted(value[layers[0]])
        tensor = np.asarray(
            [
                [[value[layer][expert][bit] for bit in bits] for expert in experts]
                for layer in layers
            ],
            dtype=np.float64,
        )
    if tensor.ndim != 3 or tensor.shape[2] != len(bits):
        raise ValueError(
            f"Each scenario must have shape [layers, experts, {len(bits)}], got {tensor.shape}"
        )
    if not np.isfinite(tensor).all():
        raise ValueError("Scenario coefficients must all be finite")
    if (tensor < 0).any():
        raise ValueError("Scenario coefficients must be non-negative")
    return np.ascontiguousarray(tensor)


def empirical_cvar(losses, alpha: float, weights=None) -> float:
    """按与求解器一致的 LP 定义计算加权上尾 CVaR。"""
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("losses must be a non-empty finite vector")
    if not 0 <= alpha < 1:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    if weights is None:
        probabilities = np.full(values.size, 1.0 / values.size)
    else:
        probabilities = np.asarray(weights, dtype=np.float64)
        if probabilities.shape != values.shape or not np.isfinite(probabilities).all():
            raise ValueError("weights must have the same shape as losses and be finite")
        if (probabilities < 0).any() or probabilities.sum() <= 0:
            raise ValueError("weights must be non-negative with positive total mass")
        probabilities = probabilities / probabilities.sum()

    # 该分段线性目标的最优 eta 必落在某个观测损失上，无需额外做连续优化。
    return float(
        min(
            eta + np.dot(probabilities, np.maximum(values - eta, 0.0)) / (1.0 - alpha)
            for eta in values
        )
    )


@dataclass(frozen=True)
class RobustSolveResult:
    allocation: dict[int, dict[int, int]]
    audit: dict


class RobustGEMQSolver:
    """在固定领域场景上求解 Mean、Worst 或 CVaR 风险。"""

    def __init__(
        self,
        scenario_coefficients: Mapping[str, object],
        *,
        objective: str,
        bits=(1, 2, 3),
        scenario_weights: Mapping[str, float] | None = None,
        alpha: float = 0.5,
        extra_constr: str = "c2c3",
        fixed_assignments: Mapping[tuple[int, int], int] | None = None,
        start_layer_idx: int = 0,
    ):
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        if not scenario_coefficients:
            raise ValueError("At least one scenario is required")
        self.objective = objective
        self.bits = tuple(int(bit) for bit in bits)
        if len(self.bits) != len(set(self.bits)) or any(bit <= 0 for bit in self.bits):
            raise ValueError("bits must be unique positive integers")
        self.names = tuple(sorted(scenario_coefficients))
        tensors = [_as_tensor(scenario_coefficients[name], self.bits) for name in self.names]
        if len({tensor.shape for tensor in tensors}) != 1:
            raise ValueError("All scenarios must have the same tensor shape")
        self.tensor = np.stack(tensors)
        self.optimization_scale = float(np.max(self.tensor))
        if self.optimization_scale == 0.0:
            self.optimization_scale = 1.0
        self.num_scenarios, self.num_layers, self.num_experts, self.num_bits = self.tensor.shape
        self.start_layer_idx = int(start_layer_idx)
        self.extra_constr = extra_constr
        if extra_constr not in ("none", "", "c2c3"):
            raise ValueError("extra_constr must be 'none' or 'c2c3'")
        if not 0 <= alpha < 1:
            raise ValueError("alpha must satisfy 0 <= alpha < 1")
        self.alpha = float(alpha)

        supplied = scenario_weights or {name: 1.0 for name in self.names}
        if set(supplied) != set(self.names):
            raise ValueError("scenario_weights keys must exactly match scenario names")
        weights = np.asarray([supplied[name] for name in self.names], dtype=np.float64)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("scenario weights must be finite, non-negative, and have positive mass")
        self.weights = weights / weights.sum()

        self.fixed_assignments = dict(fixed_assignments or {})
        for (layer, expert), bit in self.fixed_assignments.items():
            if not self.start_layer_idx <= layer < self.start_layer_idx + self.num_layers:
                raise ValueError(f"Fixed layer {layer} is outside the solver range")
            if not 0 <= expert < self.num_experts:
                raise ValueError(f"Fixed expert {expert} is outside the solver range")
            if bit not in self.bits:
                raise ValueError(f"Fixed bit {bit} is not in candidate bits {self.bits}")

    @property
    def num_binary(self) -> int:
        return self.num_layers * self.num_experts * self.num_bits

    def _binary_index(self, local_layer: int, expert: int, bit_index: int) -> int:
        return (local_layer * self.num_experts + expert) * self.num_bits + bit_index

    def _base_constraints(self, variable_count: int, total_bits: float) -> list[LinearConstraint]:
        if not np.isfinite(total_bits):
            raise ValueError("total_bits must be finite")
        rows, cols, data, lower, upper = [], [], [], [], []

        def append(entries, lo, hi):
            row = len(lower)
            for col, value in entries:
                rows.append(row)
                cols.append(col)
                data.append(value)
            lower.append(lo)
            upper.append(hi)

        append(
            ((index, self.bits[index % self.num_bits]) for index in range(self.num_binary)),
            -np.inf,
            float(total_bits),
        )
        for local_layer in range(self.num_layers):
            for expert in range(self.num_experts):
                append(
                    ((self._binary_index(local_layer, expert, k), 1.0) for k in range(self.num_bits)),
                    1.0,
                    1.0,
                )
        if self.extra_constr == "c2c3":
            if self.num_bits < 2:
                raise ValueError("c2c3 requires at least two candidate bits")
            for bit in sorted(self.bits, reverse=True)[:2]:
                bit_index = self.bits.index(bit)
                for local_layer in range(self.num_layers):
                    append(
                        (
                            (self._binary_index(local_layer, expert, bit_index), 1.0)
                            for expert in range(self.num_experts)
                        ),
                        1.0,
                        np.inf,
                    )
        for (layer, expert), bit in sorted(self.fixed_assignments.items()):
            local_layer = layer - self.start_layer_idx
            append([(self._binary_index(local_layer, expert, self.bits.index(bit)), 1.0)], 1.0, 1.0)

        matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(lower), variable_count))
        return [LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))]

    def _formulate(self, total_bits: float):
        # HiGHS 会拒绝过大的系数，并可能忽略极小系数；统一正比例缩放不改变 argmin。
        flattened = (
            self.tensor / self.optimization_scale
        ).reshape(self.num_scenarios, self.num_binary)
        if self.objective == "mean":
            c = self.weights @ flattened
            variable_count = self.num_binary
            constraints = self._base_constraints(variable_count, total_bits)
            lower = np.zeros(variable_count)
            upper = np.ones(variable_count)
        elif self.objective == "worst":
            variable_count = self.num_binary + 1
            eta = self.num_binary
            c = np.zeros(variable_count)
            c[eta] = 1.0
            constraints = self._base_constraints(variable_count, total_bits)
            risk = sp.hstack(
                [sp.csr_matrix(flattened), -np.ones((self.num_scenarios, 1))],
                format="csr",
            )
            constraints.append(LinearConstraint(risk, -np.inf, np.zeros(self.num_scenarios)))
            lower = np.zeros(variable_count)
            upper = np.concatenate([np.ones(self.num_binary), [np.inf]])
        else:
            variable_count = self.num_binary + 1 + self.num_scenarios
            eta = self.num_binary
            excess_start = eta + 1
            c = np.zeros(variable_count)
            c[eta] = 1.0
            c[excess_start:] = self.weights / (1.0 - self.alpha)
            constraints = self._base_constraints(variable_count, total_bits)
            # 每个场景的超额损失：loss_s - eta - u_s <= 0。
            risk_rows = []
            for scenario in range(self.num_scenarios):
                row = sp.lil_matrix((1, variable_count))
                row[0, : self.num_binary] = flattened[scenario]
                row[0, eta] = -1.0
                row[0, excess_start + scenario] = -1.0
                risk_rows.append(row.tocsr())
            constraints.append(
                LinearConstraint(sp.vstack(risk_rows, format="csr"), -np.inf, np.zeros(self.num_scenarios))
            )
            lower = np.zeros(variable_count)
            upper = np.concatenate([np.ones(self.num_binary), np.full(1 + self.num_scenarios, np.inf)])
        integrality = np.concatenate(
            [np.ones(self.num_binary), np.zeros(variable_count - self.num_binary)]
        )
        return c, constraints, integrality, Bounds(lower, upper)

    def _decode(self, raw: np.ndarray) -> dict[int, dict[int, int]]:
        selected = raw[: self.num_binary].reshape(self.num_layers, self.num_experts, self.num_bits)
        rounded = np.rint(selected)
        if not np.allclose(selected, rounded, atol=1e-6) or not np.all(rounded.sum(axis=2) == 1):
            raise RuntimeError("Solver output violates binary one-hot assignment")
        indices = rounded.argmax(axis=2)
        return {
            self.start_layer_idx + layer: {
                expert: self.bits[indices[layer, expert]] for expert in range(self.num_experts)
            }
            for layer in range(self.num_layers)
        }

    def scenario_losses(self, allocation: Mapping[int, Mapping[int, int]]) -> dict[str, float]:
        losses = {}
        for scenario_index, name in enumerate(self.names):
            total = 0.0
            for local_layer in range(self.num_layers):
                layer = self.start_layer_idx + local_layer
                for expert in range(self.num_experts):
                    bit = allocation[layer][expert]
                    total += self.tensor[scenario_index, local_layer, expert, self.bits.index(bit)]
            losses[name] = float(total)
        return losses

    def _risk(self, losses: Mapping[str, float]) -> float:
        values = np.asarray([losses[name] for name in self.names])
        if self.objective == "mean":
            return float(np.dot(self.weights, values))
        if self.objective == "worst":
            return float(values.max())
        return empirical_cvar(values, self.alpha, self.weights)

    def _schema_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps({"names": self.names, "bits": self.bits}).encode())
        digest.update(self.tensor.tobytes(order="C"))
        return digest.hexdigest()

    def solve(self, total_bits: float) -> RobustSolveResult:
        minimum = self.num_layers * self.num_experts * min(self.bits)
        fixed_minimum = minimum + sum(
            bit - min(self.bits) for bit in self.fixed_assignments.values()
        )
        if total_bits < fixed_minimum:
            raise RuntimeError(
                f"Infeasible bit budget {total_bits:g}; fixed assignments require at least {fixed_minimum:g}"
            )
        c, constraints, integrality, bounds = self._formulate(total_bits)
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if not result.success or result.x is None:
            raise RuntimeError(f"HiGHS failed to solve robust allocation: {result.message}")
        allocation = self._decode(result.x)
        used_bits = int(sum(bit for experts in allocation.values() for bit in experts.values()))
        one_hot_violations = sum(
            set(experts) != set(range(self.num_experts)) for experts in allocation.values()
        )
        fixed_violations = sum(
            allocation[layer][expert] != bit
            for (layer, expert), bit in self.fixed_assignments.items()
        )
        losses = self.scenario_losses(allocation)
        recomputed = self._risk(losses)
        solver_objective = float(result.fun) * self.optimization_scale
        tolerance = 1e-6 * max(1.0, abs(recomputed))
        if used_bits > total_bits + 1e-6 or one_hot_violations or fixed_violations:
            raise RuntimeError("Decoded allocation failed feasibility audit")
        if abs(solver_objective - recomputed) > tolerance:
            raise RuntimeError(
                f"Objective audit failed: solver={solver_objective}, recomputed={recomputed}"
            )
        audit = {
            "schema_version": 1,
            "solver": "scipy.optimize.milp/HiGHS",
            "scipy_version": scipy.__version__,
            "status": "optimal-audited",
            "solver_message": str(result.message),
            "objective": self.objective,
            "alpha": self.alpha if self.objective == "cvar" else None,
            "scenario_names": list(self.names),
            "scenario_weights": {name: float(self.weights[i]) for i, name in enumerate(self.names)},
            "coefficient_schema_sha256": self._schema_hash(),
            "optimization_scale": self.optimization_scale,
            "shape": list(self.tensor.shape),
            "candidate_bits": list(self.bits),
            "total_bit_budget": float(total_bits),
            "used_bits": used_bits,
            "budget_slack": float(total_bits - used_bits),
            "extra_constraint": self.extra_constr or "none",
            "fixed_assignments": {
                f"{layer}:{expert}": bit for (layer, expert), bit in sorted(self.fixed_assignments.items())
            },
            "one_hot_violations": int(one_hot_violations),
            "fixed_assignment_violations": int(fixed_violations),
            "scenario_losses": losses,
            "solver_objective": solver_objective,
            "recomputed_objective": recomputed,
            "objective_abs_error": abs(solver_objective - recomputed),
            "mip_gap": float(getattr(result, "mip_gap", np.nan)),
            "mip_node_count": int(getattr(result, "mip_node_count", 0)),
        }
        return RobustSolveResult(allocation=allocation, audit=audit)
