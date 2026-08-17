"""Exactness and audit tests for the Phase 3 robust allocation solver."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from gemq.allocation.robust_solvers import RobustGEMQSolver, empirical_cvar


BITS = (1, 2, 3)


def risk(tensor, assignment, objective, weights, alpha):
    losses = np.asarray(
        [sum(tensor[s, i, bit - 1] for i, bit in enumerate(assignment)) for s in range(tensor.shape[0])]
    )
    if objective == "mean":
        return float(weights @ losses)
    if objective == "worst":
        return float(losses.max())
    return empirical_cvar(losses, alpha, weights)


def brute_force(tensor, layers, experts, budget, objective, weights, alpha, extra, fixed=None):
    fixed = fixed or {}
    best = np.inf
    for assignment in itertools.product(BITS, repeat=layers * experts):
        if sum(assignment) > budget:
            continue
        if any(assignment[(layer * experts) + expert] != bit for (layer, expert), bit in fixed.items()):
            continue
        if extra == "c2c3":
            if any(
                not all(bit in assignment[layer * experts : (layer + 1) * experts] for bit in (2, 3))
                for layer in range(layers)
            ):
                continue
        best = min(best, risk(tensor, assignment, objective, weights, alpha))
    return best


@pytest.mark.parametrize("seed", range(24))
def test_random_small_problem_matches_brute_force(seed):
    """Twenty-four independent MILPs must agree with exhaustive enumeration."""
    rng = np.random.default_rng(seed)
    layers, experts, scenarios = 2, 3, 3
    tensor = rng.lognormal(mean=0.0, sigma=1.2, size=(scenarios, layers * experts, len(BITS)))
    # Quantization loss usually decreases with bit width; enforce that without making
    # the optimum trivial because sensitivity still varies by expert and scenario.
    tensor.sort(axis=2)
    tensor = tensor[:, :, ::-1]
    tensor4 = tensor.reshape(scenarios, layers, experts, len(BITS))
    names = {f"domain-{s}": tensor4[s] for s in range(scenarios)}
    raw_weights = rng.uniform(0.1, 2.0, size=scenarios)
    weights = raw_weights / raw_weights.sum()
    objective = ("mean", "worst", "cvar")[seed % 3]
    extra = "c2c3" if seed % 2 else "none"
    minimum = 12 if extra == "c2c3" else 6
    budget = int(rng.integers(minimum, 17))
    alpha = 0.5

    result = RobustGEMQSolver(
        names,
        objective=objective,
        scenario_weights={name: raw_weights[i] for i, name in enumerate(names)},
        alpha=alpha,
        extra_constr=extra,
    ).solve(budget)
    actual = result.audit["recomputed_objective"]
    expected = brute_force(tensor, layers, experts, budget, objective, weights, alpha, extra)
    assert actual == pytest.approx(expected, rel=1e-8, abs=1e-8)
    assert result.audit["used_bits"] <= budget
    assert result.audit["one_hot_violations"] == 0
    assert result.audit["objective_abs_error"] <= 1e-6 * max(1.0, abs(actual))


@pytest.mark.parametrize("objective", ["mean", "worst", "cvar"])
def test_duplicate_scenarios_reduce_to_single_scenario(objective):
    tensor = np.asarray([[[9.0, 3.0, 1.0], [8.0, 4.0, 0.5]]])
    duplicate = RobustGEMQSolver(
        {"a": tensor, "b": tensor.copy()}, objective=objective, extra_constr="none"
    ).solve(4)
    single = RobustGEMQSolver(
        {"a": tensor}, objective=objective, extra_constr="none"
    ).solve(4)
    assert duplicate.audit["recomputed_objective"] == pytest.approx(
        single.audit["recomputed_objective"]
    )


def test_fixed_assignment_is_enforced_and_audited():
    tensor = np.asarray([[[9.0, 2.0, 1.0], [8.0, 3.0, 0.5]]])
    result = RobustGEMQSolver(
        {"domain": tensor},
        objective="mean",
        extra_constr="none",
        fixed_assignments={(0, 0): 1},
    ).solve(4)
    assert result.allocation[0][0] == 1
    assert result.audit["fixed_assignments"] == {"0:0": 1}
    assert result.audit["fixed_assignment_violations"] == 0


def test_infeasible_budget_fails_before_solver():
    tensor = np.ones((1, 2, 3))
    solver = RobustGEMQSolver(
        {"domain": tensor}, objective="mean", extra_constr="none", fixed_assignments={(0, 0): 3}
    )
    with pytest.raises(RuntimeError, match="Infeasible bit budget"):
        solver.solve(3)


def test_c2c3_infeasibility_is_reported():
    tensor = np.ones((1, 2, 3))
    solver = RobustGEMQSolver({"domain": tensor}, objective="worst", extra_constr="c2c3")
    with pytest.raises(RuntimeError, match="HiGHS failed"):
        solver.solve(4)


def test_weighted_cvar_matches_known_fractional_tail():
    # At alpha=.5 the upper half consists of all mass at 10 plus 0.1/0.5 of loss 2.
    assert empirical_cvar([0.0, 2.0, 10.0], 0.5, [0.4, 0.4, 0.2]) == pytest.approx(5.2)


@pytest.mark.parametrize("scale", [1e-120, 1e120])
def test_numerical_extremes_remain_auditable(scale):
    tensor = scale * np.asarray([[[9.0, 2.0, 1.0], [7.0, 3.0, 0.5]]])
    result = RobustGEMQSolver(
        {"a": tensor, "b": tensor * 1.1}, objective="cvar", extra_constr="none"
    ).solve(4)
    assert np.isfinite(result.audit["recomputed_objective"])
    assert result.audit["status"] == "optimal-audited"


@pytest.mark.parametrize(
    "scenarios,match",
    [
        ({"a": np.asarray([[[np.nan, 1.0, 0.0]]])}, "finite"),
        ({"a": np.asarray([[[-1.0, 1.0, 0.0]]])}, "non-negative"),
        (
            {"a": np.ones((1, 1, 3)), "b": np.ones((2, 1, 3))},
            "same tensor shape",
        ),
    ],
)
def test_invalid_scenario_schema_is_rejected(scenarios, match):
    with pytest.raises(ValueError, match=match):
        RobustGEMQSolver(scenarios, objective="mean")


def test_scenario_weight_keys_must_match():
    with pytest.raises(ValueError, match="exactly match"):
        RobustGEMQSolver(
            {"a": np.ones((1, 1, 3)), "b": np.ones((1, 1, 3))},
            objective="mean",
            scenario_weights={"a": 1.0},
        )
