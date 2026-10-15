"""
objective_functions.py
The two objective (fitness) functions of the study.

Composite objective (formula 1 from the board), maximised:
    f_i = alpha*(d_geo*k / d_i) + (1-alpha)*(A_geo / A_cum)  ->  max,  alpha in (0,1)
Each term is a reward: the first grows as the agent nears the target
(d_i -> 0), the second grows when little work has been spent. Maximising
f_i is equivalent to minimising 1/f_i. A small epsilon guards the
divisions when d_i or A_cum approach zero.

Baseline objective (formula 2), minimised:
    f_i = d_i  ->  min
Distance only, ignoring work. Used as the control for the composite form.

`better(a, b, mode)` abstracts the optimisation direction so the rest of
the code (personal/global best updates) is mode-agnostic.
"""
from __future__ import annotations

_EPS = 1e-6


def composite_objective(d_i: float, d_geo: float, k: float,
                        a_cum: float, a_geo: float, alpha: float) -> float:
    """f_i (formula 1); higher is better."""
    distance_reward = (d_geo * k) / (d_i + _EPS)
    work_reward = a_geo / (a_cum + _EPS)
    return alpha * distance_reward + (1.0 - alpha) * work_reward


def baseline_objective(d_i: float) -> float:
    """f_i = d_i (formula 2); lower is better."""
    return d_i


def evaluate(mode: str, *, d_i: float, d_geo: float, k: float,
             a_cum: float, a_geo: float, alpha: float | None) -> float:
    if mode == "composite":
        return composite_objective(d_i, d_geo, k, a_cum, a_geo, alpha)
    if mode == "baseline":
        return baseline_objective(d_i)
    raise ValueError(f"Unknown objective mode: {mode!r}")


def better(new: float, best: float, mode: str) -> bool:
    """True if `new` fitness is better than `best` under the mode's
    optimisation direction (composite -> max, baseline -> min)."""
    if mode == "composite":
        return new > best
    return new < best


def worst_value(mode: str) -> float:
    """Initial 'best' value: -inf for maximisation, +inf for minimisation."""
    return float("-inf") if mode == "composite" else float("inf")
