"""
repulsion.py
Inter-agent repulsion force, written in a SCALE-INVARIANT (normalised)
form so that the force magnitude is comparable to the spring forces on a
real metric terrain (coordinates in metres), unlike the raw inverse-power
law of the small-scale prototype, whose force vanishes at kilometre
scales.

For each agent i, summing over neighbours j within the interaction radius
rho_0:
    F_rep(i) = w * sum_j  (delta_ij / ||delta_ij||) * (rho_0 / (||delta_ij|| + eps_len))^n
with delta_ij = u_i - u_j. The direction is the unit separation vector;
the magnitude is w at separation ~rho_0 and grows as (rho_0/r)^n as agents
approach. Thus:
    w  (weight) -- repulsion strength, directly comparable to spring forces;
    n  (power)  -- decay sharpness (larger n = shorter-range, steeper);
    rho_0       -- interaction cut-off radius (only nearby agents repel).
w = 0 disables repulsion (control condition).
"""
from __future__ import annotations

import numpy as np


class Repulsion:
    def __init__(self, weight: float, power: float = 2.0,
                 rho_0: float = np.inf, eps_len_frac: float = 0.05) -> None:
        self.weight = float(weight)
        self.power = float(power)
        self.rho_0 = float(rho_0)
        self.eps_len = eps_len_frac * (self.rho_0 if np.isfinite(self.rho_0) else 1.0)

    def force(self, pos_i: np.ndarray, others: np.ndarray) -> np.ndarray:
        if self.weight == 0.0 or len(others) == 0:
            return np.zeros_like(pos_i)
        deltas = pos_i - others
        dists = np.linalg.norm(deltas, axis=1)
        mask = (dists < self.rho_0) & (dists > 1e-9)
        if not np.any(mask):
            return np.zeros_like(pos_i)
        deltas = deltas[mask]
        dists = dists[mask]
        directions = deltas / dists[:, None]
        magnitudes = self.weight * (self.rho_0 / (dists + self.eps_len)) ** self.power
        return np.sum(directions * magnitudes[:, None], axis=0)

    def swarm_forces(self, positions: np.ndarray) -> np.ndarray:
        """Repulsion force on every agent. positions (N,2) -> (N,2"""
        n = positions.shape[0]
        out = np.zeros_like(positions)
        if self.weight == 0.0:
            return out
        for i in range(n):
            out[i] = self.force(positions[i], np.delete(positions, i, axis=0))
        return out
