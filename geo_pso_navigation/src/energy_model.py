"""
energy_model.py
Work and energy bookkeeping, following the board derivation and the
professor's energy metric.

Per-step resistive force along the slope (board):
    C = gamma*m*g*cos(theta) + m*g*sin(theta)
with the local incline angle theta given by tan(theta) = ||grad f||
(so sin(theta) = z/d_geo over a surface step). Work increment over a step
of surface length ds:
    dA = C * ds,     A_cum = sum_t C_t * ds_t
Reference work used to normalise the objective's work term:
    A_geo = m*g*d_geo            (gravitational work scale over the geodesic distance)

Dissipated mechanical energy (professor, particle_v2.py):
    E_mech = sum_t gamma * ||v||^2 * dt
Energy efficiency (professor), re-expressed with the GEODESIC progress:
    eta = E_mech / (D_geo^0 - D_geo^min)
(smaller eta = less energy spent per unit of progress toward the target).
"""
from __future__ import annotations

import numpy as np


def resistive_force_C(grad_f: np.ndarray, gamma: float, mass: float, g: float) -> float:
    """C = gamma*m*g*cos(theta) + m*g*sin(theta), tan(theta) = ||grad f||."""
    slope = float(np.linalg.norm(grad_f))
    denom = np.sqrt(1.0 + slope * slope)
    cos_theta = 1.0 / denom
    sin_theta = slope / denom
    return gamma * mass * g * cos_theta + mass * g * sin_theta


def work_increment(grad_f: np.ndarray, step_vec: np.ndarray, gamma: float,
                   mass: float, g: float) -> float:
    """dA = C * ds, where ds is the true 3D length of this step."""
    C = resistive_force_C(grad_f, gamma, mass, g)
    planar = float(np.linalg.norm(step_vec))
    dz = float(grad_f @ step_vec)
    ds_3d = float(np.hypot(planar, dz))
    return C * ds_3d


def reference_work(d_geo_start: float, mass: float, g: float) -> float:
    """A_geo = m*g*d_geo : gravitational work scale over the geodesic distance."""
    return mass * g * max(d_geo_start, 1e-9)


def dissipation_increment(velocity: np.ndarray, gamma: float, delta_t: float) -> float:
    """dE_mech = gamma * ||v||^2 * dt (Rayleigh-dissipated mechanical energy)."""
    return gamma * float(velocity @ velocity) * delta_t


def energy_efficiency(e_mech: float, d_geo_start: float, d_geo_min: float) -> float:
    """eta = E_mech / (D_geo^0 - D_geo^min); smaller is better."""
    progress = d_geo_start - d_geo_min
    return e_mech / (progress + 1e-5)
