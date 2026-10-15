"""
swarm_metrics.py
All dependent variables (metrics) collected per run, grouped as:
  navigation : final geodesic distance, converged flag, steps
  energy     : cumulative work, dissipated energy, energy efficiency
  safety    : min inter-agent gap (p5), near-miss rate
  collision  : overlap count/time (agents closer than 2*radius)
  swarm     : cohesion (mean/median/p95), path length
  dynamics  : mean/max speed, curvature term
Everything is returned as one flat dict (one row of the Excel 'all_runs'
sheet) plus optional time-series kept for plotting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import numpy as np

from src.particle import Particle
from src.energy_model import energy_efficiency


@dataclass
class RunRecord:
    """One row of results: configuration + all metrics."""
    row: dict[str, Any]
    convergence_history: List[float] = field(default_factory=list)
    objective_history: List[float] = field(default_factory=list)
    step_history: dict[str, List[float]] = field(default_factory=dict)


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def inter_agent_gaps(positions: np.ndarray, radius: float) -> np.ndarray:
    """Surface-to-surface gaps between all agent pairs."""
    n = len(positions)
    gaps = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j])) - 2 * radius
            gaps.append(d)
    return np.asarray(gaps)


def compute_metrics(swarm: List[Particle], g_best_distance: float, converged: bool,
                    steps: int, exec_time: float, agent_radius: float,
                    near_miss_threshold: float,
                    gap_history: List[np.ndarray],
                    overlap_steps: int, delta_t: float) -> dict[str, Any]:
    """Aggregate all metrics for one finished run into a flat dict."""
    positions = np.array([p.position for p in swarm])
    centroid = positions.mean(axis=0)
    cohesion = np.linalg.norm(positions - centroid, axis=1)

    path_lengths = [float(np.sum(np.linalg.norm(np.diff(np.array(p.trajectory), axis=0), axis=1)))
                    for p in swarm]

    etas = [energy_efficiency(p.e_mech, p.d_geo_start, p.d_geo_min) for p in swarm]

    min_gaps = [float(np.min(g)) for g in gap_history if len(g)]
    all_gaps = np.concatenate(gap_history) if gap_history else np.array([])
    near_miss = float(np.mean(all_gaps < near_miss_threshold)) if len(all_gaps) else 0.0

    return {
        "final_geodesic_distance_m": float(g_best_distance),
        "converged": bool(converged),
        "steps": int(steps),
        "work_cum_mean_J": float(np.mean([p.work_cum for p in swarm])),
        "e_mech_mean_J": float(np.mean([p.e_mech for p in swarm])),
        "e_mech_total_J": float(np.sum([p.e_mech for p in swarm])),
        "energy_efficiency_mean": float(np.mean(etas)),
        "energy_efficiency_median": float(np.median(etas)),
        "safety_gap_min_m": float(np.min(min_gaps)) if min_gaps else float("nan"),
        "safety_gap_p5_m": _pct(min_gaps, 5),
        "near_miss_rate": near_miss,
        "overlap_steps": int(overlap_steps),
        "overlap_time_s": float(overlap_steps * delta_t),
        "cohesion_mean_m": float(np.mean(cohesion)),
        "cohesion_median_m": float(np.median(cohesion)),
        "cohesion_p95_m": _pct(cohesion, 95),
        "path_length_mean_m": float(np.mean(path_lengths)),
        "speed_mean_mps": float(np.mean([p.speed for p in swarm])),
        "speed_max_mps": float(np.max([p.speed for p in swarm])),
        "curvature_mean": float(np.mean([abs(p.curvature) for p in swarm])),
        "repulsion_force_mean_N": float(np.mean([p.repulsion_mag for p in swarm])),
        "exec_time_s": float(exec_time),
    }
