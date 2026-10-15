"""
plotting/per_run.py
Generates the full set of per-run figures for one configuration (reference
seed) into results/per_run/<config_id>/, with self-describing file names:
    terrain_3d.png, terrain_topdown.png, geodesic_field.png,
    convergence.png, objective.png,
    energy_cumulative.png, energy_increments.png, curvature_repulsion.png,
    phase_x_vx.png, phase_speed_vs_distance.png
"""
from __future__ import annotations
import os
import numpy as np

from src.plotting import plot_terrain, plot_energy, plot_phase, plot_convergence


def generate_per_run_plots(rec, terrain, geo_field, per_run_dir: str, n_agents: int) -> str:
    cid = rec.row["config_id"]
    out = os.path.join(per_run_dir, cid)
    os.makedirs(out, exist_ok=True)
    suffix = f"({cid})"

    swarm = rec.swarm                 
    start_xy = rec.start_xy           
    target_xy = rec.target_xy        
    trajectories = [np.array(p.trajectory) for p in swarm]

    plot_terrain.plot_terrain_3d(terrain, trajectories, start_xy, target_xy, n_agents,
                                 os.path.join(out, "terrain_3d.png"))
    plot_terrain.plot_terrain_topdown(terrain, trajectories, start_xy, target_xy, n_agents,
                                      os.path.join(out, "terrain_topdown.png"))
    plot_terrain.plot_geodesic_field(terrain, geo_field, start_xy, target_xy,
                                     os.path.join(out, "geodesic_field.png"))
    plot_convergence.plot_convergence(rec.convergence_history,
                                      os.path.join(out, "convergence.png"), suffix)
    plot_convergence.plot_objective(rec.objective_history,
                                    os.path.join(out, "objective.png"), suffix)
    plot_energy.plot_energy_cumulative(rec.step_history,
                                       os.path.join(out, "energy_cumulative.png"), suffix)
    plot_energy.plot_energy_increments(rec.step_history,
                                       os.path.join(out, "energy_increments.png"), suffix)
    plot_energy.plot_curvature_repulsion(rec.step_history,
                                         os.path.join(out, "curvature_repulsion.png"), suffix)
    plot_phase.plot_phase_xvx(swarm, os.path.join(out, "phase_x_vx.png"), title_suffix=suffix)
    plot_phase.plot_speed_vs_distance(swarm, geo_field,
                                      os.path.join(out, "phase_speed_vs_distance.png"),
                                      title_suffix=suffix)
    return out
