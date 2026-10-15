"""
experiment_runner.py
Runs one configuration (one objective mode + parameter set + seed) and the
full factorial grid. Couples the study's objective function and energy
model into the physics loop, and collects all metrics per run.

Per step: move the swarm (physics), update each agent's geodesic distance
and cumulative work, evaluate the objective f_i, update personal/global
best by the objective, and record convergence (minimum geodesic distance
reached). The global-best position used by the springs is the
objective-best; convergence is measured by the true geodesic distance to
the target, so it is comparable across objective modes.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

import config
from src.terrain_loader import Terrain
from src.geodesic_distance import GeodesicField, build_geodesic_field
from src.repulsion import Repulsion
from src.physics_engine import PhysicsEngine
from src.particle import Particle, init_swarm
from src.objective_functions import evaluate, better, worst_value
from src.energy_model import reference_work
from src.swarm_metrics import RunRecord, compute_metrics, inter_agent_gaps

AGENT_RADIUS_FRAC = 0.004  


def random_start_target(terrain: Terrain, rng: np.random.Generator,
                        min_sep_frac: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = terrain.bounds
    min_sep = min_sep_frac * terrain.tile_size
    while True:
        s = rng.uniform([x0, y0], [x1, y1])
        t = rng.uniform([x0, y0], [x1, y1])
        if np.linalg.norm(t - s) >= min_sep:
            return s, t


def run_single(terrain: Terrain, geo_field: GeodesicField, start_xy: np.ndarray,
               target_xy: np.ndarray, cfg: dict, seed: int,
               record_history: bool = False) -> RunRecord:
    rng = np.random.default_rng(seed)
    tile = terrain.tile_size
    bounds = ((terrain.bounds[0], terrain.bounds[1]), (terrain.bounds[2], terrain.bounds[3]))
    mode = cfg["mode"]

    swarm = init_swarm(config.N_PARTICLES, start_xy, config.SPAWN_RADIUS_FRAC * tile,
                       config.MASS, bounds, mode, rng)

    for p in swarm:
        p.d_geo = geo_field.distance1(p.position)
        p.d_geo_start = p.d_geo
        p.d_geo_min = p.d_geo
        a_geo = reference_work(p.d_geo_start, config.MASS, config.G_ACCEL)
        p.a_geo = a_geo  
        p.objective_value = evaluate(mode, d_i=p.d_geo, d_geo=p.d_geo_start, k=cfg["k"],
                                     a_cum=max(p.work_cum, 1e-9), a_geo=a_geo, alpha=cfg["alpha"])
        p.best_objective = p.objective_value
        p.best_position = p.position.copy()

    repulsion = Repulsion(weight=cfg["repulsion_weight"], power=cfg["repulsion_power"],
                          rho_0=config.REPULSION_RHO0_FRAC * tile)
    engine = PhysicsEngine(
        terrain=terrain, geo_field=geo_field, repulsion=repulsion,
        delta_t=config.DELTA_T, g=config.G_ACCEL, gamma=cfg["gamma"],
        k1=cfg["k"], k2=cfg["k"], k_target=config.K_TARGET,
        f_max=config.F_MAX, f_max_target=config.F_MAX_TARGET, v_max=config.V_MAX, rng=rng)

    g_best = min(swarm, key=lambda p: -p.objective_value) if mode == "composite" \
        else min(swarm, key=lambda p: p.objective_value)
    g_best_position = g_best.best_position.copy()
    g_best_objective = g_best.best_objective

    closest_geo = min(p.d_geo for p in swarm)   
    agent_radius = AGENT_RADIUS_FRAC * tile
    conv_threshold = config.CONVERGENCE_FRAC * tile

    convergence_history, objective_history = [], []
    step_hist = {k: [] for k in ["work", "e_mech", "kinetic", "curvature", "repulsion_mag", "speed"]}
    gap_history: List[np.ndarray] = []
    overlap_steps = 0

    t0 = time.perf_counter()
    converged = False
    step = 0
    for step in range(1, config.MAX_STEPS + 1):
        agg = engine.step(swarm, g_best_position)
        if record_history:
            for key in step_hist:
                step_hist[key].append(agg[key])

        for p in swarm:
            p.d_geo = geo_field.distance1(p.position)
            if p.d_geo < p.d_geo_min:
                p.d_geo_min = p.d_geo
            p.objective_value = evaluate(mode, d_i=p.d_geo, d_geo=p.d_geo_start, k=cfg["k"],
                                         a_cum=max(p.work_cum, 1e-9),
                                         a_geo=p.a_geo, alpha=cfg["alpha"])  
            p.update_best()
            if better(p.best_objective, g_best_objective, mode):
                g_best_objective = p.best_objective
                g_best_position = p.best_position.copy()

        closest_geo = min(closest_geo, min(p.d_geo for p in swarm))
        convergence_history.append(closest_geo)
        objective_history.append(g_best_objective)

        positions = np.array([p.position for p in swarm])
        gaps = inter_agent_gaps(positions, agent_radius)
        gap_history.append(gaps)
        if np.any(gaps < 0):
            overlap_steps += 1

        if closest_geo <= conv_threshold:
            converged = True
            break
    exec_time = time.perf_counter() - t0

    metrics = compute_metrics(
        swarm, g_best_distance=closest_geo, converged=converged, steps=step,
        exec_time=exec_time, agent_radius=agent_radius,
        near_miss_threshold=0.5 * agent_radius, gap_history=gap_history,
        overlap_steps=overlap_steps, delta_t=config.DELTA_T)

    row = {
        "config_id": config.config_id(cfg), "seed": seed, "mode": mode,
        "alpha": cfg["alpha"], "k": cfg["k"], "gamma": cfg["gamma"],
        "repulsion_weight": cfg["repulsion_weight"], "repulsion_power": cfg["repulsion_power"],
        "is_control": cfg["is_control"],
    }
    row.update(metrics)

    rec = RunRecord(row=row, convergence_history=convergence_history,
                    objective_history=objective_history, step_history=step_hist)
    if record_history:
        rec.swarm = swarm                 
        rec.start_xy = start_xy           
        rec.target_xy = target_xy         
    return rec
