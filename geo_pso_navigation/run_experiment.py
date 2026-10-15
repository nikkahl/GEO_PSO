"""
run_experiment.py
Main entry point. Loads the REAL terrain, runs the full factorial study
(all objective modes x parameter grid x seeds), writes every run to Excel,
and generates per-run plots (reference seed) plus study-level plots.

Usage:
    pip install -r requirements.txt
    python run_experiment.py            # full grid from config.py
    python run_experiment.py --smoke    # tiny grid, fast sanity check

Outputs:
    results/data/all_results.xlsx       (all_runs, config_summary,
                                         best_combinations, convergence_histories)
    results/per_run/<config_id>/*.png   (per-configuration figures)
    results/plots/*.png                 (study-level sensitivity/Pareto/comparison)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config
from src.terrain_loader import load_terrain
from src.geodesic_distance import build_geodesic_field
from src.experiment_runner import run_single, random_start_target
from src.excel_writer import write_excel
from src.plotting.per_run import generate_per_run_plots
from src.plotting.plot_study import generate_study_plots


def apply_smoke():
    config.MAX_STEPS = 500
    config.N_SEEDS = 1
    config.SEEDS = [2000]
    config.REFERENCE_SEED = 2000
    config.OBJECTIVE_MODES = ["baseline", "composite"]
    config.ALPHA_VALUES = [0.0, 1.0]
    config.K_VALUES = [100.0]
    config.GAMMA_VALUES = [0.1, 1.0]
    config.REPULSION_WEIGHT_VALUES = [0.0, 20.0]
    config.REPULSION_POWER_VALUES = [2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny grid for a fast check")
    args = ap.parse_args()
    if args.smoke:
        apply_smoke()

    grid = config.build_grid()
    n_runs = len(grid) * len(config.SEEDS)
    print(f"Study: N = {config.N_PARTICLES} agents | {len(grid)} configurations "
          f"x {len(config.SEEDS)} seeds = {n_runs} runs")

    terrain = load_terrain(
        kind=config.TERRAIN_KIND, path=config.TERRAIN_PATH,
        grid_n=config.TERRAIN_GRID_N, elev_field=config.TERRAIN_ELEV_FIELD,
        bbox=config.TERRAIN_BBOX, cache_path=config.TERRAIN_CACHE)
    print(f"Terrain: {terrain.source} | relief {terrain.Z.max()-terrain.Z.min():.0f} m "
          f"over {terrain.bounds[1]-terrain.bounds[0]:.0f} x {terrain.bounds[3]-terrain.bounds[2]:.0f} m")

    seed_setup = {}
    for seed in config.SEEDS:
        rng = np.random.default_rng(seed)
        s, t = random_start_target(terrain, rng)
        seed_setup[seed] = (s, t, build_geodesic_field(terrain, t))

    records = []
    for seed in config.SEEDS:
        start_xy, target_xy, geo_field = seed_setup[seed]
        is_ref = (seed == config.REFERENCE_SEED)
        for cfg in grid:
            rec = run_single(terrain, geo_field, start_xy, target_xy, cfg, seed,
                             record_history=(is_ref and config.PER_CONFIG_PLOTS))
            records.append(rec)
            r = rec.row
            print(f"[seed {seed}] {r['config_id']}: converged={r['converged']} "
                  f"steps={r['steps']} d_final={r['final_geodesic_distance_m']:.1f} m "
                  f"W={r['work_cum_mean_J']:.1f} J eta={r['energy_efficiency_mean']:.2f}")
            if is_ref and config.PER_CONFIG_PLOTS and hasattr(rec, "swarm"):
                generate_per_run_plots(rec, terrain, geo_field, config.PER_RUN_DIR, config.N_PARTICLES)
                del rec.swarm  

    path = write_excel(records, config.EXCEL_PATH)
    print(f"\nExcel written: {path} ({len(records)} rows in all_runs)")

    all_df = pd.DataFrame([r.row for r in records])
    study_paths = generate_study_plots(all_df, config.PLOTS_DIR)
    print(f"Study-level plots: {len(study_paths)} figures in {config.PLOTS_DIR}/")

    from src.excel_writer import _summarise, _best_combinations
    best = _best_combinations(_summarise(all_df)).head(1)
    if len(best):
        b = best.iloc[0]
        print(f"\nBest combination: {b['config_id']} | "
              f"converged_fraction={b['converged_fraction']:.2f} | "
              f"median d_final={b['median_final_geodesic_distance_m']:.1f} m | "
              f"median eta={b['median_energy_efficiency_mean']:.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
