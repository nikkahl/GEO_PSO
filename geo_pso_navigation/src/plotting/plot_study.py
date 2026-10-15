"""
plotting/plot_study.py
Study-level figures across the whole grid, each saved to its own file:
  - parameter-sensitivity box plots (every parameter x key metrics)
  - energy-vs-distance trade-off scatter (Pareto view), coloured by alpha
  - composite vs baseline comparison (energy efficiency)
English labels, coordinate grid.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.plotting.style import apply_style
apply_style()

METRIC_LABELS = {
    "final_geodesic_distance_m": "final geodesic distance, m",
    "energy_efficiency_mean": "energy efficiency η (lower better)",
    "work_cum_mean_J": "cumulative work W, J",
    "e_mech_mean_J": "dissipated energy E_mech, J",
    "cohesion_mean_m": "swarm cohesion, m",
    "safety_gap_p5_m": "safety gap p5, m",
    "steps": "steps to convergence",
}
PARAM_LABELS = {
    "alpha": "balance weight α", "k": "stiffness k",
    "gamma": "friction γ", "repulsion_weight": "repulsion weight",
}


def _save(fig, out_path):
    fig.savefig(out_path); plt.close(fig); return out_path


def plot_sensitivity(all_df: pd.DataFrame, param: str, metric: str, out_path: str):
    sub = all_df[all_df[param].notna()]
    levels = sorted(sub[param].unique())
    data = [sub[sub[param] == lv][metric].to_numpy() for lv in levels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=[str(l) for l in levels], showmeans=True)
    ax.set_xlabel(PARAM_LABELS.get(param, param))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs {PARAM_LABELS.get(param, param)}")
    return _save(fig, out_path)


def plot_pareto(all_df: pd.DataFrame, out_path: str):
    comp = all_df[all_df["mode"] == "composite"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sc = ax.scatter(comp["final_geodesic_distance_m"], comp["energy_efficiency_mean"],
                    c=comp["alpha"], cmap="viridis", s=28, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("final geodesic distance to target, m")
    ax.set_ylabel("energy efficiency η (lower better)")
    ax.set_title("Energy-vs-distance trade-off (composite objective)")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("balance weight α")
    return _save(fig, out_path)


def plot_mode_comparison(all_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4.4))
    groups, labels = [], []
    base = all_df[all_df["mode"] == "baseline"]["energy_efficiency_mean"].to_numpy()
    if len(base):
        groups.append(base); labels.append("baseline\n(distance only)")
    for a in sorted(all_df[all_df["mode"] == "composite"]["alpha"].dropna().unique()):
        groups.append(all_df[(all_df["mode"] == "composite") & (all_df["alpha"] == a)]
                      ["energy_efficiency_mean"].to_numpy())
        labels.append(f"composite\nα={a}")
    ax.boxplot(groups, tick_labels=labels, showmeans=True)
    ax.set_ylabel("energy efficiency η (lower better)")
    ax.set_title("Energy efficiency: baseline vs composite objective")
    return _save(fig, out_path)


def generate_study_plots(all_df: pd.DataFrame, plots_dir: str) -> list:
    os.makedirs(plots_dir, exist_ok=True)
    paths = []
    sens_params = ["alpha", "k", "gamma", "repulsion_weight"]
    sens_metrics = ["final_geodesic_distance_m", "energy_efficiency_mean",
                    "work_cum_mean_J", "cohesion_mean_m", "safety_gap_p5_m", "steps"]
    idx = 0
    for pr in sens_params:
        if all_df[pr].notna().sum() == 0:
            continue
        for mt in sens_metrics:
            idx += 1
            paths.append(plot_sensitivity(all_df, pr, mt,
                         os.path.join(plots_dir, f"sensitivity_{idx:02d}_{pr}_vs_{mt}.png")))
    paths.append(plot_pareto(all_df, os.path.join(plots_dir, "pareto_energy_vs_distance.png")))
    paths.append(plot_mode_comparison(all_df, os.path.join(plots_dir, "comparison_baseline_vs_composite.png")))
    return paths
