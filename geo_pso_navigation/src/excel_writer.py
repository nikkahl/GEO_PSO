"""
excel_writer.py
Writes ALL results to one .xlsx workbook with four sheets:
  all_runs              : one row per (configuration, seed) -- every run, every metric
  config_summary        : one row per configuration, median across seeds
  best_combinations     : configurations ranked by a composite quality score
  convergence_histories : geodesic-distance-vs-iteration curve for every run
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from src.swarm_metrics import RunRecord

CONFIG_KEYS = ["mode", "alpha", "k", "gamma", "repulsion_weight", "repulsion_power"]
SUMMARY_METRICS = [
    "final_geodesic_distance_m", "steps", "work_cum_mean_J", "e_mech_mean_J",
    "energy_efficiency_mean", "energy_efficiency_median", "safety_gap_p5_m",
    "near_miss_rate", "overlap_time_s", "cohesion_mean_m", "cohesion_p95_m",
    "path_length_mean_m", "speed_mean_mps", "speed_max_mps", "curvature_mean",
    "repulsion_force_mean_N", "exec_time_s",
]


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(CONFIG_KEYS, dropna=False):
        row = dict(zip(CONFIG_KEYS, keys))
        row["config_id"] = g["config_id"].iloc[0]
        row["n_seeds"] = len(g)
        row["converged_fraction"] = float(g["converged"].mean())
        for m in SUMMARY_METRICS:
            row[f"median_{m}"] = float(np.median(g[m]))
        rows.append(row)
    return pd.DataFrame(rows)


def _best_combinations(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank configurations by a quality score that rewards reaching the
    target with low energy. Only configurations that converge in the
    majority of seeds are ranked; others are listed below them.

    score = converged_fraction / (1 + norm_distance + norm_efficiency)
    (higher = better: close to target AND energy-efficient).
    """
    s = summary.copy()
    d = s["median_final_geodesic_distance_m"].to_numpy()
    e = s["median_energy_efficiency_mean"].to_numpy()
    dn = (d - d.min()) / (d.max() - d.min() + 1e-9)
    en = (e - e.min()) / (e.max() - e.min() + 1e-9)
    s["quality_score"] = s["converged_fraction"] / (1.0 + dn + en)
    s = s.sort_values(["converged_fraction", "quality_score"], ascending=[False, False])
    return s


def write_excel(records: List[RunRecord], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    all_df = pd.DataFrame([r.row for r in records])
    summary_df = _summarise(all_df)
    best_df = _best_combinations(summary_df)

    max_len = max((len(r.convergence_history) for r in records), default=0)
    hist_rows = []
    for r in records:
        h = np.asarray(r.convergence_history, float)
        if len(h) < max_len:
            h = np.pad(h, (0, max_len - len(h)), mode="edge")
        row = {"config_id": r.row["config_id"], "seed": r.row["seed"]}
        row.update({f"iter_{i}": v for i, v in enumerate(h)})
        hist_rows.append(row)
    hist_df = pd.DataFrame(hist_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        all_df.to_excel(xw, sheet_name="all_runs", index=False)
        summary_df.to_excel(xw, sheet_name="config_summary", index=False)
        best_df.to_excel(xw, sheet_name="best_combinations", index=False)
        hist_df.to_excel(xw, sheet_name="convergence_histories", index=False)
    return path
