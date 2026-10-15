"""
plotting/plot_convergence.py
Convergence figures (per run):
  - geodesic distance to target vs iteration (closest approach)
  - objective value f_i vs iteration
English labels, coordinate grid.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from src.plotting.style import apply_style
apply_style()


def plot_convergence(convergence_history, out_path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(convergence_history, color="darkred", lw=1.2)
    ax.set_xlabel("iteration"); ax.set_ylabel("geodesic distance to target, m")
    ax.set_title(f"Convergence {title_suffix}".strip())
    fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_objective(objective_history, out_path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(objective_history, color="teal", lw=1.2)
    ax.set_xlabel("iteration"); ax.set_ylabel("objective value f (global best)")
    ax.set_title(f"Objective evolution {title_suffix}".strip())
    fig.savefig(out_path); plt.close(fig)
    return out_path
