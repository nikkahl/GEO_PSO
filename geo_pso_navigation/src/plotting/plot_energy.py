"""
plotting/plot_energy.py
Energy and work figures (per run), each saved to its own file:
  - cumulative work, dissipated energy, kinetic energy vs iteration
  - per-step work and dissipation increments vs iteration
English labels, coordinate grid.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from src.plotting.style import apply_style
apply_style()


def plot_energy_cumulative(step_history, out_path, title_suffix=""):
    work = np.cumsum(step_history["work"])
    emech = np.cumsum(step_history["e_mech"])
    kin = np.cumsum(step_history["kinetic"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(work, label="cumulative work  W = Σ C·ds", color="tab:blue")
    ax.plot(emech, label="dissipated energy  E_mech = Σ γ|v|²Δt", color="tab:green")
    ax.plot(kin, label="accumulated kinetic energy", color="tab:orange")
    ax.set_xlabel("iteration"); ax.set_ylabel("energy, J")
    ax.set_title(f"Swarm energy dynamics {title_suffix}".strip())
    ax.legend(fontsize=8)
    fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_energy_increments(step_history, out_path, title_suffix=""):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)
    a1.plot(step_history["work"], color="tab:blue", lw=0.9)
    a1.set_ylabel("work increment dA, J")
    a1.set_title(f"Per-step work and dissipation {title_suffix}".strip())
    a2.plot(step_history["e_mech"], color="tab:green", lw=0.9)
    a2.set_ylabel("dissipation dE_mech, J"); a2.set_xlabel("iteration")
    fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_curvature_repulsion(step_history, out_path, title_suffix=""):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)
    a1.plot(step_history["curvature"], color="purple", lw=0.9)
    a1.set_ylabel("curvature  s = vᵀ H v")
    a1.set_title(f"Curvature and repulsion {title_suffix}".strip())
    a2.plot(step_history["repulsion_mag"], color="darkorange", lw=0.9)
    a2.set_ylabel("mean repulsion force, N"); a2.set_xlabel("iteration")
    fig.savefig(out_path); plt.close(fig)
    return out_path
