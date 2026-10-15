"""
plotting/plot_phase.py
Phase-portrait figures (per run): position vs velocity, the dynamical-
systems view used to see whether the swarm settles or keeps oscillating.
  - (x, v_x) phase portrait for a sample of agents
  - speed vs geodesic distance to target (approach dynamics)
English labels, coordinate grid.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from src.plotting.style import apply_style
apply_style()


def plot_phase_xvx(swarm, out_path, max_agents=8, title_suffix=""):
    fig, ax = plt.subplots(figsize=(6, 5))
    for p in swarm[:max_agents]:
        tr = np.array(p.trajectory)
        vt = np.array(p.velocity_trajectory)
        ax.plot(tr[:, 0], vt[:, 0], lw=0.6, alpha=0.8)
    ax.set_xlabel("position x, m"); ax.set_ylabel("velocity v_x, m/s")
    ax.set_title(f"Phase portrait (x, v_x) {title_suffix}".strip())
    fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_speed_vs_distance(swarm, geo_field, out_path, max_agents=8, title_suffix=""):
    fig, ax = plt.subplots(figsize=(6, 5))
    for p in swarm[:max_agents]:
        tr = np.array(p.trajectory)
        vt = np.array(p.velocity_trajectory)
        speed = np.linalg.norm(vt, axis=1)
        dist = np.array([geo_field.distance1(xy) for xy in tr])
        ax.plot(dist, speed, lw=0.6, alpha=0.8)
    ax.set_xlabel("geodesic distance to target, m"); ax.set_ylabel("speed |v|, m/s")
    ax.set_title(f"Approach dynamics (speed vs distance) {title_suffix}".strip())
    ax.invert_xaxis()
    fig.savefig(out_path); plt.close(fig)
    return out_path
