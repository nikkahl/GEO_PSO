"""
plotting/plot_terrain.py
Terrain-and-trajectory figures (per run):
  - 3D terrain surface with swarm trajectories
  - top-down elevation contour map with trajectories and a coordinate grid
  - geodesic distance-to-target field with the target-pull direction
All labels in English; a coordinate grid is drawn on every panel.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from src.plotting.style import apply_style
apply_style()


def _agent_label(n):  # title suffix with agent count
    return f"N = {n} agents"


def plot_terrain_3d(terrain, trajectories, start_xy, target_xy, n_agents, out_path):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, terrain.Z.shape[0] // 120)
    ax.plot_surface(terrain.X[::step, ::step], terrain.Y[::step, ::step],
                    terrain.Z[::step, ::step], cmap="terrain", alpha=0.75,
                    linewidth=0, antialiased=True)
    for tr in trajectories:
        z = terrain.elevation(tr) + 2.0
        ax.plot(tr[:, 0], tr[:, 1], z, color="crimson", lw=0.6, alpha=0.7)
    ax.scatter(*start_xy, terrain.elevation1(start_xy) + 10, color="blue", s=55,
               marker="o", label="start")
    ax.scatter(*target_xy, terrain.elevation1(target_xy) + 10, color="lime", s=90,
               marker="*", label="target")
    ax.set_xlabel("x, m"); ax.set_ylabel("y, m"); ax.set_zlabel("elevation, m")
    ax.set_title(f"Terrain surface and swarm trajectories ({_agent_label(n_agents)})")
    ax.legend(loc="upper left")
    fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_terrain_topdown(terrain, trajectories, start_xy, target_xy, n_agents, out_path):
    fig, ax = plt.subplots(figsize=(6.6, 6))
    cs = ax.contourf(terrain.X, terrain.Y, terrain.Z, levels=25, cmap="terrain", alpha=0.9)
    ax.contour(terrain.X, terrain.Y, terrain.Z, levels=25, colors="k", linewidths=0.2, alpha=0.4)
    for tr in trajectories:
        ax.plot(tr[:, 0], tr[:, 1], color="crimson", lw=0.6, alpha=0.7)
    ax.scatter(*start_xy, color="blue", s=65, marker="o", zorder=5, label="start")
    ax.scatter(*target_xy, color="lime", s=110, marker="*", zorder=5, label="target")
    fig.colorbar(cs, ax=ax, label="elevation, m")
    ax.set_xlabel("x, m"); ax.set_ylabel("y, m")
    ax.set_title(f"Top-down terrain and trajectories ({_agent_label(n_agents)})")
    ax.set_aspect("equal"); ax.legend(loc="upper right")
    ax.grid(True, which="both", color="white", alpha=0.3)
    fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_geodesic_field(terrain, geo_field, start_xy, target_xy, out_path):
    fig, ax = plt.subplots(figsize=(6.6, 6))
    cs = ax.contourf(terrain.X, terrain.Y, geo_field.D, levels=25, cmap="magma_r")
    ax.contour(terrain.X, terrain.Y, geo_field.D, levels=25, colors="white", linewidths=0.2, alpha=0.5)
    # target-pull direction arrows (-grad D_geo)
    xs = np.linspace(*terrain.bounds[:2], 9)[1:-1]
    ys = np.linspace(*terrain.bounds[2:], 9)[1:-1]
    L = 0.05 * terrain.tile_size
    for xx in xs:
        for yy in ys:
            g = geo_field.gradient1(np.array([xx, yy]))
            nrm = np.linalg.norm(g)
            if nrm > 0:
                g = g / nrm
                ax.arrow(xx, yy, -g[0] * L, -g[1] * L, head_width=0.4 * L,
                         head_length=0.4 * L, fc="cyan", ec="cyan", alpha=0.8, lw=0.5)
    ax.scatter(*start_xy, color="cyan", s=65, marker="o", zorder=5, label="start")
    ax.scatter(*target_xy, color="lime", s=110, marker="*", zorder=5, label="target")
    fig.colorbar(cs, ax=ax, label="geodesic distance to target, m")
    ax.set_xlabel("x, m"); ax.set_ylabel("y, m")
    ax.set_title("Geodesic distance field and target-pull direction")
    ax.set_aspect("equal"); ax.legend(loc="upper right")
    fig.savefig(out_path); plt.close(fig)
    return out_path
