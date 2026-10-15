"""
geodesic_distance.py
Geodesic (on-surface) distance to a fixed target, used as the navigation
criterion. Builds an 8-connected grid graph over the terrain where each
edge weight is the true 3D segment length (so climbing a slope costs more
than moving on the flat), then runs a single-source Dijkstra from the
target to get the geodesic distance D_geo(u) from every grid node at once.
During the swarm run D_geo and its gradient are read by interpolation in
O(1) per query.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import RegularGridInterpolator

from src.terrain_loader import Terrain


@dataclass
class GeodesicField:
    terrain: Terrain
    target_xy: np.ndarray
    D: np.ndarray  

    def __post_init__(self) -> None:
        t = self.terrain
        self._d = RegularGridInterpolator((t.y_coords, t.x_coords), self.D,
                                          bounds_error=False, fill_value=None)
        dDy, dDx = np.gradient(self.D, t.y_coords, t.x_coords)
        self._gx = RegularGridInterpolator((t.y_coords, t.x_coords), dDx,
                                           bounds_error=False, fill_value=None)
        self._gy = RegularGridInterpolator((t.y_coords, t.x_coords), dDy,
                                           bounds_error=False, fill_value=None)

    def distance1(self, xy: np.ndarray) -> float:
        return float(self._d(np.asarray(xy).reshape(1, 2)[:, ::-1])[0])

    def gradient1(self, xy: np.ndarray) -> np.ndarray:
        p = np.asarray(xy).reshape(1, 2)[:, ::-1]
        return np.array([float(self._gx(p)[0]), float(self._gy(p)[0])])


def _nearest_node(terrain: Terrain, xy: np.ndarray) -> int:
    ny, nx = terrain.Z.shape
    ix = int(np.clip(np.searchsorted(terrain.x_coords, xy[0]), 0, nx - 1))
    iy = int(np.clip(np.searchsorted(terrain.y_coords, xy[1]), 0, ny - 1))
    return iy * nx + ix


def build_geodesic_field(terrain: Terrain, target_xy: np.ndarray) -> GeodesicField:
    """Single-source Dijkstra from the target over the 8-connected terrain
    graph; edge weight = true 3D segment length. Computed once per
    (terrain, target) pair."""
    ny, nx = terrain.Z.shape
    Z = terrain.Z
    dx = terrain.x_coords[1] - terrain.x_coords[0]
    dy = terrain.y_coords[1] - terrain.y_coords[0]
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    rows, cols, wts = [], [], []
    for iy in range(ny):
        for ix in range(nx):
            u = iy * nx + ix
            for oy, ox in offsets:
                jy, jx = iy + oy, ix + ox
                if 0 <= jy < ny and 0 <= jx < nx:
                    horiz = np.hypot(ox * dx, oy * dy)
                    dz = Z[jy, jx] - Z[iy, ix]
                    rows.append(u); cols.append(jy * nx + jx)
                    wts.append(np.hypot(horiz, dz))   # true 3D edge length
    graph = csr_matrix((wts, (rows, cols)), shape=(ny * nx, ny * nx))
    d1 = dijkstra(graph, directed=False, indices=_nearest_node(terrain, target_xy))
    return GeodesicField(terrain=terrain, target_xy=np.asarray(target_xy, float),
                         D=d1.reshape(ny, nx))
