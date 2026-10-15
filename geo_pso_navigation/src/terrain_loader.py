"""
terrain_loader.py
Loads a REAL terrain surface f(x, y) from real geographic data. There is
NO synthetic terrain generation anywhere in this module: the elevation
values come from a real contour-line shapefile (.shp) or a real DEM
raster (.tif).

Two supported sources (see config.TERRAIN_KIND):

  1. "contours_shp" -- a shapefile of elevation CONTOUR LINES carrying an
     elevation attribute. Points are sampled along the real contour lines
     at their real elevations and interpolated onto a regular grid
     (scipy.griddata). This is the standard GIS "surface from contours"
     reconstruction, not terrain generation: every elevation is measured
     real data, only the values between contours are interpolated.

  2. "dem_tif" -- a DEM raster read directly with rasterio; elevations are
     taken as-is.

Both return the SAME `Terrain` object, exposing f(u), grad f(u) and the
Hessian H(u), so the rest of the pipeline is identical regardless of the
source or the map region.

Dependencies for reading real files (install on your machine):
    pip install geopandas shapely scipy        # for .shp contours
    pip install rasterio                         # for .tif DEM
numpy/scipy are always required.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

_ELEV_FIELD_CANDIDATES = [
    "ELEV", "elev", "Elevation", "ELEVATION", "elevation",
    "HEIGHT", "height", "CONTOUR", "contour", "LEVEL", "level",
    "ALT", "alt", "Z", "z", "VALUE", "value", "DN",
]


@dataclass
class Terrain:
    """Continuous differentiable terrain surface f(x, y) on a metric grid.

    X, Y: (ny, nx) metric meshgrids [m]; Z: (ny, nx) elevation [m].
    """
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    source: str = "unknown"

    def __post_init__(self) -> None:
        self.x_coords = self.X[0, :]
        self.y_coords = self.Y[:, 0]
        self._f = RegularGridInterpolator(
            (self.y_coords, self.x_coords), self.Z, bounds_error=False, fill_value=None)
        # Precompute gradient and Hessian fields on the grid, then interpolate
        # them -- far cheaper than finite-differencing on every per-step query.
        dz_dy, dz_dx = np.gradient(self.Z, self.y_coords, self.x_coords)
        self._gx = RegularGridInterpolator((self.y_coords, self.x_coords), dz_dx,
                                           bounds_error=False, fill_value=None)
        self._gy = RegularGridInterpolator((self.y_coords, self.x_coords), dz_dy,
                                           bounds_error=False, fill_value=None)
        d2y, d2yx = np.gradient(dz_dy, self.y_coords, self.x_coords)
        d2xy, d2x = np.gradient(dz_dx, self.y_coords, self.x_coords)
        self._hxx = RegularGridInterpolator((self.y_coords, self.x_coords), d2x,
                                            bounds_error=False, fill_value=None)
        self._hyy = RegularGridInterpolator((self.y_coords, self.x_coords), d2y,
                                            bounds_error=False, fill_value=None)
        self._hxy = RegularGridInterpolator((self.y_coords, self.x_coords),
                                            0.5 * (d2xy + d2yx),
                                            bounds_error=False, fill_value=None)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (float(self.x_coords.min()), float(self.x_coords.max()),
                float(self.y_coords.min()), float(self.y_coords.max()))

    @property
    def tile_size(self) -> float:
        x0, x1, y0, y1 = self.bounds
        return float(min(x1 - x0, y1 - y0))

    def elevation1(self, xy: np.ndarray) -> float:
        return float(self._f(np.asarray(xy).reshape(1, 2)[:, ::-1])[0])

    def elevation(self, xy: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(xy)
        return self._f(p[:, ::-1])

    def gradient1(self, xy: np.ndarray) -> np.ndarray:
        p = np.asarray(xy).reshape(1, 2)[:, ::-1]
        return np.array([float(self._gx(p)[0]), float(self._gy(p)[0])])

    def hessian1(self, xy: np.ndarray) -> np.ndarray:
        p = np.asarray(xy).reshape(1, 2)[:, ::-1]
        hxx, hyy, hxy = float(self._hxx(p)[0]), float(self._hyy(p)[0]), float(self._hxy(p)[0])
        return np.array([[hxx, hxy], [hxy, hyy]])


def _to_metric_grid(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray,
                    is_geographic: bool, source: str) -> Terrain:
    """Build a Terrain, converting lon/lat degrees to metres if needed."""
    if is_geographic:
        lat0 = float(np.mean(ys))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
        x = (xs - xs.min()) * m_per_deg_lon
        y = (ys - ys.min()) * m_per_deg_lat
    else:
        x = xs - xs.min()
        y = ys - ys.min()
    X, Y = np.meshgrid(x, y)
    return Terrain(X=X, Y=Y, Z=np.asarray(Z, dtype=float), source=source)


def load_terrain_from_contours_shp(path: str, grid_n: int = 80,
                                   elev_field: str | None = None,
                                   bbox: tuple | None = None) -> Terrain:
    """Reconstruct an elevation surface from a REAL contour-line shapefile.

    Samples points along the contour lines (each carrying its real
    elevation), interpolates them onto a regular grid, and returns a
    Terrain. The interpolation fills values BETWEEN measured contours; it
    does not invent terrain.
    """
    import geopandas as gpd           # local import: optional heavy dependency
    from shapely.geometry import box
    from scipy.interpolate import griddata

    gdf = gpd.read_file(path)
    if bbox is not None:
        gdf = gpd.clip(gdf, box(*bbox))
    if gdf.empty:
        raise ValueError("No contour features found (check path / bbox).")

    # Resolve the elevation attribute.
    field = elev_field
    if field is None:
        for cand in _ELEV_FIELD_CANDIDATES:
            if cand in gdf.columns:
                field = cand
                break
    if field is None or field not in gdf.columns:
        raise ValueError(
            f"Could not find an elevation field in {list(gdf.columns)}. "
            f"Set TERRAIN_ELEV_FIELD in config.py to the right column name.")

    is_geographic = bool(getattr(gdf.crs, "is_geographic", False)) if gdf.crs else True

    # Sample points along every contour line at its elevation.
    pts_x, pts_y, pts_z = [], [], []
    for geom, elev in zip(gdf.geometry, gdf[field]):
        if geom is None or elev is None:
            continue
        lines = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for line in lines:
            coords = np.asarray(line.coords)
            if coords.ndim != 2 or len(coords) == 0:
                continue
            pts_x.extend(coords[:, 0])
            pts_y.extend(coords[:, 1])
            pts_z.extend([float(elev)] * len(coords))
    pts_x = np.asarray(pts_x); pts_y = np.asarray(pts_y); pts_z = np.asarray(pts_z)
    if len(pts_z) < 4:
        raise ValueError("Too few contour vertices to build a surface.")

    # Target regular grid over the data extent.
    x_lin = np.linspace(pts_x.min(), pts_x.max(), grid_n)
    y_lin = np.linspace(pts_y.min(), pts_y.max(), grid_n)
    GX, GY = np.meshgrid(x_lin, y_lin)
    Z = griddata((pts_x, pts_y), pts_z, (GX, GY), method="linear")
    # Fill outside the convex hull with nearest-contour elevation.
    nan = np.isnan(Z)
    if nan.any():
        Zn = griddata((pts_x, pts_y), pts_z, (GX, GY), method="nearest")
        Z[nan] = Zn[nan]

    # Light smoothing so the Hessian (surface curvature) stays well-behaved.
    from scipy.ndimage import gaussian_filter
    Z = gaussian_filter(Z, sigma=1.0)

    return _to_metric_grid(x_lin, y_lin, Z, is_geographic,
                           source=f"contours_shp:{os.path.basename(path)}:{field}")


def load_terrain_from_dem_tif(path: str, downsample: int = 1) -> Terrain:
    """Load a REAL DEM raster (.tif) directly with rasterio."""
    import rasterio
    with rasterio.open(path) as ds:
        Z = ds.read(1).astype(float)[::downsample, ::downsample]
        ny, nx = Z.shape
        cols, rows = np.meshgrid(np.arange(nx) * downsample, np.arange(ny) * downsample)
        xs, ys = rasterio.transform.xy(ds.transform, rows, cols)
        xs = np.asarray(xs)[0, :]
        ys = np.asarray(ys)[:, 0]
        is_geographic = bool(ds.crs.is_geographic) if ds.crs else False
    from scipy.ndimage import gaussian_filter
    Z = gaussian_filter(Z, sigma=1.0)
    return _to_metric_grid(xs, ys, Z, is_geographic, source=f"dem_tif:{os.path.basename(path)}")


def load_terrain(kind: str, path: str, grid_n: int = 80,
                 elev_field: str | None = None, bbox: tuple | None = None,
                 cache_path: str | None = None, use_cache: bool = True) -> Terrain:
    """Top-level loader: reads REAL terrain, caches the reconstructed grid.

    Raises a clear error if the file is missing -- there is intentionally
    no synthetic fallback, so a run always uses real elevation data.
    """
    if use_cache and cache_path and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        print(f"[terrain] using cached real grid: {cache_path}")
        return Terrain(X=d["X"], Y=d["Y"], Z=d["Z"], source=str(d["source"]))

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Terrain file not found: {path}\n"
            f"Provide a REAL terrain file and set TERRAIN_PATH/TERRAIN_KIND in config.py.\n"
            f"See README.md -> 'Where to get real terrain'.")

    if kind == "contours_shp":
        terrain = load_terrain_from_contours_shp(path, grid_n=grid_n, elev_field=elev_field, bbox=bbox)
    elif kind == "dem_tif":
        terrain = load_terrain_from_dem_tif(path)
    else:
        raise ValueError(f"Unknown TERRAIN_KIND: {kind!r} (use 'contours_shp' or 'dem_tif').")

    if use_cache and cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=terrain.X, Y=terrain.Y, Z=terrain.Z, source=terrain.source)
        print(f"[terrain] cached reconstructed grid to {cache_path} (delete it to reload).")
    return terrain
