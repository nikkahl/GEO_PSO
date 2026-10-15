from __future__ import annotations

import io
import math
import os
import sys
import time
import urllib.request
import numpy as np
from PIL import Image

BBOX = (23.30, 48.80, 23.60, 49.00)
ZOOM = 12                 
GRID_N = 90               
CACHE_PATH = os.path.join("results", "data", "terrain_cache.npz")

URL_TEMPLATES = [
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
    "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png",
    "https://s3.dualstack.us-east-1.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
]

def deg2tile(lat, lon, z):
    n = 2 ** z
    xt = (lon + 180.0) / 360.0 * n
    yt = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return xt, yt


def fetch_tile(x, y, z) -> np.ndarray:
    """Download one terrarium tile and decode it to elevation [m] (256x256)."""
    last_err = None
    for tmpl in URL_TEMPLATES:
        url = tmpl.format(z=z, x=x, y=y)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "geo-pso/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")
            arr = np.asarray(img, dtype=float)
            return (arr[..., 0] * 256 + arr[..., 1] + arr[..., 2] / 256) - 32768
        except Exception as e:  
            last_err = e
    raise RuntimeError(f"Could not download tile z{z}/{x}/{y}: {last_err}")


def main():
    west, south, east, north = BBOX
    z = ZOOM
    x0, y0 = deg2tile(north, west, z)
    x1, y1 = deg2tile(south, east, z)
    xmin, xmax = int(math.floor(min(x0, x1))), int(math.floor(max(x0, x1)))
    ymin, ymax = int(math.floor(min(y0, y1))), int(math.floor(max(y0, y1)))
    nx_t, ny_t = xmax - xmin + 1, ymax - ymin + 1
    total = nx_t * ny_t
    print(f"Area {BBOX}  zoom {z}  ->  {total} tiles ({nx_t} x {ny_t})")
    if total > 100:
        print("Too many tiles; lower ZOOM or shrink BBOX.", file=sys.stderr)
        sys.exit(1)

    big = np.zeros((ny_t * 256, nx_t * 256), dtype=float)
    done = 0
    for ty in range(ymin, ymax + 1):
        for tx in range(xmin, xmax + 1):
            tile = fetch_tile(tx, ty, z)
            iy, ix = (ty - ymin) * 256, (tx - xmin) * 256
            big[iy:iy + 256, ix:ix + 256] = tile
            done += 1
            print(f"  tile {done}/{total}", end="\r")
            time.sleep(0.05)
    print()

    n = 2 ** z
    lon_left = xmin / n * 360.0 - 180.0
    lon_right = (xmax + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ymin / n))))
    lat_bot = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ymax + 1) / n))))
    lons = np.linspace(lon_left, lon_right, big.shape[1])
    lats = np.linspace(lat_top, lat_bot, big.shape[0])  

    cx = (lons >= west) & (lons <= east)
    cy = (lats >= south) & (lats <= north)
    Zc = big[np.ix_(cy, cx)]
    lons_c, lats_c = lons[cx], lats[cy]
    if Zc.size == 0:
        print("Empty crop; check BBOX.", file=sys.stderr); sys.exit(1)

    def resample(A, n_out):
        ys = np.linspace(0, A.shape[0] - 1, n_out).astype(int)
        xs = np.linspace(0, A.shape[1] - 1, n_out).astype(int)
        return A[np.ix_(ys, xs)]
    Zg = resample(Zc, GRID_N)
    lons_g = np.linspace(lons_c.min(), lons_c.max(), GRID_N)
    lats_g = np.linspace(lats_c.min(), lats_c.max(), GRID_N)

    k = np.array([0.25, 0.5, 0.25])
    for _ in range(2):
        Zg = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, Zg)
        Zg = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, Zg)

    lat0 = float(np.mean(lats_g))
    m_lat = 111_320.0
    m_lon = 111_320.0 * math.cos(math.radians(lat0))
    x = (lons_g - lons_g.min()) * m_lon
    y = (lats_g - lats_g.min()) * m_lat
    if y[0] > y[-1]:
        y = y[::-1]
        Zg = Zg[::-1, :]
    X, Y = np.meshgrid(x, y)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(CACHE_PATH, X=X, Y=Y, Z=Zg.astype(float),
                        source=f"terrarium_aws:z{z}:{BBOX}")
    relief = float(Zg.max() - Zg.min())
    print(f"Saved {CACHE_PATH}")
    print(f"Grid {Zg.shape}, extent {X.max():.0f} x {Y.max():.0f} m, relief {relief:.0f} m")
    print("Now run:  python run_experiment.py --smoke")


if __name__ == "__main__":
    main()
