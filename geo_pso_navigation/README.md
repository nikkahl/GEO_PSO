# Geo-PSO Navigation — swarm navigation on real terrain with a composite objective

Physics-inspired particle-swarm navigation of a swarm of agents across a
**real terrain surface** from a random start point to a random target,
minimising a **composite objective** that balances geodesic closeness to
the target against the mechanical **work (energy)** spent. Follow-up study
to the benchmark-function article: here the objective is navigation on a
real digital terrain, not the minimum of an abstract function.

## What the study does

- **Real terrain only** — elevation comes from a real contour shapefile
  (`.shp`) or a real DEM raster (`.tif`). There is **no synthetic terrain
  generation** anywhere.
- **Geodesic distance** `D_geo` (shortest path along the surface) is the
  distance criterion, computed once per target by Dijkstra over the
  terrain graph.
- **Composite objective** (formula 1), maximised:
  `f_i = alpha * (d_geo*k / d_i) + (1-alpha) * (A_geo / A_cum)`,
  balancing distance (`d_i`) against cumulative work (`A_cum`) with weight
  `alpha`.
- **Baseline objective** (formula 2), minimised: `f_i = d_i` (distance
  only) — the control.
- **Energy model**: work `C = gamma*m*g*cos(theta) + m*g*sin(theta)`,
  `A_cum = sum C*ds`; dissipated energy `E_mech = sum gamma*|v|^2*dt`;
  energy efficiency `eta = E_mech / (D_geo^0 - D_geo^min)`.
- **Full factorial experiment** over `alpha, k, gamma, repulsion_weight,
  repulsion_power`, with repeated seeds; **every run is written to Excel**
  and gets its own folder of figures.

## How to run

```bash
pip install -r requirements.txt
# put a real terrain file into terrain_data/ and set its path in config.py
python run_experiment.py --smoke      # small grid, fast sanity check
python run_experiment.py              # full factorial grid
```

Outputs:
- `results/data/all_results.xlsx` — sheets:
  - `all_runs` — one row per (configuration, seed): every metric of every run
  - `config_summary` — median across seeds, per configuration
  - `best_combinations` — configurations ranked by a quality score (reach
    target + low energy)
  - `convergence_histories` — geodesic distance vs iteration for every run
- `results/per_run/<config_id>/` — per-configuration figures (reference seed):
  `terrain_3d.png`, `terrain_topdown.png`, `geodesic_field.png`,
  `convergence.png`, `objective.png`, `energy_cumulative.png`,
  `energy_increments.png`, `curvature_repulsion.png`, `phase_x_vx.png`,
  `phase_speed_vs_distance.png`
- `results/plots/` — study-level figures: parameter-sensitivity box plots
  (every parameter x key metric), `pareto_energy_vs_distance.png`,
  `comparison_baseline_vs_composite.png`

## Terrain is already included

This package already contains a REAL terrain grid baked into
`results/data/terrain_cache.npz` -- the **Skole Beskids (Ukrainian
Carpathians)**, a 22 x 22 km tile with ~890 m of relief (real
SRTM/ASTER elevation). You can run the study immediately:

```bash
pip install -r requirements.txt
python run_experiment.py --smoke
python run_experiment.py
```

To use a DIFFERENT area, delete `results/data/terrain_cache.npz` and get
new terrain by one of the methods below.

## Where to get real terrain (for a different area)

### Easiest: one command, no GIS libraries, no manual download

```bash
pip install pillow
python fetch_terrain.py
```

`fetch_terrain.py` downloads REAL elevation for a bounding box from the
public AWS Open Data terrain tiles (no API key, no account) and builds the
elevation grid directly into `results/data/terrain_cache.npz`. The default
area is the **Skole Beskids (Ukrainian Carpathians)** -- strong relief,
close to Lviv. Edit `BBOX` / `ZOOM` at the top of the script for another
area or a bounded part of a map, then delete the old cache and re-run.
After it finishes, just run `python run_experiment.py`. This path needs
only numpy + Pillow (no geopandas / rasterio).

### Or provide your own file (contour .shp / DEM .tif)

Put one real file in `terrain_data/` and point `config.py` at it.

**Option A — contour shapefile `.shp` (default, `TERRAIN_KIND = "contours_shp"`):**
contour lines with an elevation attribute. Free sources:
- **DIVA-GIS** (diva-gis.org) → *Download → Data by country* → choose a
  country → data type *Elevation* → gives SRTM-derived contour lines as a
  shapefile. Unzip and point `TERRAIN_PATH` at the `.shp`.
- **Natural Earth** (naturalearthdata.com) — some contour/relief data.
- any GIS contour `.shp` with an elevation field (the loader auto-detects
  common field names: `ELEV`, `elevation`, `HEIGHT`, `CONTOUR`, ...; set
  `TERRAIN_ELEV_FIELD` in `config.py` if yours differs).

To study a **bounded part** of a map, set `TERRAIN_BBOX = (min_x, min_y,
max_x, max_y)` in `config.py` (in the shapefile's own coordinate units).

**Option B — DEM raster `.tif` (`TERRAIN_KIND = "dem_tif"`):**
a direct elevation raster. Free sources: OpenTopography, Copernicus
GLO-30, SRTM. Install `rasterio` as well.

The reconstructed elevation grid is cached to
`results/data/terrain_cache.npz` so the (slow) `.shp` interpolation runs
once; delete that file to reload a new terrain or region.

Geographic (lon/lat) data is converted to a local metric frame (metres)
automatically, so all distances and energies come out in metres / joules.

## Parameter grid (`config.py`)

```
OBJECTIVE_MODES          : baseline, composite
ALPHA_VALUES             : 0.0, 0.25, 0.5, 0.75, 1.0   (composite only)
K_VALUES                 : 50, 150
GAMMA_VALUES             : 0.1, 0.3, 1.0
REPULSION_WEIGHT_VALUES  : 0 (control), 20
REPULSION_POWER_VALUES   : 2
N_SEEDS                  : 3
```
Edit these lists to rescale the study. `repulsion_weight = 0` is the
no-repulsion control; the baseline objective mode does not sweep `alpha`.
Per-configuration plots are generated for the reference seed only; set
`PER_CONFIG_PLOTS = False` in `config.py` for very large grids.

## Project layout

```
run_experiment.py            main entry point
config.py                    all parameters, paths, the factorial grid
src/
  terrain_loader.py          load REAL terrain from .shp contours or .tif DEM
  geodesic_distance.py       geodesic distance field (Dijkstra)
  objective_functions.py     composite (formula 1) + baseline (formula 2)
  energy_model.py            work C, A_cum, E_mech, efficiency eta
  repulsion.py               inter-agent repulsion (scale-invariant)
  physics_engine.py          equation of motion (Lagrangian)
  particle.py                agent state + bookkeeping
  swarm_metrics.py           all metrics (navigation/energy/safety/cohesion/dynamics)
  experiment_runner.py       run one config; random start/target
  excel_writer.py            write all runs + summary + best + histories to .xlsx
  plotting/
    style.py                 shared publication style (English labels, grid)
    plot_terrain.py          terrain 3D / top-down / geodesic field
    plot_energy.py           energy & work curves
    plot_phase.py            phase portraits
    plot_convergence.py      convergence & objective curves
    plot_study.py            sensitivity / Pareto / comparison
    per_run.py               per-configuration plot orchestrator
```

## Notes

- Mass is fixed (not swept): the study measures work, it does not vary mass.
- A velocity clamp (`V_MAX`) prevents numerical blow-up under strong
  springs with weak friction.
- The repulsion law is written in a scale-invariant (normalised) form so
  its magnitude is comparable to the spring forces on a real km-scale
  terrain; `repulsion_weight` is therefore directly comparable across runs.
