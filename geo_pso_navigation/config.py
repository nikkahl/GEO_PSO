"""
config.py
Central configuration for the swarm-navigation study: terrain source,
swarm/physics constants, the full factorial parameter grid, and output
paths. Edit values here; nothing else needs changing to rescale the study.
"""
from __future__ import annotations
import os

TERRAIN_KIND: str = "contours_shp"
TERRAIN_PATH: str = os.path.join("terrain_data", "contours.shp")
TERRAIN_ELEV_FIELD: str | None = None

TERRAIN_BBOX: tuple[float, float, float, float] | None = None
TERRAIN_GRID_N: int = 80

TERRAIN_CACHE: str = os.path.join("results", "data", "terrain_cache.npz")

N_PARTICLES: int = 30          
MASS: float = 1.0              
G_ACCEL: float = 9.8           # gravitational acceleration 
SPAWN_RADIUS_FRAC: float = 0.03  # cluster r, fraction of tile size
MAX_STEPS: int = 3000          
CONVERGENCE_FRAC: float = 0.02   
DELTA_T: float = 0.05          
K_TARGET: float = 150.0        
F_MAX: float = 25.0            
F_MAX_TARGET: float = 60.0     
V_MAX: float = 4000.0          
REPULSION_RHO0_FRAC: float = 0.02  # repulsion cut-off radius, fraction of tile size

N_SEEDS: int = 3
SEEDS: list[int] = [2000 + i for i in range(N_SEEDS)]
REFERENCE_SEED: int = SEEDS[0]   


# baseline -- fi = di -> min                                             (formula 2)
OBJECTIVE_MODES: list[str] = ["baseline", "composite"]

ALPHA_VALUES: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]  
K_VALUES: list[float] = [50.0, 150.0]                     
GAMMA_VALUES: list[float] = [0.1, 0.3, 1.0]              
REPULSION_WEIGHT_VALUES: list[float] = [0.0, 20.0]       
REPULSION_POWER_VALUES: list[float] = [2.0]             

PER_CONFIG_PLOTS: bool = True

RESULTS_DIR: str = "results"
DATA_DIR: str = os.path.join(RESULTS_DIR, "data")
PLOTS_DIR: str = os.path.join(RESULTS_DIR, "plots")
PER_RUN_DIR: str = os.path.join(RESULTS_DIR, "per_run")
EXCEL_PATH: str = os.path.join(DATA_DIR, "all_results.xlsx")

def build_grid() -> list[dict]:
    """Return every configuration of the full factorial grid.

    The repulsion_weight = 0 level is a no-repulsion control; its
    repulsion_power is irrelevant, so it is kept once (not duplicated per
    power). For the baseline objective mode alpha has no effect, so it is
    not swept there.
    """
    grid: list[dict] = []
    for mode in OBJECTIVE_MODES:
        alphas = [None] if mode == "baseline" else ALPHA_VALUES
        for alpha in alphas:
            for k in K_VALUES:
                for gamma in GAMMA_VALUES:
                    for w in REPULSION_WEIGHT_VALUES:
                        powers = [REPULSION_POWER_VALUES[0]] if w == 0.0 else REPULSION_POWER_VALUES
                        for n in powers:
                            grid.append({
                                "mode": mode,
                                "alpha": alpha,
                                "k": k,
                                "gamma": gamma,
                                "repulsion_weight": w,
                                "repulsion_power": n,
                                "is_control": (w == 0.0),
                            })
    return grid

def config_id(cfg: dict) -> str:
    """Short, filesystem-safe identifier that encodes a configuration."""
    a = "NA" if cfg["alpha"] is None else f"{cfg['alpha']:.2f}"
    return (f"{cfg['mode']}_a{a}_k{cfg['k']:.0f}_g{cfg['gamma']:.2f}"
            f"_w{cfg['repulsion_weight']:.0f}_n{cfg['repulsion_power']:.0f}")
