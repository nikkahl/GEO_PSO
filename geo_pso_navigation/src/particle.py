"""
particle.py
Swarm agent: position, velocity, mass, and full physical/energetic
bookkeeping (work, dissipated energy, best objective value, trajectories).
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
from src.objective_functions import worst_value


class Particle:
    def __init__(self, id_: int, position: np.ndarray, mass: float,
                 bounds: tuple, mode: str, rng: Optional[np.random.Generator] = None) -> None:
        self.id = id_
        self.mode = mode
        self.position = np.asarray(position, float).copy()
        self.mass = float(mass)
        self.bounds = bounds
        gen = rng if rng is not None else np.random.default_rng()
        span = 0.02 * min(bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0])
        self.velocity = gen.uniform(-span, span, size=2)

        self.objective_value = worst_value(mode)
        self.best_objective = worst_value(mode)
        self.best_position = self.position.copy()

        self.d_geo = float("inf")
        self.d_geo_start = float("inf")
        self.d_geo_min = float("inf")

        self.work_cum = 0.0          
        self.e_mech = 0.0            
        self.kinetic = 0.0          
        self.curvature = 0.0        # s = v^T H v
        self.repulsion_mag = 0.0    
        self.speed = 0.0            

        self.trajectory: List[np.ndarray] = [self.position.copy()]
        self.velocity_trajectory: List[np.ndarray] = [self.velocity.copy()]

    def clamp_to_bounds(self) -> None:
        (x0, x1), (y0, y1) = self.bounds
        self.position[0] = np.clip(self.position[0], x0, x1)
        self.position[1] = np.clip(self.position[1], y0, y1)

    def step(self, new_velocity: np.ndarray, delta_t: float) -> None:
        self.velocity = new_velocity
        self.position = self.position + self.velocity * delta_t
        self.clamp_to_bounds()
        self.speed = float(np.linalg.norm(self.velocity))
        self.trajectory.append(self.position.copy())
        self.velocity_trajectory.append(self.velocity.copy())

    def update_best(self) -> None:
        from src.objective_functions import better
        if better(self.objective_value, self.best_objective, self.mode):
            self.best_objective = self.objective_value
            self.best_position = self.position.copy()


def init_swarm(n: int, start_xy: np.ndarray, spawn_radius: float, mass: float,
               bounds: tuple, mode: str, rng: np.random.Generator) -> List[Particle]:
    swarm = []
    for i in range(n):
        offset = rng.uniform(-spawn_radius, spawn_radius, size=2)
        swarm.append(Particle(i, np.asarray(start_xy, float) + offset, mass, bounds, mode, rng))
    return swarm
