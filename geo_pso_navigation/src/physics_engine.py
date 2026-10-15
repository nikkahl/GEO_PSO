"""
physics_engine.py
Lagrangian equation of motion for the swarm, coupling the composite
objective into the PSO memory.

Per agent, per step:
  m dv/dt = -m(g + s/m) grad f            gravity + curvature correction
            - gamma m (v + (grad f . v) grad f)   Rayleigh dissipation
            + F_spring                     cognitive + social + target springs
            + F_rep                        inter-agent repulsion
  s = v^T H(u) v                           curvature term (H = Hessian of f)

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.terrain_loader import Terrain
from src.geodesic_distance import GeodesicField
from src.repulsion import Repulsion
from src.particle import Particle
from src.energy_model import work_increment, dissipation_increment


@dataclass
class PhysicsEngine:
    terrain: Terrain
    geo_field: GeodesicField
    repulsion: Repulsion
    delta_t: float
    g: float
    gamma: float
    k1: float                 
    k2: float                 
    k_target: float
    f_max: float
    f_max_target: float
    v_max: float
    rng: np.random.Generator

    @staticmethod
    def _cap(force: np.ndarray, f_max: float) -> np.ndarray:
        mag = float(np.linalg.norm(force))
        if mag > f_max and mag > 0:
            return force * (f_max / mag)
        return force

    def step(self, swarm: List[Particle], g_best_position: np.ndarray) -> dict:
        """Advance the whole swarm one step; return per-step aggregate
        quantities (means) for logging / energy plots."""
        positions = np.array([p.position for p in swarm])
        rep_forces = self.repulsion.swarm_forces(positions)

        agg = {"work": [], "e_mech": [], "kinetic": [], "curvature": [],
               "repulsion_mag": [], "speed": []}

        for i, p in enumerate(swarm):
            u, v, m = p.position, p.velocity, p.mass
            grad_f = self.terrain.gradient1(u)
            H = self.terrain.hessian1(u)
            s = float(v @ H @ v)
            grad_d = self.geo_field.gradient1(u)
            r1, r2 = self.rng.random(2)

            cognitive = self.k1 * r1 * (p.best_position - u)
            social = self.k2 * r2 * (g_best_position - u)
            target = -self.k_target * grad_d
            spring = self._cap(cognitive + social, self.f_max) + self._cap(target, self.f_max_target)

            f_rep = self._cap(rep_forces[i], self.f_max)   # cap repulsion like the springs
            p.repulsion_mag = float(np.linalg.norm(f_rep))

            gravity_acc = -(self.g + s / m) * grad_f
            dissip_acc = -self.gamma * (v + float(grad_f @ v) * grad_f)
            dv = (gravity_acc + dissip_acc + spring / m + f_rep / m) * self.delta_t

            new_v = v + dv
            speed = float(np.linalg.norm(new_v))
            if speed > self.v_max:
                new_v = new_v * (self.v_max / speed)

            step_vec = new_v * self.delta_t
            dA = work_increment(grad_f, step_vec, self.gamma, m, self.g)
            dE = dissipation_increment(new_v, self.gamma, self.delta_t)
            kinetic = 0.5 * m * float(new_v @ new_v)

            p.work_cum += dA
            p.e_mech += dE
            p.kinetic = kinetic
            p.curvature = s

            p.step(new_v, self.delta_t)

            agg["work"].append(dA); agg["e_mech"].append(dE); agg["kinetic"].append(kinetic)
            agg["curvature"].append(abs(s)); agg["repulsion_mag"].append(p.repulsion_mag)
            agg["speed"].append(p.speed)

        return {key: float(np.mean(val)) for key, val in agg.items()}
