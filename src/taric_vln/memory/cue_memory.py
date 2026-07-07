from __future__ import annotations

from dataclasses import dataclass
import math
import random

from taric_vln.config import TaricConfig
from taric_vln.geometry import (
    angle_diff,
    distance_2d,
    pixel_to_bearing,
    project_world_point_to_pixel,
    triangulate_rays_2d,
    world_bearing_from_pose,
)
from taric_vln.grounding import TraversabilityGrounder
from taric_vln.types import CameraIntrinsics, Point3D, Pose3D


@dataclass
class MemoryReadout:
    memory_ready: bool
    goal_mean: Point3D | None
    uncertainty: float
    memory_bearing_rad: float | None
    executable_bearing_rad: float | None
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_ready": self.memory_ready,
            "goal_mean": self.goal_mean.to_dict() if self.goal_mean else None,
            "uncertainty": self.uncertainty,
            "memory_bearing_rad": self.memory_bearing_rad,
            "executable_bearing_rad": self.executable_bearing_rad,
            "status": self.status,
        }


class CueMemory3D:
    def __init__(
        self,
        config: TaricConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or TaricConfig()
        self.rng = rng or random.Random(7)
        self.particles: list[Point3D] = []
        self.weights: list[float] = []
        self._bootstrap: tuple[Pose3D, float] | None = None

    @property
    def ready(self) -> bool:
        return bool(self.particles)

    def update_visible(
        self,
        pose: Pose3D,
        focus_pixel: tuple[float, float],
        camera: CameraIntrinsics,
    ) -> bool:
        bearing = pixel_to_bearing(focus_pixel, camera)
        if not self.ready:
            return self._try_initialize(pose, bearing)
        self._diffuse()
        self._weight_by_reprojection(pose, focus_pixel, camera)
        if self._effective_sample_size() < 0.5 * len(self.particles):
            self._resample()
        return True

    def readout(
        self,
        pose: Pose3D,
        traversability_scores: list[float],
        grounder: TraversabilityGrounder,
    ) -> MemoryReadout:
        if not self.ready:
            return MemoryReadout(False, None, 0.0, None, None, "memory_not_ready")

        mean = self.mean()
        uncertainty = self.uncertainty()
        bearing = world_bearing_from_pose(pose, mean)
        cone = self._uncertainty_to_cone(uncertainty)
        grounded = grounder.ground(
            preferred_bearing_rad=bearing,
            traversability_scores=traversability_scores,
            cone_half_angle_rad=cone,
        )
        if grounded.selected_sector is None:
            return MemoryReadout(
                True,
                mean,
                uncertainty,
                bearing,
                bearing,
                "memory_no_traversable_in_cone",
            )
        return MemoryReadout(
            True,
            mean,
            uncertainty,
            bearing,
            grounded.executable_bearing_rad,
            grounded.status,
        )

    def mean(self) -> Point3D:
        if not self.ready:
            return Point3D(0.0, 0.0, 0.0)
        x = sum(w * p.x for w, p in zip(self.weights, self.particles))
        y = sum(w * p.y for w, p in zip(self.weights, self.particles))
        z = sum(w * p.z for w, p in zip(self.weights, self.particles))
        return Point3D(x, y, z)

    def uncertainty(self) -> float:
        if not self.ready:
            return 0.0
        mean = self.mean()
        return sum(
            w * ((p.x - mean.x) ** 2 + (p.y - mean.y) ** 2 + (p.z - mean.z) ** 2)
            for w, p in zip(self.weights, self.particles)
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "mean": self.mean().to_dict() if self.ready else None,
            "uncertainty": self.uncertainty() if self.ready else 0.0,
            "particle_count": len(self.particles),
        }

    def _try_initialize(self, pose: Pose3D, bearing: float) -> bool:
        if self._bootstrap is None:
            self._bootstrap = (pose, bearing)
            return False

        first_pose, first_bearing = self._bootstrap
        if distance_2d(first_pose, pose) < self.config.min_triangulation_baseline_m:
            return False

        center = triangulate_rays_2d(first_pose, first_bearing, pose, bearing)
        self.particles = [
            Point3D(
                center.x + self.rng.gauss(0.0, self.config.init_noise_m),
                center.y + self.rng.gauss(0.0, self.config.init_noise_m),
                center.z + self.rng.gauss(0.0, 0.1 * self.config.init_noise_m),
            )
            for _ in range(self.config.particle_count)
        ]
        self.weights = [1.0 / self.config.particle_count] * self.config.particle_count
        return True

    def _diffuse(self) -> None:
        sigma = self.config.process_noise_m
        self.particles = [
            Point3D(
                p.x + self.rng.gauss(0.0, sigma),
                p.y + self.rng.gauss(0.0, sigma),
                p.z,
            )
            for p in self.particles
        ]

    def _weight_by_reprojection(
        self,
        pose: Pose3D,
        focus_pixel: tuple[float, float],
        camera: CameraIntrinsics,
    ) -> None:
        sigma2 = max(self.config.observation_noise_px**2, 1e-6)
        new_weights = []
        for weight, particle in zip(self.weights, self.particles):
            projection = project_world_point_to_pixel(particle, pose, camera)
            if projection is None:
                likelihood = 1e-6
            else:
                du = projection[0] - focus_pixel[0]
                dv = projection[1] - focus_pixel[1]
                likelihood = math.exp(-(du * du + dv * dv) / (2.0 * sigma2))
            new_weights.append(weight * likelihood)

        total = sum(new_weights)
        if total <= 1e-12:
            self.weights = [1.0 / len(self.particles)] * len(self.particles)
        else:
            self.weights = [w / total for w in new_weights]

    def _effective_sample_size(self) -> float:
        denom = sum(w * w for w in self.weights)
        return 0.0 if denom <= 0.0 else 1.0 / denom

    def _resample(self) -> None:
        cumulative = []
        total = 0.0
        for weight in self.weights:
            total += weight
            cumulative.append(total)

        new_particles = []
        step = 1.0 / len(self.particles)
        start = self.rng.random() * step
        idx = 0
        for m in range(len(self.particles)):
            u = start + m * step
            while idx < len(cumulative) - 1 and u > cumulative[idx]:
                idx += 1
            new_particles.append(self.particles[idx])
        self.particles = new_particles
        self.weights = [1.0 / len(self.particles)] * len(self.particles)

    def _uncertainty_to_cone(self, uncertainty: float) -> float:
        gamma = max(self.config.memory_uncertainty_gamma, 1e-6)
        return self.config.theta_min_rad + (
            1.0 - math.exp(-uncertainty / gamma)
        ) * (self.config.theta_max_rad - self.config.theta_min_rad)
