from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaricConfig:
    top_k_tiles: int = 2
    heading_sectors: int = 10
    max_bearing_deg: float = 60.0
    visible_threshold: float = 1.5
    particle_count: int = 200
    min_triangulation_baseline_m: float = 0.5
    scout_traversability_threshold: float = 0.7
    go1_traversability_threshold: float = 0.5
    risk_lambda: float = 1.0
    theta_min_deg: float = 15.0
    theta_max_deg: float = 60.0
    memory_uncertainty_gamma: float = 1.0
    process_noise_m: float = 0.1
    observation_noise_px: float = 15.0
    init_noise_m: float = 1.0
    default_speed_mps: float = 0.5

    @property
    def max_bearing_rad(self) -> float:
        return math.radians(self.max_bearing_deg)

    @property
    def theta_min_rad(self) -> float:
        return math.radians(self.theta_min_deg)

    @property
    def theta_max_rad(self) -> float:
        return math.radians(self.theta_max_deg)

    @classmethod
    def from_json(cls, path: str | Path) -> "TaricConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k_tiles": self.top_k_tiles,
            "heading_sectors": self.heading_sectors,
            "max_bearing_deg": self.max_bearing_deg,
            "visible_threshold": self.visible_threshold,
            "particle_count": self.particle_count,
            "min_triangulation_baseline_m": self.min_triangulation_baseline_m,
            "scout_traversability_threshold": self.scout_traversability_threshold,
            "go1_traversability_threshold": self.go1_traversability_threshold,
            "risk_lambda": self.risk_lambda,
            "theta_min_deg": self.theta_min_deg,
            "theta_max_deg": self.theta_max_deg,
            "memory_uncertainty_gamma": self.memory_uncertainty_gamma,
            "process_noise_m": self.process_noise_m,
            "observation_noise_px": self.observation_noise_px,
            "init_noise_m": self.init_noise_m,
            "default_speed_mps": self.default_speed_mps,
        }
