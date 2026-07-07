from __future__ import annotations

from taric_vln.config import TaricConfig
from taric_vln.geometry import wrap_angle
from taric_vln.types import ControlCommand


class HeadingController:
    """Convert executable heading into a conservative velocity command."""

    def __init__(
        self,
        config: TaricConfig | None = None,
        max_speed_mps: float | None = None,
        max_angular_z_rad_s: float = 0.8,
        yaw_gain: float = 1.0,
    ) -> None:
        self.config = config or TaricConfig()
        self.max_speed_mps = max_speed_mps or self.config.default_speed_mps
        self.max_angular_z_rad_s = max_angular_z_rad_s
        self.yaw_gain = yaw_gain

    def command(self, executable_heading_rad: float, source: str = "taric") -> ControlCommand:
        heading = wrap_angle(executable_heading_rad)
        angular = clamp(self.yaw_gain * heading, -self.max_angular_z_rad_s, self.max_angular_z_rad_s)
        speed_scale = max(0.0, 1.0 - min(abs(heading), 1.2) / 1.2)
        speed = self.max_speed_mps * (0.2 + 0.8 * speed_scale)
        return ControlCommand(
            heading_rad=heading,
            speed_mps=speed,
            angular_z_rad_s=angular,
            source=source,
        )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
