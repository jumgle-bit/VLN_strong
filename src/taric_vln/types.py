from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int = 640
    height: int = 480
    fx: float = 550.0
    fy: float = 550.0
    cx: float = 320.0
    cy: float = 240.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CameraIntrinsics":
        if not data:
            return cls()
        return cls(
            width=int(data.get("width", 640)),
            height=int(data.get("height", 480)),
            fx=float(data.get("fx", 550.0)),
            fy=float(data.get("fy", 550.0)),
            cx=float(data.get("cx", data.get("width", 640) / 2.0)),
            cy=float(data.get("cy", data.get("height", 480) / 2.0)),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }


@dataclass(frozen=True)
class Pose3D:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pose3D":
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            z=float(data.get("z", 0.0)),
            yaw=float(data.get("yaw", 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw}


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Point3D":
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            z=float(data.get("z", 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class VLMObservation:
    exploration_phrase: str = ""
    goal_phrase: str = ""
    visible: bool = False
    focus_pixel: tuple[float, float] | None = None
    tile_scores: list[float] = field(default_factory=list)
    cue_bearing_rad: float = 0.0
    traversability_scores: list[float] = field(default_factory=list)
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exploration_phrase": self.exploration_phrase,
            "goal_phrase": self.goal_phrase,
            "visible": self.visible,
            "focus_pixel": list(self.focus_pixel) if self.focus_pixel else None,
            "tile_scores": self.tile_scores,
            "cue_bearing_rad": self.cue_bearing_rad,
            "traversability_scores": self.traversability_scores,
            "confidence": self.confidence,
            "error": self.error,
        }


@dataclass
class InstructionPlan:
    instruction: str
    exploration_phrase: str = ""
    goal_phrase: str = ""
    avoid_phrases: list[str] = field(default_factory=list)
    route_cues: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "exploration_phrase": self.exploration_phrase,
            "goal_phrase": self.goal_phrase,
            "avoid_phrases": self.avoid_phrases,
            "route_cues": self.route_cues,
            "error": self.error,
        }


@dataclass(frozen=True)
class ControlCommand:
    heading_rad: float
    speed_mps: float
    angular_z_rad_s: float
    timestamp_s: float = field(default_factory=time.time)
    source: str = "taric"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "heading_rad": self.heading_rad,
            "speed_mps": self.speed_mps,
            "angular_z_rad_s": self.angular_z_rad_s,
            "timestamp_s": self.timestamp_s,
            "source": self.source,
        }


@dataclass
class CueStep:
    observation: VLMObservation
    visible_after_gate: bool
    semantic_bearing_rad: float
    executable_bearing_rad: float
    traversability_status: str
    selected_sector: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "visible_after_gate": self.visible_after_gate,
            "semantic_bearing_rad": self.semantic_bearing_rad,
            "executable_bearing_rad": self.executable_bearing_rad,
            "traversability_status": self.traversability_status,
            "selected_sector": self.selected_sector,
        }
