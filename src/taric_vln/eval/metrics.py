from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from taric_vln.geometry import distance_2d
from taric_vln.types import Point3D, Pose3D


@dataclass
class EpisodeMetrics:
    episode_id: str
    success: bool
    spl: float
    final_distance_m: float
    path_length_m: float
    shortest_path_m: float
    fail_at_cue_free: bool
    cfsr_10: bool
    cfsr_25: bool
    cfsr_50: bool

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "episode_id": self.episode_id,
            "success": self.success,
            "spl": self.spl,
            "final_distance_m": self.final_distance_m,
            "path_length_m": self.path_length_m,
            "shortest_path_m": self.shortest_path_m,
            "fail_at_cue_free": self.fail_at_cue_free,
            "cfsr_10": self.cfsr_10,
            "cfsr_25": self.cfsr_25,
            "cfsr_50": self.cfsr_50,
        }


def compute_episode_metrics(
    episode_id: str,
    poses: list[Pose3D],
    goal: Point3D,
    cue_available: list[bool],
    success_radius_m: float = 3.0,
    shortest_path_m: float | None = None,
    progress_window: int = 5,
) -> EpisodeMetrics:
    if not poses:
        raise ValueError("poses must not be empty")

    path_length = sum(distance_2d(a, b) for a, b in zip(poses, poses[1:]))
    shortest = shortest_path_m if shortest_path_m is not None else distance_2d(poses[0], goal)
    final_distance = distance_2d(poses[-1], goal)
    success = final_distance <= success_radius_m
    spl = 0.0
    if success:
        spl = shortest / max(path_length, shortest, 1e-6)

    fail_at_cue_free = (not success) and bool(cue_available) and (not cue_available[-1])
    survive_distance = cue_free_survival_distance(poses, goal, cue_available, progress_window)
    return EpisodeMetrics(
        episode_id=episode_id,
        success=success,
        spl=spl,
        final_distance_m=final_distance,
        path_length_m=path_length,
        shortest_path_m=shortest,
        fail_at_cue_free=fail_at_cue_free,
        cfsr_10=survive_distance >= 10.0,
        cfsr_25=survive_distance >= 25.0,
        cfsr_50=survive_distance >= 50.0,
    )


def cue_free_survival_distance(
    poses: list[Pose3D],
    goal: Point3D,
    cue_available: list[bool],
    progress_window: int = 5,
) -> float:
    if not poses:
        return 0.0
    last_visible = 0
    for i, visible in enumerate(cue_available[: len(poses)]):
        if visible:
            last_visible = i

    traveled = cumulative_lengths(poses)
    radii = [distance_2d(pose, goal) for pose in poses]
    stop_index = len(poses) - 1
    for idx in range(last_visible + progress_window, len(poses)):
        if radii[idx] >= radii[idx - progress_window]:
            stop_index = idx
            break
    return max(0.0, traveled[stop_index] - traveled[last_visible])


def cumulative_lengths(poses: list[Pose3D]) -> list[float]:
    lengths = [0.0]
    for a, b in zip(poses, poses[1:]):
        lengths.append(lengths[-1] + distance_2d(a, b))
    return lengths


def summarize_metrics(metrics: Iterable[EpisodeMetrics]) -> dict[str, float]:
    items = list(metrics)
    if not items:
        return {}
    n = len(items)
    return {
        "episodes": float(n),
        "sr": sum(1.0 for m in items if m.success) / n,
        "spl": sum(m.spl for m in items) / n,
        "fail_at_cf": sum(1.0 for m in items if m.fail_at_cue_free) / n,
        "cfsr_10": sum(1.0 for m in items if m.cfsr_10) / n,
        "cfsr_25": sum(1.0 for m in items if m.cfsr_25) / n,
        "cfsr_50": sum(1.0 for m in items if m.cfsr_50) / n,
        "mean_final_distance_m": sum(m.final_distance_m for m in items) / n,
        "mean_path_length_m": sum(m.path_length_m for m in items) / n,
    }


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return math.nan
    return sum(values) / len(values)
