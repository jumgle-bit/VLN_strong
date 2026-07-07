from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from taric_vln.config import TaricConfig
from taric_vln.eval import EpisodeMetrics, compute_episode_metrics
from taric_vln.grounding import TraversabilityGrounder
from taric_vln.memory import CueMemory3D
from taric_vln.perception import CueExtractor
from taric_vln.types import CameraIntrinsics, CueStep, Point3D, Pose3D


@dataclass
class EpisodeRecord:
    episode_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    metrics: EpisodeMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "steps": self.steps,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


class OfflineEpisodeRunner:
    def __init__(
        self,
        cue_extractor: CueExtractor,
        memory: CueMemory3D | None = None,
        grounder: TraversabilityGrounder | None = None,
        config: TaricConfig | None = None,
    ) -> None:
        self.config = config or TaricConfig()
        self.cue_extractor = cue_extractor
        self.grounder = grounder or TraversabilityGrounder(self.config)
        self.memory = memory or CueMemory3D(self.config)

    def run_steps(self, raw_steps: list[dict[str, Any]]) -> EpisodeRecord:
        if not raw_steps:
            raise ValueError("raw_steps must not be empty")

        episode_id = str(raw_steps[0].get("episode_id", "episode"))
        record = EpisodeRecord(episode_id=episode_id)
        poses: list[Pose3D] = []
        cue_available_gt: list[bool] = []
        goal = None

        for idx, step in enumerate(raw_steps):
            pose = Pose3D.from_dict(step.get("pose", {}))
            camera = CameraIntrinsics.from_dict(step.get("camera"))
            instruction = str(step.get("instruction", ""))
            image_path = str(step.get("image_path", ""))
            if "goal_position" in step:
                goal = Point3D.from_dict(step["goal_position"])

            previous_state = {
                "step_index": idx,
                "memory": self.memory.snapshot(),
            }
            cue_step = self.cue_extractor.extract(
                image_path=image_path,
                instruction=instruction,
                camera=camera,
                previous_state=previous_state,
            )

            command_bearing = cue_step.executable_bearing_rad
            memory_readout = None
            if cue_step.visible_after_gate and cue_step.observation.focus_pixel is not None:
                self.memory.update_visible(pose, cue_step.observation.focus_pixel, camera)
            elif self.memory.ready:
                memory_readout = self.memory.readout(
                    pose,
                    cue_step.observation.traversability_scores,
                    self.grounder,
                )
                if memory_readout.executable_bearing_rad is not None:
                    command_bearing = memory_readout.executable_bearing_rad

            poses.append(pose)
            cue_available_gt.append(bool(step.get("cue_available_gt", cue_step.visible_after_gate)))
            record.steps.append(
                {
                    "step_index": idx,
                    "pose": pose.to_dict(),
                    "cue": cue_step.to_dict(),
                    "memory": self.memory.snapshot(),
                    "memory_readout": memory_readout.to_dict() if memory_readout else None,
                    "command_bearing_rad": command_bearing,
                }
            )

        if goal is not None:
            record.metrics = compute_episode_metrics(
                episode_id=episode_id,
                poses=poses,
                goal=goal,
                cue_available=cue_available_gt,
                shortest_path_m=raw_steps[0].get("shortest_path_m"),
            )
        return record


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def group_by_episode(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        episode_id = str(row.get("episode_id", "episode"))
        grouped.setdefault(episode_id, []).append(row)
    return grouped
