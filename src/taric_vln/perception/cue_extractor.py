from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from taric_vln.config import TaricConfig
from taric_vln.geometry import pixel_to_bearing
from taric_vln.grounding import TraversabilityGrounder
from taric_vln.types import CameraIntrinsics, CueStep, VLMObservation


class VLMClient(Protocol):
    def analyze(
        self,
        image_path: str | Path,
        instruction: str,
        camera: CameraIntrinsics | None = None,
        previous_state: dict[str, Any] | None = None,
        config: TaricConfig | None = None,
    ) -> VLMObservation:
        ...


class CueExtractor:
    def __init__(
        self,
        client: VLMClient,
        config: TaricConfig | None = None,
        grounder: TraversabilityGrounder | None = None,
    ) -> None:
        self.client = client
        self.config = config or TaricConfig()
        self.grounder = grounder or TraversabilityGrounder(self.config)

    def extract(
        self,
        image_path: str | Path,
        instruction: str,
        camera: CameraIntrinsics | None = None,
        previous_state: dict[str, Any] | None = None,
    ) -> CueStep:
        camera = camera or CameraIntrinsics()
        observation = self.client.analyze(
            image_path=image_path,
            instruction=instruction,
            camera=camera,
            previous_state=previous_state,
            config=self.config,
        )

        visible = self._visible_after_gate(observation)
        semantic_bearing = observation.cue_bearing_rad
        if visible and observation.focus_pixel is not None:
            semantic_bearing = pixel_to_bearing(observation.focus_pixel, camera)

        grounding = self.grounder.ground(
            preferred_bearing_rad=semantic_bearing,
            traversability_scores=observation.traversability_scores,
        )
        return CueStep(
            observation=observation,
            visible_after_gate=visible,
            semantic_bearing_rad=semantic_bearing,
            executable_bearing_rad=grounding.executable_bearing_rad,
            traversability_status=grounding.status,
            selected_sector=grounding.selected_sector,
        )

    def _visible_after_gate(self, observation: VLMObservation) -> bool:
        if not observation.visible:
            return False
        if not observation.tile_scores:
            return observation.focus_pixel is not None
        mean_score = sum(observation.tile_scores) / max(len(observation.tile_scores), 1)
        if mean_score <= 1e-9:
            return False
        peakedness = max(observation.tile_scores) / mean_score
        return peakedness >= self.config.visible_threshold
