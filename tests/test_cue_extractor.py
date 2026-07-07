from __future__ import annotations

from taric_vln.config import TaricConfig
from taric_vln.perception import CueExtractor, MockVLMClient
from taric_vln.types import CameraIntrinsics, VLMObservation


def test_cue_extractor_applies_visibility_gate_and_grounding() -> None:
    config = TaricConfig(heading_sectors=5, max_bearing_deg=60.0)
    client = MockVLMClient(
        [
            VLMObservation(
                visible=True,
                focus_pixel=(320.0, 240.0),
                tile_scores=[0.1, 0.1, 1.0],
                traversability_scores=[0.1, 0.1, 0.1, 0.9, 0.1],
                confidence=1.0,
            )
        ]
    )
    extractor = CueExtractor(client, config=config)

    cue = extractor.extract("unused.jpg", "find target", CameraIntrinsics())

    assert cue.visible_after_gate is True
    assert cue.traversability_status == "snapped_to_traversable"
    assert cue.selected_sector == 3


def test_cue_extractor_rejects_flat_tile_scores() -> None:
    client = MockVLMClient(
        [
            VLMObservation(
                visible=True,
                focus_pixel=(320.0, 240.0),
                tile_scores=[1.0, 1.0, 1.0],
                traversability_scores=[1.0] * 10,
            )
        ]
    )
    cue = CueExtractor(client).extract("unused.jpg", "find target")

    assert cue.visible_after_gate is False
