from __future__ import annotations

from taric_vln.config import TaricConfig
from taric_vln.perception.deepseek_client import (
    DeepSeekVLMClient,
    extract_json_object,
)
from taric_vln.types import CameraIntrinsics


def test_extract_json_object_from_fenced_text() -> None:
    payload = extract_json_object('```json\n{"visible": true, "focus_pixel": [320, 240]}\n```')
    assert payload["visible"] is True
    assert payload["focus_pixel"] == [320, 240]


def test_observation_normalizes_focus_pixel_to_bearing() -> None:
    client = DeepSeekVLMClient(api_key="x", cache_dir=None)
    obs = client._observation_from_payload(
        {
            "visible": True,
            "focus_pixel": [320, 240],
            "tile_scores": [0.1, 1.0],
            "traversability_scores": [1.0] * 10,
            "confidence": 0.8,
        },
        CameraIntrinsics(),
        TaricConfig(),
    )
    assert obs.visible is True
    assert obs.cue_bearing_rad == 0.0
    assert len(obs.traversability_scores) == 10
