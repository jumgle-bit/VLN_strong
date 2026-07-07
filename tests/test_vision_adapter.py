from __future__ import annotations

import tempfile
from pathlib import Path

from taric_vln.config import TaricConfig
from taric_vln.perception.vision_adapter import PythonVisionClient, observation_from_result
from taric_vln.types import CameraIntrinsics


def test_observation_from_result_recomputes_bearing_from_focus_pixel() -> None:
    obs = observation_from_result(
        {
            "visible": True,
            "focus_pixel": [320, 240],
            "tile_scores": [0.1, 1.0],
            "traversability_scores": [1.0] * 10,
            "confidence": 0.9,
        },
        CameraIntrinsics(),
        TaricConfig(),
    )

    assert obs.visible is True
    assert obs.cue_bearing_rad == 0.0


def test_python_vision_client_loads_file_adapter() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        adapter_file = Path(tmp) / "adapter.py"
        adapter_file.write_text(
            """
class Adapter:
    def analyze(self, image_path, instruction, camera, previous_state, config):
        return {
            "visible": False,
            "cue_bearing_deg": 0,
            "traversability_scores": [1.0] * int(config["heading_sectors"]),
            "confidence": 0.5,
        }
""",
            encoding="utf-8",
        )

        client = PythonVisionClient(f"{adapter_file}:Adapter")
        obs = client.analyze("image.jpg", "go", CameraIntrinsics(), {}, TaricConfig())

        assert obs.confidence == 0.5
        assert len(obs.traversability_scores) == 10
