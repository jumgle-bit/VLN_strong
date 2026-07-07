from __future__ import annotations

import tempfile
from pathlib import Path

from taric_vln.config import TaricConfig
from taric_vln.perception.qwen_client import QwenVisionClient, parse_qwen_json_content
from taric_vln.types import CameraIntrinsics


def test_qwen_payload_uses_image_url_content() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "image.jpg"
        image.write_bytes(b"fake")
        client = QwenVisionClient(api_key="x", model="qwen-vl-plus")

        payload = client._build_payload(
            image,
            "find target",
            CameraIntrinsics(),
            previous_state={},
            config=TaricConfig(),
        )

        assert payload["model"] == "qwen-vl-plus"
        assert payload["max_tokens"] == 512
        content = payload["messages"][1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"


def test_qwen_payload_size_reports_bytes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "image.jpg"
        image.write_bytes(b"fake")
        client = QwenVisionClient(api_key="x", model="qwen-vl-plus")

        size = client.payload_size_bytes(image, "find target", config=TaricConfig())

        assert size > 0


def test_qwen_parse_error_mentions_max_tokens() -> None:
    try:
        parse_qwen_json_content('{"visible": true, "focus_pixel": [320,')
    except RuntimeError as exc:
        assert "--max-tokens" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_qwen_parser_recovers_required_fields_from_truncated_extra_output() -> None:
    content = """
{
  "exploration_phrase": "Follow the path towards the red library entrance.",
  "goal_phrase": "Locate the red 'LIBRARY' sign.",
  "visible": true,
  "focus_pixel": [400, 230],
  "tile_scores": [0.1, 0.2, 0.9],
  "cue_bearing_deg": 0.0,
  "traversability_scores": [0.0, 0.1, 0.2, 0.3, 0.9, 0.9, 0.3, 0.2, 0.1, 0.0],
  "confidence": 0.95,
  "extra": "this text never closes
"""

    payload = parse_qwen_json_content(content)

    assert payload["visible"] is True
    assert payload["focus_pixel"] == [400, 230]
    assert payload["traversability_scores"][4] == 0.9
    assert payload["_recovered_from_non_json"] is True
