from __future__ import annotations

from taric_vln.perception import MockVLMClient, QwenVisionClient
from taric_vln.ros.ros_bridge import RosBridgeSettings, build_vlm_client
from taric_vln.ros.turtlebot3_commander import TurtleBot3CommanderSettings


def test_ros_bridge_defaults_match_turtlebot3_waffle_pi_topics() -> None:
    settings = RosBridgeSettings()

    assert settings.image_topic == "/camera/rgb/image_raw"
    assert settings.camera_info_topic == "/camera/rgb/camera_info"
    assert settings.odom_topic == "/odom"
    assert settings.heading_topic == "/taric/executable_heading"
    assert settings.vision_backend == "qwen"


def test_ros_bridge_builds_qwen_client_by_default() -> None:
    client = build_vlm_client(RosBridgeSettings())

    assert isinstance(client, QwenVisionClient)
    assert client.timeout_s == 90.0
    assert client.retries == 0
    assert client.max_tokens == 1024


def test_ros_bridge_can_use_mock_backend() -> None:
    client = build_vlm_client(RosBridgeSettings(vision_backend="mock"))

    assert isinstance(client, MockVLMClient)


def test_turtlebot3_commander_defaults_are_conservative() -> None:
    settings = TurtleBot3CommanderSettings()

    assert settings.heading_topic == "/taric/executable_heading"
    assert settings.cmd_vel_topic == "/cmd_vel"
    assert settings.max_speed_mps <= 0.2
    assert settings.command_timeout_s > 0.0
