from __future__ import annotations

from taric_vln.perception import MockVLMClient, QwenVisionClient
from taric_vln.ros.episode_recorder import (
    EpisodeRecorderSettings,
    parse_bool,
    parse_optional_float,
)
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


def test_episode_recorder_defaults_target_gazebo_small_eval() -> None:
    settings = EpisodeRecorderSettings(episode_id="gazebo_test")

    assert settings.image_topic == "/camera/rgb/image_raw"
    assert settings.camera_info_topic == "/camera/rgb/camera_info"
    assert settings.odom_topic == "/odom"
    assert settings.output_root == "data/episodes/gazebo_small_eval"
    assert settings.episode_id == "gazebo_test"
    assert settings.max_frames == 50


def test_episode_recorder_parses_env_helpers() -> None:
    assert parse_bool("true") is True
    assert parse_bool("0", default=True) is False
    assert parse_bool(None, default=True) is True
    assert parse_optional_float("") is None
    assert parse_optional_float("3.5") == 3.5
